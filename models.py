import torch
import torch.nn as nn

class CustomGRUCell(nn.Module):
    """
    Manual GRU Cell implementation to avoid DirectML fused kernel issues.
    """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined linear Layer for reset and update gates
        self.x2gate = nn.Linear(input_size, 2 * hidden_size, bias=True)
        self.h2gate = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        
        # Linear layer for candidate hidden state
        self.x2cand = nn.Linear(input_size, hidden_size, bias=True)
        self.h2cand = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        # x: [batch, input_size], h: [batch, hidden_size]
        
        # Gates: Update (z) and Reset (r)
        gates = self.x2gate(x) + self.h2gate(h)
        z_gate, r_gate = gates.chunk(2, 1)
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)
        
        # Candidate hidden state (n)
        n = torch.tanh(self.x2cand(x) + r * self.h2cand(h))
        
        # Final hidden state
        h_next = (1 - z) * n + z * h
        return h_next

class CustomLSTMCell(nn.Module):
    """
    Manual LSTM Cell implementation to avoid DirectML fused kernel issues.
    """
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined linear Layer for all 4 gates
        self.x2gate = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2gate = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, h_c):
        # x: [batch, input_size], h_c: tuple of ([batch, hidden_size], [batch, hidden_size])
        h, c = h_c
        
        gates = self.x2gate(x) + self.h2gate(h)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(GRUAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Primary RNNs (used only for non-DML backends)
        self.encoder_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Manual Cells (Strict fallback for DML stability)
        self.encoder_cell = CustomGRUCell(n_features, hidden_dim)
        self.decoder_cell = CustomGRUCell(hidden_dim, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        device = x.device
        is_dml = "privateuseone" in str(device).lower()
        
        if is_dml:
            # Manual loop to bypass frozen/missing DML seeds
            h = torch.zeros(x.size(0), self.hidden_dim).to(device)
            for t in range(self.seq_len):
                h = self.encoder_cell(x[:, t, :], h)
            hidden = h
        else:
            _, hidden = self.encoder_rnn(x)
            hidden = hidden.squeeze(0)
            
        # Decoder
        decoder_input = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        if is_dml:
            decoded = torch.zeros(x.size(0), self.seq_len, self.hidden_dim).to(device)
            h_dec = hidden
            for t in range(self.seq_len):
                h_dec = self.decoder_cell(decoder_input[:, t, :], h_dec)
                decoded[:, t:t+1, :] = h_dec.unsqueeze(1)
        else:
            decoded, _ = self.decoder_rnn(decoder_input)
            
        return self.output_layer(decoded)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.encoder_rnn = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.encoder_cell = CustomLSTMCell(n_features, hidden_dim)
        self.decoder_cell = CustomLSTMCell(hidden_dim, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        device = x.device
        is_dml = "privateuseone" in str(device).lower()
        
        if is_dml:
            h = torch.zeros(x.size(0), self.hidden_dim).to(device)
            c = torch.zeros(x.size(0), self.hidden_dim).to(device)
            for t in range(self.seq_len):
                h, c = self.encoder_cell(x[:, t, :], (h, c))
            hidden = (h, c)
        else:
            _, (h, c) = self.encoder_rnn(x)
            hidden = (h.squeeze(0), c.squeeze(0))
            
        # Decoder
        h_h, _ = hidden
        decoder_input = h_h.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        if is_dml:
            decoded = torch.zeros(x.size(0), self.seq_len, self.hidden_dim).to(device)
            h_dec, c_dec = hidden
            for t in range(self.seq_len):
                h_dec, c_dec = self.decoder_cell(decoder_input[:, t, :], (h_dec, c_dec))
                decoded[:, t:t+1, :] = h_dec.unsqueeze(1)
        else:
            decoded, _ = self.decoder_rnn(decoder_input)
            
        return self.output_layer(decoded)

class CNNAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(CNNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, n_features, kernel_size=2, stride=2),
            nn.Identity()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1)

class HeavyCNNAutoencoder(nn.Module):
    """
    Much heavier CNN model to force GPU utilization and show speedup over CPU.
    """
    def __init__(self, seq_len, n_features):
        super(HeavyCNNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, n_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # [batch, seq, feat] -> [batch, feat, seq]
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Ensure output length matches input length if seq_len is odd
        if decoded.size(2) != self.seq_len:
            decoded = torch.nn.functional.interpolate(decoded, size=self.seq_len)
            
        return decoded.permute(0, 2, 1)

class DenseAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(DenseAutoencoder, self).__init__()
        self.input_dim = seq_len * n_features
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, -1, x.size(2))

class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, d_model=64, nhead=4, num_layers=2):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, n_features)

    def forward(self, x):
        x_emb = self.embedding(x) + self.pos_encoder
        memory = self.transformer_encoder(x_emb)
        decoded = self.transformer_decoder(x_emb, memory)
        return self.output_layer(decoded)

def get_model(model_name, seq_len, n_features):
    if model_name == 'lstm': return LSTMAutoencoder(seq_len, n_features)
    elif model_name == 'cnn': return CNNAutoencoder(seq_len, n_features)
    elif model_name == 'dense': return DenseAutoencoder(seq_len, n_features)
    elif model_name == 'transformer': return TransformerAutoencoder(seq_len, n_features)
    elif model_name == 'heavy_cnn': return HeavyCNNAutoencoder(seq_len, n_features)
    elif model_name == 'gru': return GRUAutoencoder(seq_len, n_features)
    else: raise ValueError(f"Unknown model: {model_name}")
