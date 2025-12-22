import torch
import torch.nn as nn

class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(GRUAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        
        # Encode
        _, hidden = self.encoder(x) # hidden: [1, batch, hidden_dim]
        
        # Repeat hidden state for decoder input
        # [batch, seq_len, hidden_dim]
        decoder_input = hidden.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input)
        
        # Reconstruct
        reconstruction = self.output_layer(decoded)
        return reconstruction

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        
        # Encode
        _, (hidden, _) = self.encoder(x) # hidden: [1, batch, hidden_dim]
        
        # Repeat hidden state for decoder input
        # [batch, seq_len, hidden_dim]
        decoder_input = hidden.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input)
        
        # Reconstruct
        reconstruction = self.output_layer(decoded)
        return reconstruction

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
            nn.Identity() # Linear activation for reconstruction
        )

    def forward(self, x):
        # x: [batch, seq_len, n_features] -> [batch, n_features, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # [batch, n_features, seq_len] -> [batch, seq_len, n_features]
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
        # Flatten
        x_flat = x.view(batch_size, -1)
        
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        
        # Reshape back
        return decoded.view(batch_size, -1, x.size(2))

class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, d_model=64, nhead=4, num_layers=2):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size = x.size(0)
        
        # Embed and add position encoding
        x_emb = self.embedding(x) + self.pos_encoder
        
        # Encode
        memory = self.transformer_encoder(x_emb)
        
        # Decode (using memory as both target and memory for autoencoding)
        # In a true autoencoder, we might mask or use a different target, 
        # but for reconstruction we can just pass the memory through.
        decoded = self.transformer_decoder(x_emb, memory)
        
        return self.output_layer(decoded)

def get_model(model_name, seq_len, n_features):
    if model_name == 'lstm':
        return LSTMAutoencoder(seq_len, n_features)
    elif model_name == 'cnn':
        return CNNAutoencoder(seq_len, n_features)
    elif model_name == 'dense':
        return DenseAutoencoder(seq_len, n_features)
    elif model_name == 'transformer':
        return TransformerAutoencoder(seq_len, n_features)
    elif model_name == 'gru':
        return GRUAutoencoder(seq_len, n_features)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    # Test models
    print("Testing Models...")
    x = torch.randn(32, 64, 1) # [Batch, Seq, Feat]
    
    models = ['lstm', 'cnn', 'dense', 'transformer', 'gru']
    for m_name in models:
        model = get_model(m_name, 64, 1)
        out = model(x)
        print(f"{m_name.upper()} Output Shape: {out.shape}")
    print("Model Test Complete.")
