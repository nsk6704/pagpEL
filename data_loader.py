import numpy as np
import torch
import pandas as pd
import requests
import io
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for Time Series Anomaly Detection.
    Generates synthetic sine/cosine waves with optional noise and anomalies.
    """
    def __init__(self, n_samples=1000, seq_len=64, n_features=1, mode='train', anomaly_ratio=0.1):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.mode = mode
        self.anomaly_ratio = anomaly_ratio
        
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        
        for _ in range(self.n_samples):
            # Base signal: Sine wave with random phase and frequency
            t = np.linspace(0, 4 * np.pi, self.seq_len)
            freq = np.random.uniform(0.8, 1.2)
            phase = np.random.uniform(0, 2 * np.pi)
            signal = np.sin(freq * t + phase)
            
            # Add noise
            noise = np.random.normal(0, 0.1, self.seq_len)
            sample = signal + noise
            
            label = 0 # Normal
            
            # Inject anomalies for test set
            if self.mode == 'test' and np.random.random() < self.anomaly_ratio:
                # Type 1: Point anomaly (spike)
                if np.random.random() < 0.5:
                    idx = np.random.randint(0, self.seq_len)
                    sample[idx] += np.random.choice([-1, 1]) * np.random.uniform(2.0, 3.0)
                # Type 2: Contextual anomaly (flatline or noise burst)
                else:
                    start = np.random.randint(0, self.seq_len // 2)
                    end = start + np.random.randint(5, 10)
                    sample[start:end] = 0 # Flatline
                
                label = 1 # Anomaly
            
            data.append(sample.reshape(-1, 1)) # (seq_len, n_features)
            labels.append(label)
            
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

class NYCTaxiDataset(Dataset):
    """
    Dataset for NYC Taxi Trip Data (Numenta Anomaly Benchmark).
    """
    URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
    
    def __init__(self, seq_len=64, mode='train', train_split=0.8):
        self.seq_len = seq_len
        self.mode = mode
        
        # Download and load data
        print("Downloading/Loading NYC Taxi Data...")
        s = requests.get(self.URL).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        # Preprocessing
        values = df['value'].values.reshape(-1, 1)
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        
        # Split
        split_idx = int(len(values_scaled) * train_split)
        
        if mode == 'train':
            self.data_raw = values_scaled[:split_idx]
        else:
            self.data_raw = values_scaled[split_idx:]
            
        self.data, self.labels = self._create_sequences(self.data_raw)

    def _create_sequences(self, data_raw):
        data = []
        labels = []
        
        for i in range(len(data_raw) - self.seq_len):
            seq = data_raw[i:i+self.seq_len].copy() # Copy to avoid modifying original
            label = 0
            
            # Inject anomalies for test set
            if self.mode == 'test' and np.random.random() < 0.15: # 15% anomaly ratio
                # Type 1: Point anomaly (Extreme Spike)
                if np.random.random() < 0.4:
                    idx = np.random.randint(0, self.seq_len)
                    # Make it a very obvious spike (6-8 sigma)
                    seq[idx] += np.random.choice([-1, 1]) * np.random.uniform(6.0, 8.0) 
                
                # Type 2: Collective Anomaly (Sudden Drop/Zero)
                elif np.random.random() < 0.7:
                    start = np.random.randint(0, self.seq_len // 2)
                    end = start + np.random.randint(10, 20)
                    seq[start:end] = -2.0 # Drop to near zero (assuming standardized data)
                
                # Type 3: Noise Burst
                else:
                    start = np.random.randint(0, self.seq_len // 2)
                    end = start + np.random.randint(10, 20)
                    seq[start:end] += np.random.normal(0, 2.0, end-start).reshape(-1, 1)

                label = 1
            
            data.append(seq)
            labels.append(label)
            
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

def get_dataloaders(dataset_name='synthetic', batch_size=32, seq_len=64, n_train=1000, n_test=200, num_workers=0, pin_memory=False):
    """
    Returns train and test dataloaders with parallel loading support.
    """
    if dataset_name == 'nyc_taxi':
        train_dataset = NYCTaxiDataset(seq_len=seq_len, mode='train')
        test_dataset = NYCTaxiDataset(seq_len=seq_len, mode='test')
    else:
        train_dataset = TimeSeriesDataset(n_samples=n_train, seq_len=seq_len, mode='train')
        test_dataset = TimeSeriesDataset(n_samples=n_test, seq_len=seq_len, mode='test', anomaly_ratio=0.2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    train_loader, test_loader = get_dataloaders()
    
    for batch_data, batch_labels in train_loader:
        print(f"Train Batch Shape: {batch_data.shape}") # [Batch, Seq_Len, Features]
        break
        
    for batch_data, batch_labels in test_loader:
        print(f"Test Batch Shape: {batch_data.shape}")
        print(f"Anomaly Count in Batch: {batch_labels.sum().item()}")
        break
    print("Data Loader Test Complete.")
