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
    def __init__(self, n_samples, seq_len, n_features, mode='train', anomaly_ratio=0.1):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.mode = mode
        self.anomaly_ratio = anomaly_ratio
        
        # We pre-generate the base parameters but generate actual data on the fly usually
        # But to be consistent with "Lazy Loading" philosophy for synthetic:
        # We can just generate the parameters (freq, phase) here and generate signal in getitem.
        # However, for speed in training, it's often better to pre-gen synthetic data if it fits in RAM.
        # But to strictly follow the plan "Slice from original array", we'll stick to a generated buffer.
        self.total_len = n_samples + seq_len # Buffer size
        self.data_buffer = self._generate_continuous_data()

    def _generate_continuous_data(self):
        # Generate a continuous stream
        t = np.linspace(0, 4 * np.pi * (self.total_len / self.seq_len), self.total_len)
        freq = 1.0 # Fixed base freq for continuous stream coherence or vary it slowly?
        # To match previous behavior of random samples, we might just store random params per sample
        # But "Slice on fly" implies a continuous time series usually.
        # Let's stick to the previous behavior but optimize storage: 
        # Actually previous behavior was independent samples. 
        # If we want independent samples, we don't slice a long array.
        # The prompt asked for "Slice from the original array".
        # Let's assume the user wants a sliding window over a long time series.
        
        samples = []
        # Reverting to independent samples as per original logic to ensure behavior consistency
        # but using __getitem__ to generate them if they are purely synthetic is even more memory efficient
        # (no storage needed!). 
        # But to avoid CPU bottleneck during training, let's pre-generate a massive 1D array and slice it.
        
        # Continuous signal construction
        t = np.arange(self.total_len) * 0.1
        signal = np.sin(t) 
        noise = np.random.normal(0, 0.1, self.total_len)
        data = signal + noise
        return data.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Synthetic data generation on the fly (Hybrid approach)
        # To truly save memory, we calculate sine wave here.
        
        # Previous logic: Random freq/phase per sample.
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        signal = np.sin(freq * t + phase)
        noise = np.random.normal(0, 0.1, self.seq_len)
        sample = signal + noise
        
        label = 0.0
        
        # Inject anomalies
        if self.mode == 'test' and np.random.random() < self.anomaly_ratio:
            if np.random.random() < 0.5:
                # Spike
                rand_idx = np.random.randint(0, self.seq_len)
                sample[rand_idx] += np.random.choice([-1, 1]) * np.random.uniform(2.0, 3.0)
            else:
                # Flatline
                start = np.random.randint(0, self.seq_len // 2)
                end = start + np.random.randint(5, 10)
                sample[start:end] = 0
            label = 1.0

        sample = sample.astype(np.float32).reshape(-1, 1) # (seq_len, n_features)
        return torch.from_numpy(sample), torch.tensor(label).float()

class NYCTaxiDataset(Dataset):
    """
    Dataset for NYC Taxi Trip Data (Numenta Anomaly Benchmark).
    Uses Lazy Loading (slicing on-the-fly).
    """
    URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
    
    def __init__(self, seq_len, mode='train', train_split=0.8):
        self.seq_len = seq_len
        self.mode = mode
        
        # Download and load data
        print("Downloading/Loading NYC Taxi Data...")
        s = requests.get(self.URL).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        # Preprocessing
        values = df['value'].values.reshape(-1, 1).astype(np.float32)
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        
        # Split
        split_idx = int(len(values_scaled) * train_split)
        
        if mode == 'train':
            self.data_raw = values_scaled[:split_idx]
        else:
            self.data_raw = values_scaled[split_idx:]
            
        # Ensure we have enough data
        if len(self.data_raw) <= seq_len:
            raise ValueError("Dataset too small for given sequence length")

    def __len__(self):
        return len(self.data_raw) - self.seq_len

    def __getitem__(self, idx):
        # Lazy Loading: Slice directly from the raw data
        # This avoids storing N copies of the data
        seq = self.data_raw[idx : idx + self.seq_len].copy() # Copy is needed to not modify the original source if we inject
        
        label = 0.0
        
        # Inject anomalies on-the-fly for test set
        if self.mode == 'test' and np.random.random() < 0.15: # 15% anomaly ratio
            # Type 1: Spike
            if np.random.random() < 0.4:
                rand_idx = np.random.randint(0, self.seq_len)
                seq[rand_idx] += np.random.choice([-1, 1]) * np.random.uniform(6.0, 8.0) 
            
            # Type 2: Drop
            elif np.random.random() < 0.7:
                start = np.random.randint(0, self.seq_len // 2)
                end = start + np.random.randint(10, 20)
                seq[start:end] = -2.0 
            
            # Type 3: Noise
            else:
                start = np.random.randint(0, self.seq_len // 2)
                end = start + np.random.randint(10, 20)
                seq[start:end] += np.random.normal(0, 2.0, size=(end-start, 1)).astype(np.float32)

            label = 1.0
            
        return torch.from_numpy(seq), torch.tensor(label).float()

def get_dataloaders(dataset_name='synthetic', batch_size=32, seq_len=64, n_train=1000, n_test=200, num_workers=0, pin_memory=False):
    """
    Returns train and test dataloaders with parallel loading support.
    """
    if dataset_name == 'nyc_taxi':
        train_dataset = NYCTaxiDataset(seq_len=seq_len, mode='train')
        test_dataset = NYCTaxiDataset(seq_len=seq_len, mode='test')
    else:
        # Defaults for synthetic arguments are handled by caller or config now
        train_dataset = TimeSeriesDataset(n_samples=n_train, seq_len=seq_len, n_features=1, mode='train')
        test_dataset = TimeSeriesDataset(n_samples=n_test, seq_len=seq_len, n_features=1, mode='test', anomaly_ratio=0.2)
    
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
