import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', model_name="model"):
    """
    Trains a single model.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_time': 0}
    
    start_time = time.time()
    
    print(f"[{model_name}] Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_x)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
    total_time = time.time() - start_time
    history['train_time'] = total_time
    print(f"[{model_name}] Training finished in {total_time:.2f}s")
    
    return model.state_dict(), history

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates model and returns reconstruction errors (anomaly scores).
    """
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    anomaly_scores = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            
            # Loss per sample: Mean over features and sequence
            loss = criterion(output, batch_x)
            loss = loss.mean(dim=[1, 2]) # [Batch]
            
            anomaly_scores.extend(loss.cpu().numpy())
            true_labels.extend(batch_y.numpy())
            
    return np.array(anomaly_scores), np.array(true_labels)
