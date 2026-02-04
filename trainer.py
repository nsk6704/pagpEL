import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', model_name="model", use_amp=False):
    """
    Trains a single model with performance tracking.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Mixed Precision Setup
    is_cuda = "cuda" in str(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_cuda))
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_time': 0,
        'throughput': [], # samples / sec
        'max_memory': 0     # MB
    }
    
    start_time = time.time()
    total_samples = 0
    
    print(f"[{model_name}] Starting training on {device} (AMP: {use_amp})...")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        epoch_start = time.time()
        epoch_samples = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.cuda.amp.autocast(enabled=(use_amp and is_cuda)):
                output = model(batch_x)
                loss = criterion(output, batch_x)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            epoch_samples += batch_size
            total_samples += batch_size
            
        epoch_duration = time.time() - epoch_start
        throughput = epoch_samples / epoch_duration
        history['throughput'].append(throughput)
        
        # Track GPU memory if on CUDA
        if is_cuda:
            mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            history['max_memory'] = max(history['max_memory'], mem)
            
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                with torch.cuda.amp.autocast(enabled=(use_amp and is_cuda)):
                    output = model(batch_x)
                    loss = criterion(output, batch_x)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{model_name}] Ep {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Throughput: {throughput:.1f} samp/s")
            
    total_time = time.time() - start_time
    history['train_time'] = total_time
    print(f"[{model_name}] Finished in {total_time:.2f}s | Avg Throughput: {np.mean(history['throughput']):.1f} samp/s")
    
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
            try:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                
                # Loss per sample: Mean over features and sequence
                loss = criterion(output, batch_x)
                loss = loss.mean(dim=[1, 2]) # [Batch]
                
                anomaly_scores.extend(loss.cpu().numpy())
                true_labels.extend(batch_y.numpy())
            except Exception as e:
                print(f"Error in evaluate_model on {device}: {str(e)}")
                # Fallback to CPU for this batch if GPU fails during eval
                batch_x_cpu = batch_x.cpu()
                model_cpu = model.cpu()
                output = model_cpu(batch_x_cpu)
                loss = criterion(output, batch_x_cpu).mean(dim=[1, 2])
                anomaly_scores.extend(loss.detach().numpy())
                true_labels.extend(batch_y.numpy())
                # Move model back
                model.to(device)
            
    return np.array(anomaly_scores), np.array(true_labels)
