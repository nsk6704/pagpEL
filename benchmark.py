import torch
import time
import json
import os
import numpy as np
try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False
from data_loader import get_dataloaders
from models import get_model
from trainer import train_model

def run_benchmark():
    """
    Runs a series of benchmarks comparing CPU and GPU (if available).
    """
    results = {
        "hardware": {
            "cpu": "AMD Ryzen 7" if "AMD" in os.popen('wmic cpu get name').read() else "Unknown CPU",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("DirectML AMD GPU" if dml_available else "None"),
            "vram": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
        },
        "benchmarks": []
    }

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if dml_available:
        devices.append('dml')
    
    # Configuration for benchmarks
    # We use extreme params to ensure the GPU cores are fully utilized
    batch_sizes = [128, 512, 1024]
    model_type = 'heavy_cnn' # Highly parallel model
    epochs = 2
    seq_len = 256 # Long sequence to overwhelm CPU caches
    n_features = 1

    print("=== Starting Hardware Benchmark ===")
    
    for device in devices:
        for bs in batch_sizes:
            print(f"\nBenchmarking {model_type.upper()} on {device.upper()} | Batch Size: {bs}")
            
            # Prepare data
            train_loader, val_loader = get_dataloaders(
                batch_size=bs, 
                seq_len=seq_len, 
                n_train=5000, # More data for sustained throughput
                num_workers=0, # Sync loading for purity
                pin_memory=(device == 'cuda' or device == 'dml')
            )
            
            # Use AMP for GPU to show even more gain
            use_amp = (device == 'cuda')
            
            model = get_model(model_type, seq_len, n_features)
            
            # Warmup
            model_device = device
            if dml_available and device == 'dml':
                model_device = torch_directml.device()

            if device == 'cuda':
                dummy_x = torch.randn(bs, seq_len, n_features).to(device)
                for _ in range(5):
                    _ = model.to(device)(dummy_x)
            elif dml_available and device == 'dml':
                dummy_x = torch.randn(bs, seq_len, n_features).to(model_device)
                for _ in range(5):
                    _ = model.to(model_device)(dummy_x)
            
            # Actual Training Benchmark
            _, history = train_model(
                model, 
                train_loader, 
                val_loader, 
                epochs=epochs, 
                device=model_device, 
                model_name=f"Bench-{device}-{bs}",
                use_amp=use_amp
            )
            
            results["benchmarks"].append({
                "device": device,
                "batch_size": bs,
                "avg_throughput": float(np.mean(history['throughput'])),
                "total_time": float(history['train_time']),
                "max_memory": float(history['max_memory']),
                "use_amp": use_amp
            })

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
