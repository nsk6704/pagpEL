import torch
import time
import json
import os
import numpy as np
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False
from data_loader import get_dataloaders
from models import get_model
from trainer import train_model

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def run_benchmark():
    """
    Runs a series of benchmarks comparing CPU and GPU (if available).
    """
    config = load_config()
    
    # Dynamic Hardware Detection
    cpu_name = "Unknown CPU"
    if cpuinfo:
        cpu_name = cpuinfo.get_cpu_info().get('brand_raw', "Unknown CPU")
    elif "AMD" in os.popen('wmic cpu get name').read():
         cpu_name = "AMD Ryzen (Fallback Detection)"
    
    gpu_name = "None"
    vram = "N/A"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    elif dml_available:
        gpu_name = "DirectML GPU (AMD/Intel/NVIDIA)"
        vram = "Shared (DirectML)"

    results = {
        "hardware": {
            "cpu": cpu_name,
            "gpu": gpu_name,
            "vram": vram
        },
        "benchmarks": []
    }
    
    print("=== Starting Hardware Benchmark ===")
    print(f"System: {cpu_name} | {gpu_name}")

    benchmarks = config.get('benchmarks', [])
    
    for bench_config in benchmarks:
        device = bench_config['device']
        
        # Skip unavailable devices
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Skipping CUDA benchmark (not available)")
            continue
        if device == 'dml' and not dml_available:
            print(f"Skipping DirectML benchmark (not available)")
            continue
            
        model_type = bench_config['model_type']
        epochs = bench_config['epochs']
        seq_len = bench_config['seq_len']
        n_features = bench_config['n_features']
        use_amp = bench_config.get('use_amp', False)
        
        for bs in bench_config['batch_sizes']:
            print(f"\nBenchmarking {model_type.upper()} on {device.upper()} | Batch Size: {bs}")
            
            # Prepare data
            # Adjust n_train dynamically if needed or keep static for consistency across internal runs
            train_loader, val_loader = get_dataloaders(
                batch_size=bs, 
                seq_len=seq_len, 
                n_train=5000, 
                num_workers=0, # Sync execution for accurate timing
                pin_memory=(device == 'cuda' or device == 'dml')
            )
            
            try:
                model = get_model(model_type, seq_len, n_features)
                
                # Device setup
                model_device = device
                if dml_available and device == 'dml':
                    model_device = torch_directml.device()

                # Warmup
                print("  Warming up...", end="\r")
                if device == 'cuda':
                    dummy_x = torch.randn(bs, seq_len, n_features).to(device)
                    for _ in range(3):
                        _ = model.to(device)(dummy_x)
                elif dml_available and device == 'dml':
                    dummy_x = torch.randn(bs, seq_len, n_features).to(model_device)
                    for _ in range(3):
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
                
                avg_throughput = float(np.mean(history['throughput']))
                print(f"  Result: {avg_throughput:.2f} samples/sec")
                
                results["benchmarks"].append({
                    "device": device,
                    "batch_size": bs,
                    "avg_throughput": avg_throughput,
                    "total_time": float(history['train_time']),
                    "max_memory": float(history['max_memory']),
                    "use_amp": use_amp
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM Error: Batch size {bs} too large for {device}")
                    torch.cuda.empty_cache()
                else:
                    print(f"  Error: {e}")

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
