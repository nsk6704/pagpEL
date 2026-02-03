from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import torch
import os
import numpy as np
from typing import List, Optional
import time
import asyncio
import sys
import json
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False

from parallel_engine import ParallelTrainer
from ensemble import EnsemblePredictor
from data_loader import get_dataloaders

app = FastAPI(title="Anomaly Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
class GlobalState:
    training_active = False
    progress = 0
    results = None
    ensemble_metrics = None
    logs = []

state = GlobalState()

# Load Defaults from Config
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except:
        return {}

default_conf = load_config()
training_defaults = default_conf.get('training', {})

# Models
class TrainConfig(BaseModel):
    dataset: str = "nyc_taxi" 
    epochs: int = training_defaults.get('default_epochs', 5)
    batch_size: int = training_defaults.get('batch_size', 32)
    seq_len: int = 64
    n_models: int = 5
    device: str = "cpu" # cpu, cuda, or dml
    use_amp: bool = False

class WorkerStatus(BaseModel):
    rank: int
    model: str
    status: str
    duration: float
    device: Optional[str] = "cpu"
    throughput: Optional[float] = 0.0

class TrainingStatus(BaseModel):
    active: bool
    progress: float
    logs: List[str]
    workers: List[WorkerStatus] = []

def log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    state.logs.append(f"[{timestamp}] {message}")

# Background Task
def run_training_pipeline(config: TrainConfig):
    state.training_active = True
    state.progress = 0
    state.logs = []
    state.results = None
    state.ensemble_metrics = None
    
    try:
        log(f"Starting pipeline on {str(config.device)} (AMP: {config.use_amp})...")
        state.progress = 10
        
        # 1. Prepare Data
        # Ensure we pass all required args to get_dataloaders
        get_dataloaders(dataset_name=config.dataset, batch_size=config.batch_size, seq_len=config.seq_len)
        log("Data ready.")
        state.progress = 20
        
        # 2. Configure Models
        model_types = ['lstm', 'cnn', 'dense', 'transformer', 'gru']
        configs = []
        
        # Log to terminal for debugging
        print(f"DEBUG: Internal Resolve - config.device: {config.device}, dml_available: {dml_available}")

        for i in range(config.n_models):
            m_type = model_types[i % len(model_types)]
            c = {
                'model_name': m_type,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'seq_len': config.seq_len,
                'n_features': 1,
                'device': config.device, # Pass the string
                'use_amp': config.use_amp if config.device == "cuda" else False,
                'num_workers': 0, 
                'pin_memory': True if config.device == "cuda" else False
            }
            configs.append(c)
            
        # 3. Parallel Training
        log(f"Spawning {config.n_models} parallel models...")
        trainer = ParallelTrainer(configs)
        
        # Initialize workers in state
        state.workers = []
        for i, c in enumerate(configs):
            state.workers.append({
                'rank': i,
                'model': c['model_name'],
                'status': 'pending',
                'duration': 0.0,
                'device': str(c['device'])
            })
        
        def worker_callback(msg):
            for w in state.workers:
                if w['rank'] == msg['rank']:
                    w['status'] = msg['status']
                    w['duration'] = msg['duration']
                    if 'throughput' in msg:
                        w['throughput'] = msg['throughput']
                    break
        
        state.progress = 30
        results = trainer.run(status_callback=worker_callback)
        state.progress = 80
        log("Ensemble training complete.")
        
        if not results:
            log("Error: No models trained.")
            state.training_active = False
            return
 
        state.results = results
        
        # 4. Ensemble Evaluation
        log("Evaluating ensemble metrics...")
        _, test_loader = get_dataloaders(dataset_name=config.dataset, batch_size=config.batch_size, seq_len=config.seq_len)
        
        ensemble = EnsemblePredictor(results)
        metrics = ensemble.evaluate_ensemble(test_loader, device=config.device)
        
        state.ensemble_metrics = metrics
        log(f"Pipeline finished. AUC: {metrics['auc_roc']:.4f}")
        state.progress = 100
        
    except Exception as e:
        log(f"Error: {str(e)}")
    finally:
        state.training_active = False

@app.post("/train")
async def start_training(config: TrainConfig):
    if state.training_active:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Run in background thread
    t = threading.Thread(target=run_training_pipeline, args=(config,))
    t.start()
    
    return {"message": "Training started"}

@app.get("/status", response_model=TrainingStatus)
async def get_status():
    return {
        "active": state.training_active,
        "progress": state.progress,
        "logs": state.logs[-10:], 
        "workers": getattr(state, 'workers', [])
    }

async def status_event_generator(request):
    while True:
        if await request.is_disconnected():
            break
            
        status_data = {
            "active": state.training_active,
            "progress": state.progress,
            "logs": state.logs[-10:],
            "workers": getattr(state, 'workers', [])
        }
        
        yield {"data": json.dumps(status_data)}
        await asyncio.sleep(0.5)

@app.get("/stream-status")
async def stream_status(request: Request):
    return EventSourceResponse(status_event_generator(request))

@app.get("/results")
async def get_results():
    if not state.ensemble_metrics:
        return {"ready": False}
    
    scores = state.ensemble_metrics['ensemble_scores']
    labels = state.ensemble_metrics['true_labels']
    
    if len(scores) > 1000:
        indices = np.linspace(0, len(scores)-1, 1000).astype(int)
        scores = scores[indices]
        labels = labels[indices]
        
    def sanitize(val):
        if isinstance(val, (float, np.float32, np.float64)) and (np.isnan(val) or np.isinf(val)):
            return 0.0
        return float(val)

    return {
        "ready": True,
        "metrics": {
            "auc_roc": sanitize(state.ensemble_metrics['auc_roc']),
            "pr_auc": sanitize(state.ensemble_metrics['pr_auc'])
        },
        "plot_data": {
            "scores": [sanitize(s) for s in scores.tolist()],
            "labels": labels.tolist()
        }
    }

@app.get("/sample-data")
async def get_sample_data(dataset: str = "nyc_taxi"):
    try:
        _, test_loader = get_dataloaders(dataset_name=dataset, batch_size=200, seq_len=64, n_test=200)
        data, labels = next(iter(test_loader))
        
        normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
        anomaly_idx = (labels == 1).nonzero(as_tuple=True)[0]
        
        sample_normal = data[normal_idx[0]].numpy().flatten().tolist() if len(normal_idx) > 0 else []
        sample_anomaly = data[anomaly_idx[0]].numpy().flatten().tolist() if len(anomaly_idx) > 0 else []
        
        return {
            "normal": sample_normal,
            "anomaly": sample_anomaly
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hardware")
async def get_hardware_info():
    import subprocess
    
    def get_ps_output(cmd):
        try:
            return subprocess.check_output(["powershell", "-Command", cmd]).decode().strip()
        except:
            return "Unknown"

    cpu_name = get_ps_output("(Get-CimInstance Win32_Processor).Name")
    gpu_name_ps = get_ps_output("(Get-CimInstance Win32_VideoController).Name")
    
    cuda_available = torch.cuda.is_available()
    
    info = {
        "cpu": cpu_name if cpu_name != "Unknown" else "CPU",
        "cuda_available": cuda_available,
        "dml_available": dml_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else (gpu_name_ps if gpu_name_ps != "Unknown" else "N/A"),
        "vram_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if cuda_available else "N/A"
    }
    return info

@app.post("/benchmark")
async def trigger_benchmark():
    import subprocess
    # Run standalone benchmark script
    try:
        subprocess.Popen([sys.executable, "benchmark.py"])
        return {"message": "Benchmark started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/results")
async def get_benchmark_results():
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", 'r') as f:
            return json.load(f)
    return {"message": "No results found"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
