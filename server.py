from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import torch
import numpy as np
from typing import List, Optional
import time
import asyncio
import json
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

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

# Models
class TrainConfig(BaseModel):
    dataset: str = "nyc_taxi" # synthetic or nyc_taxi
    epochs: int = 5
    batch_size: int = 32
    seq_len: int = 64
    n_models: int = 5
    device: str = "cpu" # cpu or cuda

class WorkerStatus(BaseModel):
    rank: int
    model: str
    status: str
    duration: float

class TrainingStatus(BaseModel):
    active: bool
    progress: float
    logs: List[str]
    workers: List[WorkerStatus] = []

class Metrics(BaseModel):
    auc_roc: float
    pr_auc: float

# Background Task
def run_training_pipeline(config: TrainConfig):
    state.training_active = True
    state.progress = 0
    state.logs = []
    state.results = None
    state.ensemble_metrics = None
    
    try:
        log(f"Starting training on {config.dataset} dataset...")
        state.progress = 10
        
        # 1. Prepare Data
        # We just check if we can load it
        get_dataloaders(dataset_name=config.dataset, batch_size=config.batch_size, seq_len=config.seq_len)
        log("Data loaded successfully.")
        state.progress = 20
        
        # 2. Configure Models
        model_types = ['lstm', 'cnn', 'dense', 'transformer', 'gru']
        configs = []
        for i in range(config.n_models):
            m_type = model_types[i % len(model_types)]
            c = {
                'model_name': m_type,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'seq_len': config.seq_len,
                'n_features': 1,
                'device': config.device
            }
            configs.append(c)
            
        # 3. Parallel Training
        log(f"Spawning {config.n_models} workers on {config.device}...")
        trainer = ParallelTrainer(configs)
        
        # Initialize workers in state
        state.workers = []
        for i, c in enumerate(configs):
            state.workers.append({
                'rank': i,
                'model': c['model_name'],
                'status': 'pending',
                'duration': 0.0
            })
        
        def worker_callback(msg):
            # Update worker status in real-time
            for w in state.workers:
                if w['rank'] == msg['rank']:
                    w['status'] = msg['status']
                    w['duration'] = msg['duration']
                    break
        
        state.progress = 30
        results = trainer.run(status_callback=worker_callback)
        state.progress = 80
        log("Training complete.")
        
        if not results:
            log("Error: No models trained successfully.")
            state.training_active = False
            return

        state.results = results
        
        # 4. Ensemble Evaluation
        log("Evaluating ensemble...")
        _, test_loader = get_dataloaders(dataset_name=config.dataset, batch_size=config.batch_size, seq_len=config.seq_len)
        
        ensemble = EnsemblePredictor(results)
        metrics = ensemble.evaluate_ensemble(test_loader, device=config.device)
        
        state.ensemble_metrics = metrics
        log(f"Evaluation Complete. AUC: {metrics['auc_roc']:.4f}")
        state.progress = 100
        
    except Exception as e:
        log(f"Error: {str(e)}")
        print(f"Pipeline Error: {e}")
    finally:
        state.training_active = False

def log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    state.logs.append(f"[{timestamp}] {message}")

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
        "logs": state.logs[-10:], # Return last 10 logs
        "workers": getattr(state, 'workers', [])
    }

# SSE Generator
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
        await asyncio.sleep(0.5) # Update every 500ms

@app.get("/stream-status")
async def stream_status(request: Request):
    return EventSourceResponse(status_event_generator(request))

@app.get("/results")
async def get_results():
    if not state.ensemble_metrics:
        return {"ready": False}
    
    # Convert numpy arrays to lists for JSON serialization
    # We'll downsample the plot data if it's too large
    scores = state.ensemble_metrics['ensemble_scores']
    labels = state.ensemble_metrics['true_labels']
    
    # Downsample for frontend performance (max 1000 points)
    if len(scores) > 1000:
        indices = np.linspace(0, len(scores)-1, 1000).astype(int)
        scores = scores[indices]
        labels = labels[indices]
        
    # Sanitize NaN values for JSON
    def sanitize(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return 0.0
        return val

    return {
        "ready": True,
        "metrics": {
            "auc_roc": sanitize(float(state.ensemble_metrics['auc_roc'])),
            "pr_auc": sanitize(float(state.ensemble_metrics['pr_auc']))
        },
        "plot_data": {
            "scores": [sanitize(s) for s in scores.tolist()],
            "labels": labels.tolist()
        }
    }

@app.get("/sample-data")
async def get_sample_data(dataset: str = "nyc_taxi"):
    """
    Returns a sample of the raw data (normal vs anomaly) for visualization.
    """
    try:
        # Load a small batch
        _, test_loader = get_dataloaders(dataset_name=dataset, batch_size=200, seq_len=64, n_test=200)
        
        # Get one batch
        data, labels = next(iter(test_loader))
        
        # Find a normal sample and an anomaly sample
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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
