<!-- markdownlint-disable-next-line -->
<p align="center">
  <img src="" alt="AnomalyDetect" width="120" />
  <br />
  <h1>AnomalyDetect</h1>
  <p>High-Performance Anomaly Detection Framework</p>
  <p>
    <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c?style=flat&logo=pytorch" alt="PyTorch" />
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License" />
  </p>
</p>

---

AnomalyDetect is a production-ready anomaly detection framework built for speed and scale. It combines parallel model training, multi-architecture ensembles, and GPU acceleration into a single, easy-to-use package.

## Why AnomalyDetect?

Building robust anomaly detectors is hard. Building fast ones shouldn't be.

- **5 Models, 1 Click** — Train LSTM, CNN, Dense, Transformer, and GRU simultaneously
- **GPU-Native** — Full CUDA and DirectML support out of the box
- **Intelligent Fallback** — Automatically falls back to CPU if GPU training fails
- **Production-Ready** — REST API + Web UI included
- **Open Source** — MIT licensed, community driven

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run the web interface
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

Or use the CLI:

```bash
python main.py --epochs 10 --batch_size 32 --n_models 5 --device dml
```

## Python API

```python
from parallel_engine import ParallelTrainer
from ensemble import EnsemblePredictor
from data_loader import get_dataloaders

# Load your data
train_loader, test_loader = get_dataloaders("synthetic", batch_size=32, seq_len=64)

# Train 5 models in parallel
trainer = ParallelTrainer(n_models=5, device="dml")
results = trainer.train(model_configs, train_loader, val_loader)

# Ensemble predictions
ensemble = EnsemblePredictor(results)
metrics = ensemble.evaluate_ensemble(test_loader, device="dml")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
```

## Features

| Feature | Description |
|---------|-------------|
| **Parallel Training** | Train multiple models simultaneously using multiprocessing |
| **Multi-Architecture Ensemble** | LSTM, CNN, Dense, Transformer, GRU vote together |
| **GPU Acceleration** | Native CUDA (NVIDIA) and DirectML (AMD) support |
| **Auto-Fallback** | Seamlessly falls back to CPU on GPU errors |
| **Real-Time Dashboard** | Live training progress and metrics via React UI |
| **REST API** | Programmatic access for integration into your apps |

## Supported Hardware

| Device | Use Case |
|--------|---------|
| CPU | Compatible everywhere |
| CUDA | NVIDIA GPUs |
| DirectML | AMD GPUs (Windows) |

## Architecture

```
AnomalyDetect/
├── main.py              # CLI entry point
├── server.py           # FastAPI server + Web UI
├── parallel_engine.py   # Distributed training
├── ensemble.py        # Ensemble voting
├── models.py          # Model zoo (LSTM, CNN, Dense, Transformer, GRU)
├── data_loader.py    # Data pipelines
├── trainer.py        # Training utilities
├── benchmark.py      # Performance benchmarking
├── config.json      # Configuration
├── frontend/        # React dashboard
└── requirements.txt
```

## Configuration

Customize behavior in `config.json`:

```json
{
  "system": {
    "use_gpu": true,
    "use_dml": true,
    "num_workers_ratio": 0.75
  },
  "training": {
    "default_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config` | GET | Current configuration |
| `/config` | POST | Update configuration |
| `/train` | POST | Start training job |
| `/results` | GET | Training results |
| `/logs` | GET | Live training logs (SSE) |
| `/stop` | POST | Stop active training |

## Requirements

- Python 3.11+
- PyTorch 2.0+
- (Optional) torch-directml for AMD GPU support

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>AnomalyDetect</strong> — Find anomalies before they find you.
</p>