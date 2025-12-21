import argparse
import torch
import time
from parallel_engine import ParallelTrainer
from ensemble import EnsemblePredictor
from data_loader import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description="Parallel & GPU-Accelerated Anomaly Detection")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--n_features', type=int, default=1, help='Number of features')
    parser.add_argument('--n_models', type=int, default=4, help='Number of models to train in parallel')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = 'cpu'
        
    print(f"Configuration: {args}")
    
    # 1. Define Model Configurations
    # We will mix different architectures
    model_types = ['lstm', 'cnn', 'dense']
    configs = []
    
    for i in range(args.n_models):
        m_type = model_types[i % len(model_types)]
        config = {
            'model_name': m_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'n_features': args.n_features,
            'device': args.device
        }
        configs.append(config)
        
    # 2. Parallel Training
    print("\n=== Starting Parallel Training Phase ===")
    start_time = time.time()
    
    trainer = ParallelTrainer(configs)
    results = trainer.run()
    
    train_time = time.time() - start_time
    print(f"=== Training Phase Complete in {train_time:.2f}s ===")
    
    if not results:
        print("No models were trained successfully. Exiting.")
        return

    # 3. Ensemble Evaluation
    print("\n=== Starting Ensemble Evaluation Phase ===")
    # Load test data (same for all)
    _, test_loader = get_dataloaders(batch_size=args.batch_size, seq_len=args.seq_len)
    
    ensemble = EnsemblePredictor(results)
    metrics = ensemble.evaluate_ensemble(test_loader, device=args.device)
    
    print("\n=== Final Results ===")
    print(f"Ensemble AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Ensemble PR-AUC:  {metrics['pr_auc']:.4f}")
    print("=====================")

if __name__ == "__main__":
    main()
