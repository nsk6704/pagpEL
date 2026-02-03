import torch
import torch.multiprocessing as mp
import numpy as np
import time
from data_loader import get_dataloaders
from models import get_model
from trainer import train_model

def train_worker(rank, config, return_dict, status_queue):
    """
    Worker function for multiprocessing.
    """
    model_name = config['model_name']
    epochs = config['epochs']
    batch_size = config['batch_size']
    seq_len = config['seq_len']
    n_features = config['n_features']
    device = config['device']
    use_amp = config.get('use_amp', False)
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', False)
    
    # Resolve DML device if needed
    if device == 'dml':
        import torch_directml
        device = torch_directml.device()
    
    # Signal start
    status_queue.put({
        'rank': rank,
        'model': model_name,
        'status': 'training',
        'duration': 0.0,
        'device': device
    })
    
    # Each worker gets its own dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, 
        seq_len=seq_len,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Initialize model
    model = get_model(model_name, seq_len, n_features)
    
    # Train
    try:
        start_time = time.time()
        state_dict, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=epochs, 
            device=device, 
            model_name=f"W-{rank}-{model_name}",
            use_amp=use_amp
        )
        end_time = time.time()
        duration = end_time - start_time
        
        # Signal completion
        status_queue.put({
            'rank': rank,
            'model': model_name,
            'status': 'success',
            'duration': duration,
            'throughput': float(np.mean(history['throughput'])) if history.get('throughput') else 0
        })
        
        # IMPORTANT: Move state_dict to CPU before returning!
        # DirectML tokens cannot be serialized across processes easily.
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        return_dict[rank] = {
            'status': 'success',
            'model_state': cpu_state_dict,
            'history': history,
            'config': config,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }
    except Exception as e:
        status_queue.put({
            'rank': rank,
            'model': model_name,
            'status': 'failed',
            'duration': 0.0
        })
        
        return_dict[rank] = {
            'status': 'error',
            'error': str(e),
            'start_time': time.time(),
            'end_time': time.time(),
            'duration': 0
        }

class ParallelTrainer:
    def __init__(self, configs):
        """
        configs: List of dictionaries, each containing config for one model.
        """
        self.configs = configs

    def run(self, status_callback=None):
        """
        Spawns processes to train models in parallel.
        Respects max_workers to avoid bombing the system.
        """
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Method already set

        manager = mp.Manager()
        return_dict = manager.dict()
        status_queue = manager.Queue()
        
        # Determine max workers
        import os
        config_limit = self.configs[0].get('max_concurrent_workers') if self.configs else None
        max_workers = config_limit or max(1, os.cpu_count() - 1) # Leave one core for OS/UI
        
        print(f"Starting parallel training with {len(self.configs)} tasks (Max Concurrent: {max_workers})...")
        
        processes = []
        task_queue = list(enumerate(self.configs))
        active_pids = []
        
        while task_queue or active_pids:
            # Spawn new workers if slots available
            while task_queue and len(active_pids) < max_workers:
                rank, config = task_queue.pop(0)
                p = mp.Process(target=train_worker, args=(rank, config, return_dict, status_queue))
                p.start()
                active_pids.append(p)
            
            # Check status
            while not status_queue.empty():
                msg = status_queue.get()
                if status_callback:
                    status_callback(msg)
            
            # Clean up finished processes
            still_active = []
            for p in active_pids:
                if p.is_alive():
                    still_active.append(p)
                else:
                    p.join() # Ensure resource release
            active_pids = still_active
            
            time.sleep(0.1)
            
        print("Parallel training complete.")
        
        # Collect results
        results = []
        for rank in range(len(self.configs)):
            res = return_dict.get(rank)
            if res and res['status'] == 'success':
                results.append(res)
            else:
                err = res.get('error', 'Unknown error') if res else 'No result returned'
                print(f"Worker {rank} failed: {err}")
                
        return results
