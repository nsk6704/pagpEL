import torch
import torch.multiprocessing as mp
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
    
    # Signal start
    status_queue.put({
        'rank': rank,
        'model': model_name,
        'status': 'training',
        'duration': 0.0
    })
    
    # Each worker gets its own dataloaders (to avoid pickling issues with some datasets)
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, seq_len=seq_len)
    
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
            model_name=f"Worker-{rank}-{model_name}"
        )
        end_time = time.time()
        duration = end_time - start_time
        
        # Signal completion
        status_queue.put({
            'rank': rank,
            'model': model_name,
            'status': 'success',
            'duration': duration
        })
        
        return_dict[rank] = {
            'status': 'success',
            'model_state': state_dict,
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
        """
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        status_queue = manager.Queue()
        processes = []
        
        print(f"Starting parallel training with {len(self.configs)} workers...")
        
        for rank, config in enumerate(self.configs):
            p = mp.Process(target=train_worker, args=(rank, config, return_dict, status_queue))
            p.start()
            processes.append(p)
            
        # Monitor loop
        active_workers = len(processes)
        
        while active_workers > 0:
            # Check for updates
            while not status_queue.empty():
                msg = status_queue.get()
                if status_callback:
                    status_callback(msg)
                
                if msg['status'] in ['success', 'failed']:
                    active_workers -= 1
            
            # Check if processes are still alive (safety net)
            if not any(p.is_alive() for p in processes) and status_queue.empty():
                break
                
            time.sleep(0.1)
            
        for p in processes:
            p.join()
            
        print("Parallel training complete.")
        
        # Collect results
        results = []
        for rank in range(len(self.configs)):
            res = return_dict.get(rank)
            if res and res['status'] == 'success':
                results.append(res)
            else:
                print(f"Worker {rank} failed: {res.get('error', 'Unknown error')}")
                
        return results
