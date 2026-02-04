import numpy as np
import torch
from models import get_model
from trainer import evaluate_model
from data_loader import get_dataloaders
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class EnsemblePredictor:
    def __init__(self, results):
        """
        results: List of dictionaries from ParallelTrainer.
        """
        self.models = []
        self.configs = []
        
        for res in results:
            config = res['config']
            state_dict = res['model_state']
            
            model = get_model(config['model_name'], config['seq_len'], config['n_features'])
            model.load_state_dict(state_dict)
            model.eval()
            
            self.models.append(model)
            self.configs.append(config)
            
    def evaluate_ensemble(self, test_loader, device='cpu'):
        """
        Aggregates predictions from all models.
        """
        # Resolve DML device if needed
        target_device = device
        if device == 'dml':
            import torch_directml
            target_device = torch_directml.device()
            
        all_scores = []
        true_labels = None
        
        print(f"Evaluating ensemble of {len(self.models)} models on {target_device}...")
        
        for i, model in enumerate(self.models):
            scores, labels = evaluate_model(model, test_loader, device=target_device)
            
            # Normalize scores to [0, 1] range for each model to ensure fair voting
            # Simple Min-Max scaling based on batch statistics
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
            else:
                scores = np.zeros_like(scores)
                
            all_scores.append(scores)
            
            if true_labels is None:
                true_labels = labels
        
        # Aggregate: Mean Voting
        all_scores = np.array(all_scores)
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Calculate Metrics
        auc_roc = roc_auc_score(true_labels, ensemble_scores)
        precision, recall, _ = precision_recall_curve(true_labels, ensemble_scores)
        pr_auc = auc(recall, precision)
        
        return {
            'auc_roc': auc_roc,
            'pr_auc': pr_auc,
            'ensemble_scores': ensemble_scores,
            'true_labels': true_labels
        }
