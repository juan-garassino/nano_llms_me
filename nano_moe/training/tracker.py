import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

class ExperimentTracker:
    """Tracks all experiment metrics and handles visualization"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.create_directory_structure()

        # Initialize tracking variables
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': {},
            'expert_usage': [],
            'routing_weights': [],
            'learning_rate': [],
            'routing_entropy': [],
            'thinking_depth': []
        }

        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_epoch': 0
        }

    def create_directory_structure(self):
        dirs = ['models', 'plots', 'data', 'logs']
        for dir_path in dirs:
            (self.save_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def log_epoch_metrics(self, epoch: int, train_loss: float, train_acc: float,
                         val_accs: Dict[str, float], expert_gates: torch.Tensor,
                         thinking_depth: float = 0.0, lr: float = 0.0):
        
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['thinking_depth'].append(thinking_depth)

        for dataset, acc in val_accs.items():
            if dataset not in self.metrics['val_acc']:
                self.metrics['val_acc'][dataset] = []
            self.metrics['val_acc'][dataset].append(acc)

        expert_usage = expert_gates.cpu().numpy()
        self.metrics['expert_usage'].append(expert_usage)
        
        entropy = -np.sum(expert_usage * np.log(expert_usage + 1e-8))
        self.metrics['routing_entropy'].append(entropy)

        avg_val_acc = np.mean(list(val_accs.values()))
        if avg_val_acc > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = avg_val_acc
            self.best_metrics['best_epoch'] = epoch

    def save_metrics(self):
        metrics_file = self.save_dir / 'data/training_metrics.json'
        
        # Simple serialization helper
        def convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.number): return float(o)
            return o

        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=convert)
