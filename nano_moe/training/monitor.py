import os
import json
import time
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()

def info(msg, **kw): console.print(Panel(msg, **({"border_style":"cyan","title":"Info","box":box.ROUNDED} | kw)))
def ok(msg, **kw):   console.print(Panel(msg, **({"border_style":"green","title":"OK","box":box.ROUNDED} | kw)))

class TrainingMonitor:
    """Comprehensive training monitor that saves everything organized."""
    def __init__(self, experiment_name="nano_moe_experiment"):
        self.experiment_name = experiment_name
        self.base_dir = Path("training_outputs") / experiment_name
        self._setup_folders()
        # Tracking
        self.metrics = defaultdict(list)
        self.attention_snapshots = []
        self.embedding_snapshots = []
        self.moe_snapshots = []
        self.step_count = 0
        self.epoch_count = 0

    def _setup_folders(self):
        folders = [
            self.base_dir,
            self.base_dir / "models",
            self.base_dir / "plots" / "training_curves",
            self.base_dir / "plots" / "attention_analysis",
            self.base_dir / "plots" / "embedding_evolution",
            self.base_dir / "plots" / "moe_analysis",
            self.base_dir / "plots" / "animations",
            self.base_dir / "logs",
        ]
        for f in folders: f.mkdir(parents=True, exist_ok=True)
        info(f"Output structure at: {self.base_dir}")

    def log_training_step(self, epoch, step, train_loss, val_loss=None, lr=None, grad_norm=None):
        self.metrics['epoch'].append(epoch)
        self.metrics['step'].append(self.step_count)
        self.metrics['train_loss'].append(float(train_loss))
        if val_loss is not None: self.metrics['val_loss'].append(float(val_loss))
        if lr is not None: self.metrics['learning_rate'].append(float(lr))
        if grad_norm is not None: self.metrics['grad_norm'].append(float(grad_norm))
        self.step_count += 1

    def capture_model_state(self, model):
        # Capture embedding stats if available
        if hasattr(model, 'embed') and hasattr(model.embed, 'freq_basis'):
            # For SFPT/Phase models
            with torch.no_grad():
                emb = model.embed.freq_basis.data.detach().cpu().numpy()
            self.embedding_snapshots.append({
                'step': self.step_count,
                'epoch': self.epoch_count,
                'norm': float(np.linalg.norm(emb)),
                'mean': float(np.mean(emb)),
                'std': float(np.std(emb)),
            })
        
        # Capture Phase Operator stats if available
        if hasattr(model, 'program_synth') and hasattr(model.program_synth, 'library'):
            # Capture operator usage or weights
            pass

    def _plot_training_curves(self):
        if not self.metrics['train_loss']:
            return
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        steps = np.array(self.metrics['step'])
        train_losses = np.array(self.metrics['train_loss'])

        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(steps, train_losses, linewidth=2, alpha=0.9, label='Training Loss')
        if self.metrics.get('val_loss'):
            vl = np.array(self.metrics['val_loss'])
            # align lengths if needed
            if len(vl) == len(steps):
                ax1.plot(steps, vl, linestyle='--', linewidth=2, alpha=0.9, label='Validation Loss')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Progress'); ax1.legend(); ax1.grid(True, alpha=0.3)

        if self.metrics.get('learning_rate'):
            ax2 = fig.add_subplot(gs[0, 2])
            lrs = np.array(self.metrics['learning_rate'])
            ax2.plot(steps[:len(lrs)], lrs, linewidth=2)
            ax2.set_xlabel('Step'); ax2.set_ylabel('LR'); ax2.set_title('LR Schedule'); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

        out = self.base_dir / "plots" / "training_curves" / "training_summary.png"
        plt.suptitle(f'{self.experiment_name} - Training Analysis', fontsize=16)
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        ok(f"Training curves saved to {out}")

    def generate_all(self):
        info("Generating visualizations...")
        self._plot_training_curves()
        # Add more plots as needed
        ok("Visualizations complete")
