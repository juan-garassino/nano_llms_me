import torch
import torch.nn as nn
import numpy as np
import random
from rich.console import Console
from rich.panel import Panel

from .config import TrainingConfig
from .data.loaders import get_multimnist_loaders
from .data.text import get_text_loaders
from .models.moe import IntegratedMoE
from .models.phase import SparseFourierPhaseTransformer
from .models.phase_symbolic import HybridPhaseSymbolicARC
from .training.tracker import ExperimentTracker
from .training.trainer import train_epoch, eval_model

console = Console()

def main():
    # Configuration
    cfg = TrainingConfig()
    
    # Setup
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(Panel(f"🚀 Starting Pipeline on {device} | Model: {cfg.model_type} | Data: {cfg.dataset_type}", style="bold green"))
    
    # Data
    if cfg.dataset_type == "text":
        # Use first dataset name as the target text dataset
        dataset_name = cfg.dataset_names[0] if cfg.dataset_names else "tinyshakespeare"
        loaders, vocab_size = get_text_loaders(dataset_name, cfg.batch_size, cfg.seq_len, num_workers=cfg.num_workers)
        num_classes = vocab_size
        # Override feature_dim if needed? No, feature_dim is model dim.
    else:
        loaders, class_counts = get_multimnist_loaders(cfg.dataset_names, cfg.batch_size, num_workers=cfg.num_workers)
        num_classes = max(class_counts.values())
    
    # Model
    if cfg.model_type == "sfpt":
        model = SparseFourierPhaseTransformer(
            vocab_size=num_classes,
            dim=cfg.feature_dim,
            depth=cfg.depth,
            n_heads=cfg.n_heads,
            n_freqs=cfg.n_freqs,
            top_k=cfg.top_k,
            expansion=cfg.expansion
        ).to(device)
    elif cfg.model_type == "phase_symbolic":
        model = HybridPhaseSymbolicARC(
            vocab_size=num_classes,
            dim=cfg.feature_dim,
            n_layer=cfg.depth,
            n_heads=cfg.n_heads,
            n_freqs=cfg.n_freqs,
            top_k=cfg.top_k,
            expansion=cfg.expansion,
            dropout=0.1,
            n_ops=cfg.n_ops,
            max_program_len=cfg.max_program_len
        ).to(device)
    else:
        model = IntegratedMoE(
            num_experts=cfg.experts,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    tracker = ExperimentTracker(cfg.save_dir)
    
    # Training Loop
    for epoch in range(1, cfg.epochs + 1):
        losses, gates, thinking = train_epoch(model, loaders, optimizer, criterion, device, cfg.dataset_names, tracker, epoch)
        
        accs = eval_model(model, loaders, device, cfg.dataset_names)
        avg_loss = np.mean(losses)
        avg_think = np.mean(thinking) if thinking else 0.0
        
        # Log
        tracker.log_epoch_metrics(epoch, avg_loss, np.mean(list(accs.values())), accs, torch.stack(gates).mean(0), thinking_depth=avg_think)
        
        # Display
        console.print(f"[bold]Epoch {epoch}[/]: Loss={avg_loss:.4f}, Avg Acc={np.mean(list(accs.values())):.4f}, Thinking Depth={avg_think:.2f}")
        
    tracker.save_metrics()
    console.print(f"[bold green]Training Complete! Results saved to {cfg.save_dir}[/]")

if __name__ == "__main__":
    main()
