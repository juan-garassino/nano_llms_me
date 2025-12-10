import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from .config import TrainingConfig
from .datasets.loaders import get_multimnist_loaders
from .datasets.text import get_text_loaders
from .models.moe import IntegratedMoE
from .models.phase import SparseFourierPhaseTransformer
from .models.phase_symbolic import HybridPhaseSymbolicARC
from .models.true_phase import TrueHolographicSFPT
from .models.universal_phase import UniversalSFPT
from .models.quantum_diffusion import QuantumDenoiser
from .training.tracker import ExperimentTracker
from .training.trainer import train_epoch, eval_model

console = Console()

def main(cfg: Optional[TrainingConfig] = None):
    # Configuration
    if cfg is None:
        cfg = TrainingConfig()
    
    # Setup
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(Panel(f"🚀 Starting Pipeline on {device} | Model: {cfg.model_type} | Data: {cfg.dataset_type}", style="bold green"))
    
    # Data
    if cfg.dataset_type == "text":
        # Use first dataset name as the target text dataset
        if not cfg.dataset_names:
            cfg.dataset_names = ["tinyshakespeare"]
        dataset_name = cfg.dataset_names[0]
        loaders, vocab_size = get_text_loaders(dataset_name, cfg.batch_size, cfg.seq_len, num_workers=cfg.num_workers)
        num_classes = vocab_size
        # Override feature_dim if needed? No, feature_dim is model dim.
    else:
        if not cfg.dataset_names:
            cfg.dataset_names = ["mnist", "fashionmnist"]
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
    elif cfg.model_type == "true_phase":
        model = TrueHolographicSFPT(
            vocab_size=num_classes,
            layers=cfg.depth,
            dim=cfg.feature_dim
        ).to(device)
    elif cfg.model_type == "universal_phase":
        model = UniversalSFPT(
            vocab_size=vocab_size,
            layers=cfg.depth,
            dim=cfg.feature_dim,
            freqs=cfg.n_freqs,
            A=3.0
        ).to(device)
    elif cfg.model_type == "quantum_diffusion":
        # Quantum Diffusion is specialized for ARC, so we might need to handle it differently
        # For now, just instantiate it.
        model = QuantumDenoiser(dim=cfg.feature_dim).to(device)
    elif cfg.model_type == "moe": # Assuming "moe" is the type for IntegratedMoE
        model = IntegratedMoE(
            num_experts=cfg.experts,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    # Dispatch to specialized training loops if needed
    if cfg.model_type == "phase_symbolic" or cfg.dataset_type == "arc":
        from .train_arc import train_arc
        # Pass config to train_arc if it accepts it, or just run it
        # Currently train_arc() creates its own config, we should update it to accept cfg
        # For now, let's just call it.
        train_arc(cfg) 
        return

    if cfg.model_type == "quantum_diffusion":
        from .models.quantum_diffusion import train_quantum_diffusion
        train_quantum_diffusion(cfg)
        return
    
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
