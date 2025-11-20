import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from .config import TrainingConfig
from .models.phase_symbolic import HybridPhaseSymbolicARC
# from .training.monitor import TrainingMonitor
from .training.monitor import TrainingMonitor
from .inference import system2_reasoning_arc
from .data.arc import ARCDataset, collate_arc

console = Console()

def train_arc():
    # Config
    cfg = TrainingConfig()
    cfg.model_type = "phase_symbolic"
    cfg.dataset_type = "arc"
    
    # Monitor
    # monitor = TrainingMonitor(experiment_name="arc_phase_symbolic")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Panel(f"🚀 Starting ARC Training on {device}", style="bold magenta"))
    
    # Data
    dataset = ARCDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_arc)
    
    # Model
    model = HybridPhaseSymbolicARC(
        vocab_size=11,
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training Loop
    step_count = 0
    for epoch in range(1): # Just 1 epoch for verification
        total_loss = 0
        
        # Limit to 2 batches for quick verification
        for i, batch in enumerate(track(loader, description=f"Epoch {epoch}")):
            if i >= 2: break
            # Batch processing (padding)
            max_demos = max(len(item[0]) for item in batch)
            batch_demos_in = []
            batch_demos_out = []
            batch_test_in = []
            batch_test_out = []
            
            for demos_in, demos_out, test_in, test_out in batch:
                curr_demos = len(demos_in)
                pad_count = max_demos - curr_demos
                d_in = torch.stack(demos_in)
                d_out = torch.stack(demos_out)
                if pad_count > 0:
                    pad = torch.zeros(pad_count, 30, 30, dtype=torch.long)
                    d_in = torch.cat([d_in, pad], dim=0)
                    d_out = torch.cat([d_out, pad], dim=0)
                batch_demos_in.append(d_in)
                batch_demos_out.append(d_out)
                batch_test_in.append(test_in)
                batch_test_out.append(test_out)
            
            demos_in_tensor = torch.stack(batch_demos_in).to(device)
            demos_out_tensor = torch.stack(batch_demos_out).to(device)
            test_in_tensor = torch.stack(batch_test_in).to(device)
            test_out_tensor = torch.stack(batch_test_out).to(device)
            
            # Forward
            logits = model.forward_arc(demos_in_tensor, demos_out_tensor, test_in_tensor)
            loss = F.cross_entropy(logits.view(-1, 11), test_out_tensor.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            
            # Log to monitor
            # if step_count % 10 == 0:
            #     monitor.log_training_step(epoch, step_count, loss.item())
            #     monitor.capture_model_state(model)
                
        avg_loss = total_loss / len(loader)
        console.print(f"[bold green]Epoch {epoch} | Loss: {avg_loss:.4f}[/]")
        
        # Validation with System-2 (on a small subset of training data for now as we don't have val set loaded)
        # Just run on the last batch
        with torch.no_grad():
            s2_logits = system2_reasoning_arc(model, demos_in_tensor, demos_out_tensor, test_in_tensor, num_samples=4)
            s2_loss = F.cross_entropy(s2_logits.view(-1, 11), test_out_tensor.view(-1))
            console.print(f"[cyan]System-2 Validation Loss: {s2_loss.item():.4f}[/]")
            
    # monitor.generate_all()
    console.print("[bold blue]Training Complete. Check training_outputs/ for plots.[/]")

if __name__ == "__main__":
    try:
        print("Starting script...")
        train_arc()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
