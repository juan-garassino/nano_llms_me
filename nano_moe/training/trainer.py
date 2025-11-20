import torch
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

try:
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def train_epoch(model, loaders, optimizer, criterion, device, dataset_names, tracker, epoch):
    model.train()
    steps = min(len(loaders[name]["train_loader"]) for name in dataset_names)
    iters = {name: iter(loaders[name]["train_loader"]) for name in dataset_names}
    
    all_losses = []
    all_gates = []
    all_thinking = []

    progress_ctx = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) if RICH_AVAILABLE else None

    if progress_ctx: progress_ctx.start()
    task = progress_ctx.add_task("Training...", total=steps) if progress_ctx else None

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0
        batch_gates = []
        batch_think = []

        for idx, name in enumerate(dataset_names):
            try:
                x, y = next(iters[name])
            except StopIteration:
                iters[name] = iter(loaders[name]["train_loader"])
                x, y = next(iters[name])

            x, y = x.to(device), y.to(device)
            
            # Forward pass with new return values
            # Check if model is SFPT and we are doing text generation (dataset_names[0] is text)
            # We can infer from logits shape.
            
            # For SFPT text, we pass classification=False
            is_text_gen = (logits.dim() == 3) if 'logits' in locals() else False # Can't check before forward
            
            # We need to pass classification=False if it's text.
            # But train_epoch is generic.
            # Let's check if the model has 'classification' arg in forward.
            import inspect
            forward_params = inspect.signature(model.forward).parameters
            kwargs = {'return_attention': True}
            if 'classification' in forward_params:
                 # Heuristic: if y has shape (B, L), it's sequence task
                 if y.dim() == 2:
                     kwargs['classification'] = False
            
            logits, aux, gate, _, _, think_stats = model(x, idx, **kwargs)
            
            if logits.dim() == 3: # (B, L, V)
                # Flatten for CrossEntropyLoss
                B, L, V = logits.shape
                loss = criterion(logits.view(-1, V), y.view(-1)) + 0.01 * aux
            else:
                loss = criterion(logits, y) + 0.01 * aux
                
            total_loss += loss
            
            if gate is not None:
                batch_gates.append(gate.detach().cpu())
            
            # Track thinking depth
            if think_stats and 'iterations_used' in think_stats:
                batch_think.append(think_stats['iterations_used'].mean().item())

        total_loss.backward()
        optimizer.step()
        
        all_losses.append(total_loss.item())
        if batch_gates:
            all_gates.append(torch.stack(batch_gates).mean(0))
        else:
            all_gates.append(torch.zeros(1)) # Dummy
            
        if batch_think:
            all_thinking.append(np.mean(batch_think))
        
        if progress_ctx: progress_ctx.update(task, advance=1)

    if progress_ctx: progress_ctx.stop()
    
    return all_losses, all_gates, all_thinking

@torch.no_grad()
def eval_model(model, loaders, device, dataset_names):
    model.eval()
    accs = {}
    for name in dataset_names:
        loader = loaders[name]["test_loader"]
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Handle classification flag
            import inspect
            forward_params = inspect.signature(model.forward).parameters
            kwargs = {}
            if 'classification' in forward_params:
                 if y.dim() == 2:
                     kwargs['classification'] = False
            
            logits, _, _ = model(x, **kwargs)
            
            if logits.dim() == 3: # Sequence
                pred = logits.argmax(2) # (B, L)
                correct += (pred==y).sum().item()
                total += y.numel()
            else:
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        accs[name] = correct/total
    return accs
