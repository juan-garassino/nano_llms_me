import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from rich.console import Console

console = Console()

def apply_pruning(model: nn.Module, prune_pct: float,
                  path_with_masks="pruned_with_masks.pt",
                  path_clean="pruned_clean.pt"):
    """
    Apply global unstructured pruning to all Linear layers in the model.
    Saves two checkpoints: one with masks (for further training) and one clean (for inference).
    """
    if prune_pct <= 0:
        return model

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
            
    if not parameters_to_prune:
        console.print("[yellow]No linear layers found to prune.[/yellow]")
        return model

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_pct,
    )
    console.print(f"[bold green]Applied global {prune_pct*100:.1f}% sparsity.[/bold green]")

    # Save with masks
    torch.save(model.state_dict(), path_with_masks)
    console.print(f"Saved pruned model with masks -> {path_with_masks}")

    # Remove pruning wrappers (make permanent)
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")

    # Save clean
    torch.save(model.state_dict(), path_clean)
    console.print(f"Saved cleaned pruned model -> {path_clean}")

    return model

def apply_quantization(model: nn.Module, quantize_mode: str):
    """
    Apply dynamic quantization or FP16 conversion.
    """
    if quantize_mode == "int8_dynamic":
        # Only quantize Linear layers
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        console.print("[bold blue]Applied int8 dynamic quantization.[/bold blue]")
    elif quantize_mode == "fp16":
        model = model.half()
        console.print("[bold blue]Converted model to fp16.[/bold blue]")
    
    return model
