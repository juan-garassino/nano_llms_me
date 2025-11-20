"""
Demo: Evolutionary Strategies for nano_moe

This script demonstrates three evolutionary approaches:
1. OpenAI-ES: Evolve neural network weights
2. CMA-ES: Optimize hyperparameters
3. Genetic Algorithm: Evolve operator libraries for ARC
"""

import torch
import torch.nn as nn
from nano_moe.training.evolution import (
    evolve_weights, ESConfig, 
    optimize_hyperparams,
    evolve_operator_library_example
)
from nano_moe.models.moe import IntegratedMoE
from nano_moe.config import TrainingConfig
from rich.console import Console

console = Console()

# =============================================================================
# Demo 1: Evolve MoE Weights with OpenAI-ES
# =============================================================================

def demo_es_weights():
    """Demonstrate weight evolution for a small MoE model."""
    console.print("\n[bold cyan]=== Demo 1: OpenAI-ES Weight Evolution ===[/bold cyan]\n")
    
    # Create small model
    cfg = TrainingConfig()
    model = IntegratedMoE(
        num_experts=3,
        feature_dim=64,
        hidden_dim=128,
        num_classes=10,
        router_hidden=64,
        k=1
    )
    
    # Define fitness function (simple XOR-like task)
    def fitness_fn(model):
        """Evaluate model on a toy task."""
        model.eval()
        # Generate random data
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        
        with torch.no_grad():
            logits, _, _, _ = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            
        # Fitness = negative loss (higher is better)
        return -loss.item()
    
    # Evolve
    config = ESConfig(
        population_size=20,
        max_iterations=30,
        learning_rate=0.01,
        sigma=0.02
    )
    
    model, history = evolve_weights(model, fitness_fn, config)
    
    console.print(f"[green]Final fitness: {history[-1]:.4f}[/green]")
    console.print(f"[green]Improvement: {history[-1] - history[0]:.4f}[/green]")


# =============================================================================
# Demo 2: Optimize Hyperparameters with CMA-ES
# =============================================================================

def demo_cmaes_hyperparams():
    """Demonstrate hyperparameter optimization."""
    console.print("\n[bold cyan]=== Demo 2: CMA-ES Hyperparameter Search ===[/bold cyan]\n")
    
    # Define hyperparameter space
    param_bounds = {
        "learning_rate": (1e-5, 1e-2),
        "dropout": (0.0, 0.5),
        "expansion": (2.0, 8.0),
    }
    
    # Define objective (simulate training)
    def objective(params):
        """Simulate training with these hyperparameters."""
        # In practice: train model and return validation accuracy
        # For demo: just a synthetic function with a known optimum
        lr = params["learning_rate"]
        dropout = params["dropout"]
        expansion = params["expansion"]
        
        # Synthetic fitness (peaks at lr=3e-4, dropout=0.1, expansion=4)
        fitness = -(
            (lr - 3e-4)**2 / (1e-4)**2 +
            (dropout - 0.1)**2 / 0.01 +
            (expansion - 4.0)**2 / 4.0
        )
        
        return fitness
    
    best_params = optimize_hyperparams(param_bounds, objective, n_iterations=30)
    
    console.print(f"[green]Optimal hyperparameters:[/green]")
    for k, v in best_params.items():
        console.print(f"  {k}: {v:.6f}")


# =============================================================================
# Demo 3: Evolve Operator Libraries with Genetic Algorithm
# =============================================================================

def demo_ga_operators():
    """Demonstrate operator evolution for ARC."""
    console.print("\n[bold cyan]=== Demo 3: Genetic Algorithm for Operator Library ===[/bold cyan]\n")
    
    best_library = evolve_operator_library_example()
    
    console.print("[green]Best Operator Library:[/green]")
    for i, op in enumerate(best_library):
        console.print(f"  Operator {i}: {' -> '.join(op)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    console.print("[bold magenta]Evolutionary Strategies Demo for nano_moe[/bold magenta]")
    console.print("This demonstrates gradient-free optimization methods.\n")
    
    # Run demos
    demo_es_weights()
    demo_cmaes_hyperparams()
    demo_ga_operators()
    
    console.print("\n[bold green]✓ All demos completed![/bold green]")
