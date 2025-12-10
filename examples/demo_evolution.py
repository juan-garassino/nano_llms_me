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
import os
import json
import matplotlib.pyplot as plt

console = Console()

# =============================================================================
# Demo 1: Evolve MoE Weights with OpenAI-ES
# =============================================================================

def demo_es_weights():
    """Demonstrate weight evolution for a small MoE model."""
    console.print("\n[bold cyan]=== Demo 1: OpenAI-ES Weight Evolution ===[/bold cyan]\n")
    
    # Create output directory
    os.makedirs("results/evolution", exist_ok=True)
    
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
    
    # Save results
    torch.save(model.state_dict(), "results/evolution/evolved_moe_weights.pt")
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title("OpenAI-ES Weight Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.savefig("results/evolution/es_weight_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save evolution log
    with open("results/evolution/es_weight_log.json", "w") as f:
        json.dump({
            "config": {
                "population_size": config.population_size,
                "max_iterations": config.max_iterations,
                "learning_rate": config.learning_rate,
                "sigma": config.sigma
            },
            "history": history,
            "final_fitness": history[-1],
            "improvement": history[-1] - history[0]
        }, f, indent=2)
    
    console.print("[green]💾 Saved ES weight evolution results to results/evolution/[/green]")
    
    return history


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
    
    # Save hyperparameter optimization results
    with open("results/evolution/cmaes_hyperparams.json", "w") as f:
        json.dump({
            "param_bounds": param_bounds,
            "best_params": best_params,
            "target_params": {
                "learning_rate": 3e-4,
                "dropout": 0.1,
                "expansion": 4.0
            }
        }, f, indent=2)
    
    console.print("[green]💾 Saved CMA-ES hyperparameter results to results/evolution/[/green]")
    
    return best_params


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
    
    # Save operator library
    with open("results/evolution/ga_operator_library.json", "w") as f:
        json.dump({
            "best_library": [list(op) for op in best_library],
            "description": "Evolved operator library for ARC tasks"
        }, f, indent=2)
    
    console.print("[green]💾 Saved GA operator library to results/evolution/[/green]")
    
    return best_library


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    console.print("[bold magenta]Evolutionary Strategies Demo for nano_moe[/bold magenta]")
    console.print("This demonstrates gradient-free optimization methods.\n")
    
    # Run demos
    es_history = demo_es_weights()
    best_params = demo_cmaes_hyperparams()
    best_library = demo_ga_operators()
    
    # Create summary report
    summary = {
        "evolution_summary": {
            "es_weight_evolution": {
                "final_fitness": es_history[-1],
                "improvement": es_history[-1] - es_history[0],
                "generations": len(es_history)
            },
            "cmaes_hyperparams": best_params,
            "ga_operator_count": len(best_library)
        }
    }
    
    with open("results/evolution/evolution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    console.print("\n[bold green]✓ All demos completed![/bold green]")
    console.print("[green]📊 Check results/evolution/ for all outputs and visualizations[/green]")
