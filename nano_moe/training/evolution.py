"""
Evolutionary Strategies for Neural Network Training and Structure Search.

Implements:
1. OpenAI-ES (Natural Evolution Strategies) - for weight optimization
2. CMA-ES (Covariance Matrix Adaptation) - for hyperparameter search
3. Genetic Algorithms - for discrete structure evolution (e.g., operator libraries)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.progress import track

console = Console()


# =============================================================================
# 1. OpenAI-ES: Natural Evolution Strategies for Neural Networks
# =============================================================================

@dataclass
class ESConfig:
    """Configuration for Evolution Strategies."""
    population_size: int = 50
    sigma: float = 0.02  # Noise standard deviation
    learning_rate: float = 0.01
    elite_frac: float = 0.2  # Top fraction to select
    max_iterations: int = 300
    
def flatten_params(model: nn.Module) -> np.ndarray:
    """Flatten all model parameters into a single vector."""
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().reshape(-1))
    return torch.cat(params).numpy()

def unflatten_params(model: nn.Module, flat_params: np.ndarray):
    """Unflatten parameter vector back into model."""
    flat_tensor = torch.from_numpy(flat_params).float()
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_tensor[idx:idx+numel].reshape(p.shape).to(p.dtype)
        idx += numel

def evolve_weights(
    model: nn.Module,
    fitness_fn: Callable[[nn.Module], float],
    config: ESConfig = ESConfig(),
    device: str = "cpu"
) -> Tuple[nn.Module, List[float]]:
    """
    Evolve neural network weights using Natural Evolution Strategies.
    
    Args:
        model: Neural network to optimize
        fitness_fn: Function that takes model and returns scalar fitness (higher is better)
        config: ES configuration
        device: Device to run on
        
    Returns:
        Optimized model and fitness history
    """
    console.print(f"[bold green]Starting OpenAI-ES with population={config.population_size}[/bold green]")
    
    # Get initial parameters
    theta = flatten_params(model)
    n_params = len(theta)
    
    fitness_history = []
    best_fitness = -float('inf')
    
    for iteration in track(range(config.max_iterations), description="ES Iterations"):
        # Sample perturbations
        epsilons = []
        fitnesses = []
        
        for _ in range(config.population_size):
            # Sample Gaussian noise
            eps = np.random.randn(n_params)
            epsilons.append(eps)
            
            # Perturb parameters
            theta_try = theta + config.sigma * eps
            
            # Evaluate fitness
            unflatten_params(model, theta_try)
            model.to(device)
            fitness = fitness_fn(model)
            fitnesses.append(fitness)
            
        fitnesses = np.array(fitnesses)
        epsilons = np.array(epsilons)
        
        # Track best
        best_idx = fitnesses.argmax()
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_theta = theta + config.sigma * epsilons[best_idx]
            
        # Normalize fitness (rank-based)
        ranks = np.argsort(fitnesses)
        normalized_fitness = np.zeros_like(fitnesses)
        for i, rank in enumerate(ranks):
            normalized_fitness[rank] = i / (config.population_size - 1) - 0.5
            
        # Update parameters (weighted average of perturbations)
        grad = (epsilons.T @ normalized_fitness) / config.population_size
        theta += config.learning_rate * grad
        
        fitness_history.append(best_fitness)
        
        if iteration % 10 == 0:
            console.print(f"Iteration {iteration} | Best Fitness: {best_fitness:.4f}")
            
    # Load best parameters
    unflatten_params(model, best_theta)
    
    return model, fitness_history


# =============================================================================
# 2. CMA-ES: Covariance Matrix Adaptation for Hyperparameter Search
# =============================================================================

class CMAES:
    """
    Simple CMA-ES implementation for hyperparameter optimization.
    Optimizes over continuous parameter spaces.
    """
    def __init__(self, dim: int, population_size: int = None, sigma: float = 0.3):
        self.dim = dim
        self.population_size = population_size or (4 + int(3 * np.log(dim)))
        self.mu = self.population_size // 2  # Number of parents
        
        # Initialize mean and covariance
        self.mean = np.random.randn(dim)
        self.sigma = sigma
        self.C = np.eye(dim)  # Covariance matrix
        self.pc = np.zeros(dim)  # Evolution path for C
        self.ps = np.zeros(dim)  # Evolution path for sigma
        
        # Strategy parameters
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1 / (self.weights ** 2).sum()
        
        self.cc = 4 / (dim + 4)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(dim+1))-1) + self.cs
        
    def ask(self) -> List[np.ndarray]:
        """Sample population from distribution."""
        # Eigendecomposition for sampling
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 0))
        
        population = []
        for _ in range(self.population_size):
            z = np.random.randn(self.dim)
            y = B @ (D * z)
            x = self.mean + self.sigma * y
            population.append(x)
            
        return population
        
    def tell(self, population: List[np.ndarray], fitnesses: List[float]):
        """Update distribution based on fitness."""
        # Sort by fitness (descending)
        idx = np.argsort(fitnesses)[::-1][:self.mu]
        
        # Compute new mean
        old_mean = self.mean.copy()
        self.mean = sum(self.weights[i] * population[idx[i]] for i in range(self.mu))
        
        # Update evolution paths
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 0))
        invsqrtC = B @ np.diag(1/D) @ B.T
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC @ (self.mean - old_mean) / self.sigma
        
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1-self.cs)**(2*1)) / 
                np.sqrt(self.dim) < 1.4 + 2/(self.dim+1))
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma
        
        # Update covariance matrix
        artmp = [(population[idx[i]] - old_mean) / self.sigma for i in range(self.mu)]
        self.C = ((1 - self.c1 - self.cmu) * self.C + 
                  self.c1 * np.outer(self.pc, self.pc) +
                  self.cmu * sum(self.weights[i] * np.outer(artmp[i], artmp[i]) for i in range(self.mu)))
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))


def optimize_hyperparams(
    param_bounds: Dict[str, Tuple[float, float]],
    objective_fn: Callable[[Dict[str, float]], float],
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Optimize hyperparameters using CMA-ES.
    
    Args:
        param_bounds: Dictionary mapping parameter names to (min, max) bounds
        objective_fn: Function that takes param dict and returns scalar fitness
        n_iterations: Number of CMA-ES iterations
        
    Returns:
        Best hyperparameters found
    """
    console.print("[bold green]Starting CMA-ES Hyperparameter Optimization[/bold green]")
    
    param_names = list(param_bounds.keys())
    dim = len(param_names)
    
    # Normalize to [0, 1]
    def denormalize(x):
        return {name: param_bounds[name][0] + x[i] * (param_bounds[name][1] - param_bounds[name][0])
                for i, name in enumerate(param_names)}
    
    cma = CMAES(dim=dim)
    best_params = None
    best_fitness = -float('inf')
    
    for iteration in track(range(n_iterations), description="CMA-ES"):
        population = cma.ask()
        fitnesses = []
        
        for x in population:
            # Clip to [0, 1]
            x_clipped = np.clip(x, 0, 1)
            params = denormalize(x_clipped)
            fitness = objective_fn(params)
            fitnesses.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = params
                
        cma.tell(population, fitnesses)
        
        if iteration % 10 == 0:
            console.print(f"Iteration {iteration} | Best Fitness: {best_fitness:.4f}")
            
    console.print(f"[bold green]Best Parameters:[/bold green] {best_params}")
    return best_params


# =============================================================================
# 3. Genetic Algorithm for Discrete Structure Search
# =============================================================================

@dataclass
class Individual:
    """Individual in genetic algorithm."""
    genome: Any
    fitness: float = -float('inf')

def genetic_algorithm(
    init_population_fn: Callable[[], Any],
    fitness_fn: Callable[[Any], float],
    mutate_fn: Callable[[Any], Any],
    crossover_fn: Callable[[Any, Any], Any],
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> Any:
    """
    Generic Genetic Algorithm for evolving discrete structures.
    
    Args:
        init_population_fn: Function that returns a random genome
        fitness_fn: Function that evaluates genome fitness
        mutate_fn: Function that mutates a genome
        crossover_fn: Function that crosses two genomes
        population_size: Size of population
        generations: Number of generations
        mutation_rate: Probability of mutation
        elite_size: Number of top individuals to preserve
        
    Returns:
        Best genome found
    """
    console.print(f"[bold green]Starting Genetic Algorithm (pop={population_size}, gen={generations})[/bold green]")
    
    # Initialize population
    population = [Individual(genome=init_population_fn()) for _ in range(population_size)]
    
    best_ever = None
    best_fitness_ever = -float('inf')
    
    for gen in track(range(generations), description="GA Generations"):
        # Evaluate fitness
        for ind in population:
            ind.fitness = fitness_fn(ind.genome)
            if ind.fitness > best_fitness_ever:
                best_fitness_ever = ind.fitness
                best_ever = ind.genome
                
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        if gen % 10 == 0:
            console.print(f"Gen {gen} | Best: {population[0].fitness:.4f} | Avg: {np.mean([i.fitness for i in population]):.4f}")
            
        # Selection + Crossover + Mutation
        new_population = population[:elite_size]  # Elitism
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = max(np.random.choice(population, 3), key=lambda x: x.fitness)
            parent2 = max(np.random.choice(population, 3), key=lambda x: x.fitness)
            
            # Crossover
            child_genome = crossover_fn(parent1.genome, parent2.genome)
            
            # Mutation
            if np.random.rand() < mutation_rate:
                child_genome = mutate_fn(child_genome)
                
            new_population.append(Individual(genome=child_genome))
            
        population = new_population
        
    console.print(f"[bold green]Best Fitness Ever:[/bold green] {best_fitness_ever:.4f}")
    return best_ever


# =============================================================================
# Example: Evolving Phase Operator Libraries for ARC
# =============================================================================

def evolve_operator_library_example():
    """
    Example: Evolve a library of geometric operators for ARC.
    Each operator is represented as a sequence of transformations.
    """
    
    # Define operator primitives
    PRIMITIVES = ["rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v", "identity", "invert_colors"]
    
    def init_operator():
        """Initialize a random operator (sequence of primitives)."""
        length = np.random.randint(1, 4)
        return [np.random.choice(PRIMITIVES) for _ in range(length)]
    
    def init_population():
        """Initialize a library of operators."""
        return [init_operator() for _ in range(8)]  # Library of 8 operators
    
    def fitness(library):
        """Evaluate library on ARC tasks (placeholder)."""
        # In practice: apply operators to ARC grids and check consistency
        # For now: just a dummy score based on diversity
        unique_ops = len(set(tuple(op) for op in library))
        return unique_ops / len(library)  # Reward diversity
    
    def mutate(library):
        """Mutate one operator in the library."""
        lib_copy = [op.copy() for op in library]
        idx = np.random.randint(len(lib_copy))
        if np.random.rand() < 0.5 and len(lib_copy[idx]) > 1:
            # Remove a primitive
            lib_copy[idx].pop(np.random.randint(len(lib_copy[idx])))
        else:
            # Add a primitive
            lib_copy[idx].append(np.random.choice(PRIMITIVES))
        return lib_copy
    
    def crossover(lib1, lib2):
        """Crossover two libraries."""
        # Single-point crossover
        point = np.random.randint(1, len(lib1))
        return lib1[:point] + lib2[point:]
    
    best_library = genetic_algorithm(
        init_population_fn=init_population,
        fitness_fn=fitness,
        mutate_fn=mutate,
        crossover_fn=crossover,
        population_size=30,
        generations=50
    )
    
    return best_library
