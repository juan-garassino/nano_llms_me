from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    lr: float = 1e-3
    experts: int = 4
    feature_dim: int = 128
    hidden_dim: int = 256
    # Data
    dataset_type: str = "text" # "image" or "text" or "arc"
    dataset_names: list = None # ["mnist", "fashionmnist"] or ["wikitext"]
    batch_size: int = 32 # Reduced for text sequence length
    seq_len: int = 128 # For text
    save_dir: str = "integrated_results"
    num_workers: int = 0
    
    # Model
    model_type: str = "sfpt" # "moe" or "sfpt" or "phase_symbolic"
    
    # SFPT Params
    n_freqs: int = 2048
    top_k: int = 32
    expansion: int = 4
    depth: int = 4
    n_heads: int = 8
    
    # Phase-Symbolic Params
    n_ops: int = 16
    max_program_len: int = 5
