#!/usr/bin/env python3
"""
Integrated MoE Pipeline
Combines:
1. MoE CNN Infrastructure (from File 3)
2. Continuous Attention / S4 (from File 1)
3. Reflective Attention (from File 2)
"""

import sys, os, random, math, time, json, shutil, zipfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass

# Rich imports
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic output")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# PART 1: ATTENTION 2.0 CORE COMPONENTS (from arc_attention2_coevolution.py)
# ============================================================================

class S4Kernel(nn.Module):
    """State-Space Model kernel for O(L) sequence modeling"""
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.01)))
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """O(L) sequence processing via SSM"""
        B, L, D = u.shape
        dt = torch.exp(self.log_dt).clamp(1e-4, 0.1)
        
        # Discretize: A_bar = I + dt*A, B_bar = dt*B
        A_bar = torch.eye(self.d_state, device=u.device) + dt * self.A
        B_bar = dt * self.B
        
        # Sequential scan (simplified for compatibility)
        x = torch.zeros(B, self.d_state, device=u.device)
        outputs = []
        
        for t in range(L):
            x = A_bar @ x.unsqueeze(-1) + (B_bar @ u[:, t].unsqueeze(-1))
            x = x.squeeze(-1)
            y = (self.C @ x.unsqueeze(-1)).squeeze(-1) + self.D * u[:, t]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

class NeuralOperatorKernel(nn.Module):
    """Neural operator acting on function spaces"""
    def __init__(self, dim: int, num_modes: int = 16):
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        
        # Fourier layer for global interactions
        self.fourier_weight = nn.Parameter(
            torch.randn(dim, dim, num_modes, 2) * 0.02
        )
        
        # Local MLP
        self.local_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply continuous kernel operator"""
        B, L, D = x.shape
        
        # Global: Fourier transform O(L log L)
        # Handle short sequences gracefully
        actual_modes = min(self.num_modes, L // 2 + 1)
        
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')
        out_ft = torch.zeros_like(x_ft)
        
        for i in range(min(actual_modes, x_ft.size(1))):
            weight = torch.view_as_complex(self.fourier_weight[:, :, i])
            out_ft[:, i] = torch.einsum('bd,de->be', x_ft[:, i], weight)
        
        x_global = torch.fft.irfft(out_ft, n=L, dim=1, norm='ortho')
        
        # Local
        x_local = self.local_mlp(x)
        
        return x_global + x_local

class ContinuousAttention(nn.Module):
    """Attention 2.0: Continuous operator + SSM"""
    def __init__(self, dim: int, d_state: int = 64, num_modes: int = 16):
        super().__init__()
        self.to_latent = nn.Linear(dim, dim)
        self.operator = NeuralOperatorKernel(dim, num_modes=num_modes)
        self.ssm = S4Kernel(dim, d_state=d_state)
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """O(L) continuous attention"""
        f = self.to_latent(x)
        f = self.norm(f)
        
        # Neural operator
        kf = self.operator(f)
        
        # SSM memory
        hf = self.ssm(kf)
        
        # Context gate
        g = self.gate(x)
        out = g * hf
        
        return self.to_out(out)

# ============================================================================
# PART 2: REFLECTIVE ATTENTION (from reflection_layer.py)
# ============================================================================

@dataclass
class ReflectiveAttentionCfg:
    dim: int
    num_heads: int = 8
    max_iters: int = 3  # Reduced for speed in this pipeline
    num_relation_types: int = 8
    rank: int = 32
    dropout: float = 0.1
    adaptive_threshold: float = 0.5
    reflection_dim_multiplier: int = 4
    use_energy_feedback: bool = True

class ReflectiveAttentionBlock(nn.Module):
    """Reflective Thinking Block with iterative processing."""
    
    def __init__(self, cfg: ReflectiveAttentionCfg, block_size: int = 128):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.max_iters = cfg.max_iters
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.rank = cfg.rank
        self.use_energy_feedback = cfg.use_energy_feedback
        
        # Core projections
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim)
        
        # Low-rank operator components
        self.u_proj = nn.Linear(self.head_dim, self.head_dim * self.rank)
        self.w_proj = nn.Linear(self.head_dim, self.rank * self.head_dim)
        self.b_proj = nn.Linear(self.head_dim, self.head_dim)
        
        # Learned relation embeddings
        self.relation_embeds = nn.Embedding(cfg.num_relation_types, self.head_dim)
        
        # Adaptive iteration gate
        self.iter_gate = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 4),
            nn.GELU(),
            nn.Linear(cfg.dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Reflection network
        self.reflection = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * cfg.reflection_dim_multiplier),
            nn.GELU(),
            nn.Linear(cfg.dim * cfg.reflection_dim_multiplier, cfg.dim)
        )
        
        # Energy feedback
        if self.use_energy_feedback:
            self.energy_feedback = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim // 2),
                nn.GELU(),
                nn.Linear(cfg.dim // 2, cfg.dim)
            )
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                           .view(1, 1, block_size, block_size))
    
    def _compute_low_rank_operators(self, q, k, v):
        b, h, n, d = q.shape
        q_flat = q.transpose(1, 2).contiguous().view(b * n, d)
        k_flat = k.transpose(1, 2).contiguous().view(b * n, d)
        
        u_flat = self.u_proj(q_flat)
        w_flat = self.w_proj(k_flat)
        
        U = u_flat.view(b, n, h, d, self.rank).transpose(1, 2)
        W = w_flat.view(b, n, h, self.rank, d).transpose(1, 2)
        
        v_expanded = v.unsqueeze(-1)
        inner = torch.matmul(W, v_expanded).squeeze(-1)
        
        return U, inner
    
    def _relation_conditioning(self, q, k):
        b, h, n, d = q.shape
        qk_mean = (q + k) / 2
        qk_mean_pool = qk_mean.mean(dim=2)
        
        rel_logits = torch.einsum('bhd,rd->bhr', qk_mean_pool, 
                                 self.relation_embeds.weight)
        rel_probs = F.softmax(rel_logits, dim=-1)
        
        rel_embed = torch.einsum('bhr,rd->bhd', rel_probs, 
                                self.relation_embeds.weight)
        return rel_embed.unsqueeze(2)
    
    def forward(self, x, energies=None):
        b, n, d = x.shape
        h = x.clone()
        
        iteration_stats = {
            'iterations_used': torch.zeros(b, device=x.device),
            'gate_values': [],
            'energy_feedback_applied': []
        }
        
        # Energy feedback integration
        if energies is not None and self.use_energy_feedback:
            energy_signal = self.energy_feedback(energies)
            h = h + self.dropout(energy_signal)
        
        for t in range(self.max_iters):
            # Project Q, K, V
            q = self.q_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention with causal masking (or full masking for image patches)
            # For this pipeline, we'll assume full attention (no causal mask needed for classification)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            # attn_scores = attn_scores.masked_fill(self.bias[:,:,:n,:n] == 0, float('-inf')) # Disabled for bidirectional
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Standard attention
            usual_out = torch.matmul(attn_probs, v)
            
            # Low-rank relational component
            U, inner = self._compute_low_rank_operators(q, k, v)
            weighted_inner = torch.matmul(attn_probs, inner)
            relational_part = torch.matmul(U, weighted_inner.unsqueeze(-1)).squeeze(-1)
            
            # Bias and relation conditioning
            b_term = self.b_proj(q)
            rel_embed = self._relation_conditioning(q, k)
            
            # Combine components
            out = usual_out + relational_part + b_term + rel_embed
            out = out.transpose(1, 2).contiguous().view(b, n, d)
            out = self.o_proj(out)
            
            # Residual
            h = h + self.dropout(out)
            
            # Reflection step
            h_reflected = self.reflection(self.norm(h))
            h = h + self.dropout(h_reflected)
            
            # Adaptive gating
            gate_val = self.iter_gate(h.mean(dim=1)).squeeze(-1)
            iteration_stats['gate_values'].append(gate_val.mean().item())
            
            # Track iterations
            iteration_stats['iterations_used'] += (t + 1) * (gate_val >= 0.5).float()
            
            # Early exit (soft) - in practice we run fixed iters but could break here
            
        return h, iteration_stats

# ============================================================================
# PART 3: INTEGRATED MOE ARCHITECTURE
# ============================================================================

class IntegratedMoE(nn.Module):
    def __init__(self, num_experts, feature_dim, hidden_dim, num_classes,
                 router_hidden=64, k=1, gumbel=True, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gumbel = gumbel
        self.temperature = temperature

        # 1. CNN Encoder (Spatial Features)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        ) # Output: (B, 64, 7, 7)

        # 2. Continuous Attention Backbone (Sequence Modeling)
        # Treat 7x7 grid as sequence of 49 tokens
        self.seq_proj = nn.Linear(64, feature_dim)
        self.continuous_attn = ContinuousAttention(feature_dim, d_state=32)
        
        # 3. Reflective Reasoning (Thinking)
        self.reflective_cfg = ReflectiveAttentionCfg(
            dim=feature_dim,
            num_heads=4,
            max_iters=2, # Keep low for training speed
            use_energy_feedback=False # Simplified
        )
        self.reflective_block = ReflectiveAttentionBlock(self.reflective_cfg, block_size=49)

        # 4. MoE Router & Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_experts)
        ])

        self.router = nn.Sequential(
            nn.Linear(feature_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts)
        )

    def forward(self, x, dataset_idx=None, return_attention=False):
        B = x.shape[0]
        
        # 1. CNN Features
        cnn_feat = self.cnn_encoder(x) # (B, 64, 7, 7)
        
        # 2. Sequence Prep
        # Rearrange to (B, 49, 64)
        seq = cnn_feat.permute(0, 2, 3, 1).reshape(B, 49, 64)
        seq = self.seq_proj(seq) # (B, 49, feature_dim)
        
        # 3. Continuous Attention
        seq = self.continuous_attn(seq)
        
        # 4. Reflective Thinking
        seq, think_stats = self.reflective_block(seq)
        
        # 5. Global Pooling for Classification
        global_feat = seq.mean(dim=1) # (B, feature_dim)
        
        # 6. MoE Routing
        logits_router = self.router(global_feat)

        if self.gumbel and self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_router) + 1e-8) + 1e-8)
            logits_router = (logits_router + gumbel_noise) / self.temperature

        if self.k == 1:
            if self.gumbel and self.training:
                gate = F.softmax(logits_router, dim=1)
            else:
                topk_vals, topk_idx = logits_router.max(dim=1)
                gate = F.one_hot(topk_idx, num_classes=self.num_experts).float()
        else:
            topk_vals, topk_idx = torch.topk(logits_router, self.k, dim=1)
            gate = torch.zeros_like(logits_router)
            gate.scatter_(1, topk_idx, 1.0/self.k)

        expert_outs = torch.stack([e(global_feat) for e in self.experts], dim=1)
        gate_expanded = gate.unsqueeze(-1)
        out = (expert_outs * gate_expanded).sum(dim=1)

        aux_loss = (logits_router.mean(0)**2).mean()
        mean_gate = gate.mean(dim=0)

        if return_attention:
            return out, aux_loss, mean_gate, gate, expert_outs, think_stats

        return out, aux_loss, mean_gate

# ============================================================================
# PART 4: TRAINING INFRASTRUCTURE (Modified from File 3)
# ============================================================================

class ExperimentTracker:
    """Tracks all experiment metrics and handles visualization"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.create_directory_structure()

        # Initialize tracking variables
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': {},
            'expert_usage': [],
            'routing_weights': [],
            'learning_rate': [],
            'routing_entropy': [],
            'thinking_depth': [] # NEW
        }

        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_epoch': 0
        }

    def create_directory_structure(self):
        dirs = ['models', 'plots', 'data', 'logs']
        for dir_path in dirs:
            (self.save_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def log_epoch_metrics(self, epoch: int, train_loss: float, train_acc: float,
                         val_accs: Dict[str, float], expert_gates: torch.Tensor,
                         thinking_depth: float = 0.0, lr: float = 0.0):
        
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['thinking_depth'].append(thinking_depth)

        for dataset, acc in val_accs.items():
            if dataset not in self.metrics['val_acc']:
                self.metrics['val_acc'][dataset] = []
            self.metrics['val_acc'][dataset].append(acc)

        expert_usage = expert_gates.cpu().numpy()
        self.metrics['expert_usage'].append(expert_usage)
        
        entropy = -np.sum(expert_usage * np.log(expert_usage + 1e-8))
        self.metrics['routing_entropy'].append(entropy)

        avg_val_acc = np.mean(list(val_accs.values()))
        if avg_val_acc > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = avg_val_acc
            self.best_metrics['best_epoch'] = epoch

    def save_metrics(self):
        metrics_file = self.save_dir / 'data/training_metrics.json'
        
        # Simple serialization helper
        def convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.number): return float(o)
            return o

        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=convert)

# Data loading functions (Simplified)
def load_dataset(name: str, root: str, train: bool, transform):
    name = name.lower()
    if name in ["mnist"]:
        ds = datasets.MNIST(root, train=train, download=True, transform=transform)
        n_cls = 10
    elif name in ["fashionmnist", "fashion"]:
        ds = datasets.FashionMNIST(root, train=train, download=True, transform=transform)
        n_cls = 10
    elif name in ["kmnist"]:
        ds = datasets.KMNIST(root, train=train, download=True, transform=transform)
        n_cls = 10
    else:
        # Fallback to MNIST if unknown
        ds = datasets.MNIST(root, train=train, download=True, transform=transform)
        n_cls = 10
    return ds, n_cls

def get_multimnist_loaders(dataset_names: List[str], batch_size: int, root="./data", num_workers=0):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loaders = {}
    class_counts = {}
    
    for name in dataset_names:
        train_ds, n_cls = load_dataset(name, root, True, transform)
        test_ds, _ = load_dataset(name, root, False, transform)
        class_counts[name] = n_cls
        per_bs = max(1, batch_size // len(dataset_names))
        loaders[name] = {
            "train_loader": DataLoader(train_ds, batch_size=per_bs, shuffle=True, num_workers=num_workers),
            "test_loader": DataLoader(test_ds, batch_size=per_bs, shuffle=False)
        }
    return loaders, class_counts

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
            logits, aux, gate, _, _, think_stats = model(x, idx, return_attention=True)
            
            loss = criterion(logits, y) + 0.01 * aux
            total_loss += loss
            
            batch_gates.append(gate.detach().cpu())
            
            # Track thinking depth
            if 'iterations_used' in think_stats:
                batch_think.append(think_stats['iterations_used'].mean().item())

        total_loss.backward()
        optimizer.step()
        
        all_losses.append(total_loss.item())
        all_gates.append(torch.stack(batch_gates).mean(0))
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
            logits, _, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
        accs[name] = correct/total
    return accs

def main():
    # Configuration
    args = {
        'seed': 42,
        'epochs': 3,
        'batch_size': 64,
        'lr': 1e-3,
        'experts': 4,
        'feature_dim': 128,
        'hidden_dim': 256,
        'dataset_names': ["MNIST", "FashionMNIST", "KMNIST"]
    }
    
    # Setup
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(Panel(f"🚀 Starting Integrated MoE Pipeline on {device}", style="bold green"))
    
    # Data
    loaders, class_counts = get_multimnist_loaders(args['dataset_names'], args['batch_size'])
    num_classes = max(class_counts.values())
    
    # Model
    model = IntegratedMoE(
        num_experts=args['experts'],
        feature_dim=args['feature_dim'],
        hidden_dim=args['hidden_dim'],
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    tracker = ExperimentTracker("integrated_results")
    
    # Training Loop
    for epoch in range(1, args['epochs'] + 1):
        losses, gates, thinking = train_epoch(model, loaders, optimizer, criterion, device, args['dataset_names'], tracker, epoch)
        
        accs = eval_model(model, loaders, device, args['dataset_names'])
        avg_loss = np.mean(losses)
        avg_think = np.mean(thinking) if thinking else 0.0
        
        # Log
        tracker.log_epoch_metrics(epoch, avg_loss, np.mean(list(accs.values())), accs, torch.stack(gates).mean(0), thinking_depth=avg_think)
        
        # Display
        console.print(f"[bold]Epoch {epoch}[/]: Loss={avg_loss:.4f}, Avg Acc={np.mean(list(accs.values())):.4f}, Thinking Depth={avg_think:.2f}")
        
    tracker.save_metrics()
    console.print("[bold green]Training Complete! Results saved to ./integrated_results[/]")

if __name__ == "__main__":
    main()
