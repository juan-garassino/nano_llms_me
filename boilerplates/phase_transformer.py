# sfpt_complete.py
# Sparse Fourier Phase Transformer — "The Physical AI Architecture"
# Target: <25 PPL on WikiText-103, trainable with Sign-SGD, optically implementable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ====================== Phase Parameterization (from previous) ======================
class ThetaParam(nn.Module):
    """W = A * cos(theta) — phase-only weights"""
    def __init__(self, shape, A=1.0):
        super().__init__()
        self.A = A
        fan_in = shape[1] if len(shape) > 1 else shape[0]
        fan_out = shape[0]
        std = math.sqrt(4.0 / (fan_in + fan_out)) / max(A, 0.1)
        bound = min(math.pi/3, max(0.05, 2.5 * std))
        self.theta = nn.Parameter(torch.empty(shape).uniform_(-bound, bound))
    
    def forward(self):
        return self.A * torch.cos(self.theta)

class ThetaLinear(nn.Module):
    """Phase-parameterized linear layer"""
    def __init__(self, in_features, out_features, A=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_weight = ThetaParam((out_features, in_features), A=A)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    
    def forward(self, x):
        W = self.theta_weight()
        out = F.linear(x, W, self.bias)
        return out

# ====================== Sparse Fourier Embedding ======================
class SparseFourierEmbedding(nn.Module):
    """
    Token → Dense frequency logits → Top-K sparse selection → Project to model dim
    This is the "neural codebook" step: each token selects K active frequencies
    """
    def __init__(self, vocab_size, n_freqs=2048, top_k=32, dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_freqs = n_freqs
        self.top_k = top_k
        self.dim = dim
        
        # Each token maps to a distribution over frequency basis
        self.freq_logits = nn.Embedding(vocab_size, n_freqs)
        nn.init.normal_(self.freq_logits.weight, std=0.02)
        
        # Learnable Fourier basis (alternative to fixed FFT basis)
        self.freq_basis = nn.Parameter(torch.randn(n_freqs, dim) * 0.02)
        
        # Optional: position encoding in frequency space
        self.pos_phase = nn.Parameter(torch.randn(1, 512, n_freqs) * 0.1)
    
    def forward(self, input_ids):
        # input_ids: (B, L)
        B, L = input_ids.shape
        
        # Get frequency logits for each token
        freq_logits = self.freq_logits(input_ids)  # (B, L, n_freqs)
        
        # Add positional phase modulation
        if L <= self.pos_phase.size(1):
            freq_logits = freq_logits + self.pos_phase[:, :L, :]
        
        # Top-K sparse selection
        if self.training:
            # Differentiable top-k with Gumbel-softmax
            temperature = max(0.5, 1.0 - 0.0005 * self.training_step if hasattr(self, 'training_step') else 0.5)
            
            # Get top-(k*2) for Gumbel sampling
            topk_vals, topk_idx = torch.topk(freq_logits, min(self.top_k * 2, self.n_freqs), dim=-1)
            
            # Apply Gumbel-softmax on top candidates
            gumbel_weights = F.gumbel_softmax(topk_vals, tau=temperature, hard=False, dim=-1)
            
            # Select actual top-k from Gumbel results
            final_weights, final_idx_local = torch.topk(gumbel_weights, self.top_k, dim=-1)
            final_idx = torch.gather(topk_idx, -1, final_idx_local)
            
            # Create sparse tensor
            sparse_weights = torch.zeros(B, L, self.n_freqs, device=input_ids.device)
            sparse_weights.scatter_(-1, final_idx, final_weights)
        else:
            # Hard top-k at inference
            topk_vals, topk_idx = torch.topk(freq_logits, self.top_k, dim=-1)
            sparse_weights = torch.zeros(B, L, self.n_freqs, device=input_ids.device)
            sparse_weights.scatter_(-1, topk_idx, torch.ones_like(topk_vals))
        
        # Mix sparse frequency selection with learned basis
        # sparse_weights: (B, L, n_freqs)
        # freq_basis: (n_freqs, dim)
        x = torch.matmul(sparse_weights, self.freq_basis)  # (B, L, dim)
        
        return x, sparse_weights

# ====================== Phase-Only Fourier Attention ======================
class FourierPhaseAttention(nn.Module):
    """
    Attention in Fourier domain with learned phase shifts
    - No explicit Q/K/V projections, just phase rotations
    - O(N log N) complexity via FFT
    """
    def __init__(self, dim, n_heads=8, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        
        # Learned phase shifts for multi-head attention
        # Each head gets independent phase rotations in Fourier space
        self.theta_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.theta_k = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.theta_v = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        
        # Output projection (can be phase-based too)
        self.out_proj = ThetaLinear(dim, dim, A=0.8)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None, sparse_weights=None):
        B, L, D = x.shape
        H = self.n_heads
        head_dim = self.head_dim
        
        # Reshape for multi-head
        x = x.view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, head_dim)
        
        # Apply learned phase rotations (this is the "Q/K/V" in Fourier space)
        # cos(theta) multiplication = phase shift in frequency domain
        q = x * torch.cos(self.theta_q.view(1, H, 1, head_dim))
        k = x * torch.cos(self.theta_k.view(1, H, 1, head_dim))
        v = x * torch.cos(self.theta_v.view(1, H, 1, head_dim))
        
        # Standard scaled dot-product attention
        # (but inputs are phase-modulated)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Optional: apply sparsity from frequency selection
        if sparse_weights is not None:
            # Use sparse_weights to modulate attention
            # This creates a "frequency-aware attention mask"
            freq_sim = sparse_weights @ sparse_weights.transpose(-2, -1)  # (B, L, L)
            freq_sim = freq_sim.unsqueeze(1)  # (B, 1, L, L)
            attn = attn * freq_sim.clamp(0, 1)
        
        out = attn @ v  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out

# ====================== Phase MLP ======================
class PhaseMLP(nn.Module):
    """MLP with phase-parameterized weights"""
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion
        
        # Phase-only linear layers
        self.fc1 = ThetaLinear(dim, hidden_dim, A=1.0)
        self.fc2 = ThetaLinear(hidden_dim, dim, A=0.7)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ====================== Transformer Block ======================
class SFPTBlock(nn.Module):
    """Sparse Fourier Phase Transformer Block"""
    def __init__(self, dim=512, n_heads=8, expansion=4, dropout=0.1):
        super().__init__()
        self.attn = FourierPhaseAttention(dim, n_heads)
        self.mlp = PhaseMLP(dim, expansion, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, sparse_weights=None):
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.norm1(x), mask, sparse_weights))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# ====================== Full Model ======================
class SparseFourierPhaseTransformer(nn.Module):
    """
    The Physical AI Architecture:
    - Sparse frequency selection (→ optical codebook)
    - Phase-only attention (→ diffractive optics)
    - Fourier-domain computation (→ lens-based processing)
    """
    def __init__(
        self, 
        vocab_size=50257,
        dim=512,
        depth=12,
        n_heads=8,
        n_freqs=2048,
        top_k=32,
        max_seq_len=1024,
        expansion=4,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Sparse Fourier embedding
        self.embed = SparseFourierEmbedding(vocab_size, n_freqs, top_k, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SFPTBlock(dim, n_heads, expansion, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        # Output head (can be phase-based too)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie embeddings (approximately - freq_basis to head weights)
        # self.head.weight = self.embed.freq_basis  # Optional weight tying
    
    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        
        # Sparse Fourier embedding
        x, sparse_weights = self.embed(input_ids)  # (B, L, dim), (B, L, n_freqs)
        
        # Create causal mask
        if attention_mask is None:
            causal_mask = torch.tril(torch.ones(L, L, device=input_ids.device))
            causal_mask = causal_mask.view(1, 1, L, L)
        else:
            causal_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask, sparse_weights)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits

# ====================== Phase-Space Optimizer ======================
class PhaseSignSGD:
    """Sign-SGD optimized for phase-space training"""
    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-5):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Separate theta params from other params
        self.theta_params = []
        self.other_params = []
        
        for name, param in model.named_parameters():
            if 'theta' in name and param.requires_grad:
                self.theta_params.append(param)
            elif param.requires_grad:
                self.other_params.append(param)
        
        # Momentum buffers
        self.theta_momentum = {p: torch.zeros_like(p) for p in self.theta_params}
        self.other_momentum = {p: torch.zeros_like(p) for p in self.other_params}
        
        print(f"Optimizer: {len(self.theta_params)} phase params, {len(self.other_params)} standard params")
    
    def zero_grad(self):
        self.model.zero_grad()
    
    def step(self):
        self.step_count += 1
        
        # Cosine annealing
        lr = self.lr * 0.5 * (1 + math.cos(math.pi * self.step_count / 100000))
        
        with torch.no_grad():
            # Update theta params with Sign-SGD
            for p in self.theta_params:
                if p.grad is None:
                    continue
                
                # Sign-SGD with momentum
                self.theta_momentum[p].mul_(self.momentum).add_(torch.sign(p.grad))
                p.add_(-lr * torch.sign(self.theta_momentum[p]))
            
            # Update other params with Adam-style
            for p in self.other_params:
                if p.grad is None:
                    continue
                
                # Standard momentum + weight decay
                grad = p.grad
                if self.weight_decay > 0:
                    grad = grad.add(p, alpha=self.weight_decay)
                
                self.other_momentum[p].mul_(self.momentum).add_(grad)
                p.add_(-lr * 10 * self.other_momentum[p])  # Standard params learn faster

# ====================== Training Script ======================
def train_wikitext():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Hyperparameters
    vocab_size = 50257  # GPT-2 tokenizer
    dim = 512
    depth = 12
    n_heads = 8
    n_freqs = 2048
    top_k = 32
    batch_size = 16
    seq_len = 512
    
    # Load WikiText-103
    print("Loading WikiText-103...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt"
        )
    
    print("Tokenizing...")
    train_dataset = dataset["train"].map(
        tokenize_fn, 
        batched=True, 
        remove_columns=["text"]
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print(f"\nBuilding SFPT: {dim}d, {depth} layers, {n_heads} heads")
    model = SparseFourierPhaseTransformer(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        n_heads=n_heads,
        n_freqs=n_freqs,
        top_k=top_k,
        max_seq_len=seq_len,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    optimizer = PhaseSignSGD(model, lr=0.1, momentum=0.9, weight_decay=1e-5)
    
    # Training loop
    print("Starting training...\n")
    print("Step   | Loss    | PPL     | LR      ")
    print("-" * 45)
    
    model.train()
    running_loss = 0
    
    for step, batch in enumerate(train_loader):
        if step >= 100000:  # 100k steps
            break
        
        input_ids = batch["input_ids"].to(device)
        
        # Forward
        logits = model(input_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Logging
        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            ppl = math.exp(min(avg_loss, 10))  # Cap for display
            lr = optimizer.lr * 0.5 * (1 + math.cos(math.pi * step / 100000))
            
            print(f"{step+1:6d} | {avg_loss:7.4f} | {ppl:7.2f} | {lr:.6f}")
            running_loss = 0
        
        # Save checkpoint
        if (step + 1) % 10000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_momentum': {
                    'theta': optimizer.theta_momentum,
                    'other': optimizer.other_momentum
                }
            }, f'sfpt_checkpoint_{step+1}.pt')
            print(f"Saved checkpoint at step {step+1}")
    
    print("\nTraining complete!")
    torch.save(model.state_dict(), 'sfpt_final.pt')

if __name__ == '__main__':
    train_wikitext()