import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class ReflectiveAttentionCfg:
    dim: int
    num_heads: int = 8
    max_iters: int = 3
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
        # Flatten: (b, h, n, d) -> (b, n, h, d) -> (b*n*h, d)
        q_flat = q.transpose(1, 2).contiguous().view(b * n * h, d)
        k_flat = k.transpose(1, 2).contiguous().view(b * n * h, d)
        
        u_flat = self.u_proj(q_flat)
        w_flat = self.w_proj(k_flat)
        
        # Reshape back: (b*n*h, d*rank) -> (b, n, h, d, rank)
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
            
            # Attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
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
            
        return h, iteration_stats
