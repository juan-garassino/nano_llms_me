import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ContinuousAttention
from .reflection import ReflectiveAttentionBlock, ReflectiveAttentionCfg

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
