import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. THE SHARED PHYSICS (The "Brain")
# ==========================================
# This architecture is invariant. It does not change between Text and ARC.

class ThetaLinear(nn.Module):
    """ The Optical Synapse: W = A * cos(theta) """
    def __init__(self, in_features, out_features, A=3.0, bias=True):
        super().__init__()
        self.A = A
        self.theta = nn.Parameter(torch.empty(out_features, in_features).uniform_(-math.pi, math.pi))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward_pass(self, x):
        # Physical constraint: Weights are phase-dependent
        W = self.A * torch.cos(self.theta)
        return F.linear(x, W, self.bias)

class SparseFourierEmbedding(nn.Module):
    """ 
    The Optical Prism.
    Maps discrete tokens (Word IDs or Color IDs) into Frequency Chords.
    """
    def __init__(self, vocab_size, n_freqs=2048, dim=384):
        super().__init__()
        self.n_freqs = n_freqs
        self.freq_logits = nn.Embedding(vocab_size, n_freqs)
        nn.init.normal_(self.freq_logits.weight, std=0.1) 
        
        self.freq_basis = nn.Parameter(torch.randn(n_freqs, dim) * 0.02)
        # Holographic Positional Encoding (Phase Shift)
        self.pos_phase = nn.Parameter(torch.randn(1, 1024, n_freqs) * 0.02)

    def forward(self, input_ids, top_k_ratio=1.0):
        B, L = input_ids.shape
        logits = self.freq_logits(input_ids)
        
        # Apply Positional Phase Shift
        if L <= self.pos_phase.size(1):
            logits = logits + self.pos_phase[:, :L, :]
        else:
            logits = logits + self.pos_phase[:, :self.pos_phase.size(1), :]

        # Spectral Sparsity (The "Filter")
        k = max(1, int(self.n_freqs * top_k_ratio))
        top_vals, top_inds = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_inds, 1.0)
        
        # Recombine frequencies
        return torch.matmul(logits * mask, self.freq_basis)

class SFPTBlock(nn.Module):
    """ A Single Optical Layer """
    def __init__(self, dim, A=3.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_mix = ThetaLinear(dim, dim, A=A)
        self.mlp_up = ThetaLinear(dim, dim * 4, A=A)
        self.mlp_down = ThetaLinear(dim * 4, dim, A=A)

    def forward(self, x):
        # Interference Mixing
        res = x
        x = self.norm1(x)
        x = self.attn_mix.forward_pass(x)
        x = x + res
        
        # Spectral Processing (MLP)
        res = x
        x = self.norm2(x)
        x = self.mlp_up.forward_pass(x)
        x = F.gelu(x)
        x = self.mlp_down.forward_pass(x)
        x = x + res
        return x

class UniversalSFPT(nn.Module):
    """ The Container """
    def __init__(self, vocab_size, layers=12, dim=384, freqs=2048, A=3.0):
        super().__init__()
        self.embed = SparseFourierEmbedding(vocab_size, n_freqs=freqs, dim=dim)
        self.blocks = nn.ModuleList([SFPTBlock(dim, A=A) for _ in range(layers)])
        self.norm_f = nn.LayerNorm(dim)
        # Intensity Detector (Standard Linear Head for readout)
        self.head = nn.Linear(dim, vocab_size, bias=False) 

    def forward(self, idx, dataset_idx=None, targets=None, top_k_ratio=1.0, **kwargs):
        x = self.embed(idx, top_k_ratio)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # Return format compatible with trainer
        # Trainer expects: logits, aux_loss, gates, _, _, think_stats
        return logits, torch.tensor(0.0).to(logits.device), None, None, None, {}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            logits = self(idx_cond, top_k_ratio=0.1)[0]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
