"""
Wavelet-based attention for handling sharp discontinuities.

Complements Fourier attention by providing localized frequency analysis,
which is essential for sharp edges and sudden transitions that Fourier methods struggle with.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveletAttention(nn.Module):
    """
    Wavelet-based attention using Discrete Wavelet Transform (DWT).
    
    Wavelets provide localized frequency analysis, making them ideal for:
    - Sharp edges/discontinuities
    - Transient patterns
    - Multi-scale features
    """
    
    def __init__(self, dim, n_levels=3, wavelet_type='haar'):
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels
        self.wavelet_type = wavelet_type
        
        # Learnable wavelet coefficients for each level
        self.level_weights = nn.Parameter(torch.ones(n_levels))
        
        # Projection layers for wavelet coefficients
        self.detail_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_levels)
        ])
        self.approx_proj = nn.Linear(dim, dim)
        
        # Output fusion
        self.fusion = nn.Linear(dim * (n_levels + 1), dim)
    
    def haar_wavelet_transform(self, x):
        """
        Compute Haar wavelet transform.
        
        Args:
            x: (B, L, D) input tensor
        Returns:
            approx, details: Approximation and detail coefficients
        """
        B, L, D = x.shape
        
        # Ensure even length for Haar transform
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L += 1
        
        # Reshape for pair-wise operations
        x_pairs = x.view(B, L // 2, 2, D)
        
        # Haar wavelet: approximation (average) and detail (difference)
        approx = (x_pairs[:, :, 0, :] + x_pairs[:, :, 1, :]) / math.sqrt(2)
        detail = (x_pairs[:, :, 0, :] - x_pairs[:, :, 1, :]) / math.sqrt(2)
        
        return approx, detail
    
    def multi_level_dwt(self, x):
        """
        Compute multi-level discrete wavelet transform.
        
        Args:
            x: (B, L, D) input
        Returns:
            List of (approx, detail) tuples for each level
        """
        levels = []
        current = x
        
        for _ in range(self.n_levels):
            approx, detail = self.haar_wavelet_transform(current)
            levels.append((approx, detail))
            current = approx  # Use approximation for next level
        
        return levels
    
    def forward(self, x):
        """
        Apply wavelet attention.
        
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) attention output
        """
        B, L, D = x.shape
        
        # Multi-level wavelet decomposition
        levels = self.multi_level_dwt(x)
        
        # Process each level's details
        processed = []
        for i, (approx, detail) in enumerate(levels):
            # Weight and project detail coefficients
            weight = torch.sigmoid(self.level_weights[i])
            detail_weighted = weight * self.detail_proj[i](detail)
            
            # Upsample to original length
            detail_up = F.interpolate(
                detail_weighted.transpose(1, 2),
                size=L,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            processed.append(detail_up)
        
        # Process final approximation
        final_approx = levels[-1][0]
        approx_weighted = self.approx_proj(final_approx)
        approx_up = F.interpolate(
            approx_weighted.transpose(1, 2),
            size=L,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        processed.append(approx_up)
        
        # Fuse all scales
        fused = torch.cat(processed, dim=-1)
        output = self.fusion(fused)
        
        return output


class HybridFourierWaveletAttention(nn.Module):
    """
    Combines Fourier attention (global patterns) with Wavelet attention (local discontinuities).
    
    This hybrid approach handles:
    - Fourier: Periodic patterns, global symmetries, smooth transformations
    - Wavelet: Sharp edges, transient events, multi-scale features
    """
    
    def __init__(self, dim, n_heads=8, n_freqs=64, top_k=32, n_wavelet_levels=3,
                 fourier_weight=0.7, wavelet_weight=0.3):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Fourier branch
        self.n_freqs = n_freqs
        self.top_k = top_k
        self.qkv_fourier = nn.Linear(dim, 3 * dim)
        
        # Wavelet branch
        self.wavelet_attn = WaveletAttention(dim, n_levels=n_wavelet_levels)
        
        # Learnable branch weights
        self.fourier_weight = nn.Parameter(torch.tensor(fourier_weight))
        self.wavelet_weight = nn.Parameter(torch.tensor(wavelet_weight))
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
    
    def fourier_attention(self, x):
        """
        Fourier-domain attention.
        
        Args:
            x: (B, L, D) input
        Returns:
            (B, L, D) Fourier attention output
        """
        B, L, D = x.shape
        
        # QKV projection
        qkv = self.qkv_fourier(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        
        # Transform to frequency domain
        q_freq = torch.fft.rfft(q, dim=2, norm='ortho')
        k_freq = torch.fft.rfft(k, dim=2, norm='ortho')
        v_freq = torch.fft.rfft(v, dim=2, norm='ortho')
        
        # Compute attention in frequency domain
        # Use magnitude for attention, preserve phase for value
        q_mag = torch.abs(q_freq)
        k_mag = torch.abs(k_freq)
        
        # Select top-k frequencies
        k_topk, k_indices = torch.topk(k_mag.mean(dim=-1), self.top_k, dim=-1)
        
        # Gather top-k for q, k, v
        k_indices_expanded = k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        q_sparse = torch.gather(q_freq, 2, k_indices_expanded)
        k_sparse = torch.gather(k_freq, 2, k_indices_expanded)
        v_sparse = torch.gather(v_freq, 2, k_indices_expanded)
        
        # Complex attention
        attn = torch.einsum('bhld,bhmd->bhlm', q_sparse, k_sparse.conj())
        attn = F.softmax(attn.real / math.sqrt(self.head_dim), dim=-1)
        
        # Apply attention to values
        out_sparse = torch.einsum('bhlm,bhmd->bhld', attn, v_sparse)
        
        # Inverse transform
        out_freq = torch.zeros_like(v_freq)
        out_freq.scatter_(2, k_indices_expanded, out_sparse)
        out = torch.fft.irfft(out_freq, n=L, dim=2, norm='ortho')
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)
        return out
    
    def forward(self, x):
        """
        Hybrid Fourier-Wavelet attention.
        
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) attention output
        """
        # Compute both branches
        fourier_out = self.fourier_attention(x)
        wavelet_out = self.wavelet_attn(x)
        
        # Weighted combination (normalized)
        w_f = torch.sigmoid(self.fourier_weight)
        w_w = torch.sigmoid(self.wavelet_weight)
        total_weight = w_f + w_w
        
        output = (w_f * fourier_out + w_w * wavelet_out) / total_weight
        
        # Final projection
        output = self.proj(output)
        output = self.dropout(output)
        
        return output


class AdaptiveSparsity(nn.Module):
    """
    Learnable sparsity pattern for frequency selection.
    
    Instead of hard top-k selection, learns which frequencies are important
    via soft attention weights.
    """
    
    def __init__(self, n_freqs, init_sparsity=0.5):
        super().__init__()
        self.n_freqs = n_freqs
        
        # Learnable importance per frequency
        self.importance = nn.Parameter(
            torch.randn(n_freqs) * 0.1 + math.log(init_sparsity / (1 - init_sparsity))
        )
        
        # Temperature for soft selection
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, spectrum):
        """
        Apply learned sparsity pattern.
        
        Args:
            spectrum: (B, H, F, D) frequency spectrum
        Returns:
            Sparse spectrum with same shape
        """
        # Soft importance weights (Gumbel-Softmax for differentiability)
        temp = F.softplus(self.temperature) + 0.1
        weights = torch.sigmoid(self.importance / temp)
        
        # Expand to match spectrum dimensions
        weights_expanded = weights.view(1, 1, -1, 1)
        
        # Apply soft mask
        return spectrum * weights_expanded
    
    def get_active_frequencies(self, threshold=0.5):
        """Get indices of active frequencies above threshold."""
        weights = torch.sigmoid(self.importance)
        return (weights > threshold).nonzero(as_tuple=True)[0]
