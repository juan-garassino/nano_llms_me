"""
Fourier-domain CLIP for multimodal alignment in frequency space.

Instead of aligning embeddings in Euclidean space, aligns frequency spectra.
This captures deeper structural patterns in both vision and language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierImageEncoder(nn.Module):
    """
    Image encoder that outputs frequency-domain representation.
    """
    
    def __init__(self, emb_dim=128, n_freqs=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_freqs = n_freqs
        
        # CNN backbone for initial features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Project to frequency coefficients
        self.to_freq = nn.Linear(128, n_freqs * 2)  # Real + Imag
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) images
        Returns:
            freq_emb: (B, n_freqs) complex frequency representation
        """
        # CNN features
        h = self.cnn(x).squeeze(-1).squeeze(-1)  # (B, 128)
        
        # Project to complex frequencies
        freq_flat = self.to_freq(h)  # (B, n_freqs * 2)
        real, imag = freq_flat.chunk(2, dim=-1)
        freq_emb = torch.complex(real, imag)
        
        # Normalize in frequency space
        freq_emb = freq_emb / (torch.abs(freq_emb).mean(dim=-1, keepdim=True) + 1e-8)
        
        return freq_emb


class FourierTextEncoder(nn.Module):
    """
    Text encoder that outputs frequency-domain representation.
    """
    
    def __init__(self, vocab_size, emb_dim=128, n_freqs=64, max_len=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_freqs = n_freqs
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.01)
        
        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, dim_feedforward=emb_dim*4,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        
        # Project to frequency coefficients
        self.to_freq = nn.Linear(emb_dim, n_freqs * 2)
    
    def forward(self, tok):
        """
        Args:
            tok: (B, L) tokenized text
        Returns:
            freq_emb: (B, n_freqs) complex frequency representation
        """
        # Text encoding
        mask = (tok == 0)
        x = self.embed(tok) + self.pos[:, :tok.size(1)]
        h = self.encoder(x, src_key_padding_mask=mask)
        
        # Mean pool (excluding padding)
        mask_expanded = mask.unsqueeze(-1).expand_as(h)
        h_masked = h.masked_fill(mask_expanded, 0)
        pooled = h_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        
        # Project to complex frequencies
        freq_flat = self.to_freq(pooled)
        real, imag = freq_flat.chunk(2, dim=-1)
        freq_emb = torch.complex(real, imag)
        
        # Normalize
        freq_emb = freq_emb / (torch.abs(freq_emb).mean(dim=-1, keepdim=True) + 1e-8)
        
        return freq_emb


class FourierContrastiveLoss(nn.Module):
    """
    Contrastive loss in frequency domain.
    
    Uses complex dot product: <z1, z2> = Re(z1 * conj(z2))
    This captures both magnitude and phase alignment.
    """
    
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
    
    def complex_similarity(self, z1, z2):
        """
        Compute similarity between complex embeddings.
        
        Args:
            z1, z2: (B, F) complex tensors
        Returns:
            (B, B) similarity matrix
        """
        # Complex inner product: Re(<z1, conj(z2)>)
        similarity = torch.real(z1 @ z2.conj().T)
        
        # Normalize by magnitudes
        mag1 = torch.abs(z1).sum(dim=-1, keepdim=True)
        mag2 = torch.abs(z2).sum(dim=-1, keepdim=True)
        similarity = similarity / (mag1 @ mag2.T + 1e-8)
        
        return similarity
    
    def forward(self, img_freq, txt_freq):
        """
        Compute contrastive loss in frequency space.
        
        Args:
            img_freq: (B, F) complex image frequencies
            txt_freq: (B, F) complex text frequencies
        Returns:
            loss, temperature
        """
        temp = self.log_temp.exp()
        
        # Similarity in frequency space
        logits = self.complex_similarity(img_freq, txt_freq) / temp
        labels = torch.arange(img_freq.size(0), device=img_freq.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2, temp


class FourierCLIP(nn.Module):
    """
    Complete Fourier-domain CLIP model.
    
    Aligns vision and language in frequency space rather than embedding space,
    potentially capturing deeper structural patterns.
    """
    
    def __init__(self, vocab_size, emb_dim=128, n_freqs=64, max_len=16):
        super().__init__()
        
        self.img_encoder = FourierImageEncoder(emb_dim, n_freqs)
        self.txt_encoder = FourierTextEncoder(vocab_size, emb_dim, n_freqs, max_len)
        self.criterion = FourierContrastiveLoss()
    
    def forward(self, imgs, tok):
        """
        Forward pass for training.
        
        Args:
            imgs: (B, 1, H, W) images
            tok: (B, L) tokenized text
        Returns:
            Dictionary with loss and frequency embeddings
        """
        img_freq = self.img_encoder(imgs)
        txt_freq = self.txt_encoder(tok)
        
        loss, temp = self.criterion(img_freq, txt_freq)
        
        return {
            "loss": loss,
            "temperature": temp,
            "img_freq": img_freq.detach(),
            "txt_freq": txt_freq.detach()
        }
    
    @torch.no_grad()
    def encode_image(self, imgs):
        """Encode images to frequency domain."""
        return self.img_encoder(imgs)
    
    @torch.no_grad()
    def encode_text(self, tok):
        """Encode text to frequency domain."""
        return self.txt_encoder(tok)
    
    @torch.no_grad()
    def compute_similarity(self, img_freq, txt_freq):
        """Compute frequency-domain similarity."""
        return self.criterion.complex_similarity(img_freq, txt_freq)


def visualize_frequency_alignment(img_freq, txt_freq, labels=None):
    """
    Visualize how image and text frequencies align.
    
    Args:
        img_freq: (B, F) complex image frequencies
        txt_freq: (B, F) complex text frequencies
        labels: Optional class labels
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy
    img_mag = torch.abs(img_freq).cpu().numpy()
    txt_mag = torch.abs(txt_freq).cpu().numpy()
    img_phase = torch.angle(img_freq).cpu().numpy()
    txt_phase = torch.angle(txt_freq).cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnitude spectra
    axes[0, 0].plot(img_mag.T, alpha=0.3, color='blue')
    axes[0, 0].set_title("Image Magnitude Spectra")
    axes[0, 0].set_xlabel("Frequency")
    axes[0, 0].set_ylabel("Magnitude")
    
    axes[0, 1].plot(txt_mag.T, alpha=0.3, color='red')
    axes[0, 1].set_title("Text Magnitude Spectra")
    axes[0, 1].set_xlabel("Frequency")
    axes[0, 1].set_ylabel("Magnitude")
    
    # Phase patterns
    axes[1, 0].imshow(img_phase, aspect='auto', cmap='twilight')
    axes[1, 0].set_title("Image Phase Patterns")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Sample")
    
    axes[1, 1].imshow(txt_phase, aspect='auto', cmap='twilight')
    axes[1, 1].set_title("Text Phase Patterns")
    axes[1, 1].set_xlabel("Frequency")
    axes[1, 1].set_ylabel("Sample")
    
    plt.tight_layout()
    plt.savefig("fourier_clip_alignment.png", dpi=150)
    plt.show()
    
    print("✅ Saved visualization to fourier_clip_alignment.png")
