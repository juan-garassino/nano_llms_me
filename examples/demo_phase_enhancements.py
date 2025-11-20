"""
Demo script showcasing advanced phase transformer enhancements.

Features:
1. HybridFourierWaveletAttention - Combines global (Fourier) and local (Wavelet) patterns
2. AdaptiveSparsity - Learnable frequency selection
3. TruePhaseOptimizer - Proper phase-aware optimization
4. FourierCLIP - Frequency-domain multimodal alignment

Run with: python demo_phase_enhancements.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T

from nano_moe.models.wavelet import (
    HybridFourierWaveletAttention,
    AdaptiveSparsity,
    WaveletAttention
)
from nano_moe.training.phase_optimizer import (
    TruePhaseOptimizer,
    AdaptivePhaseOptimizer,
    PhaseSignSGD
)
from nano_moe.models.fourier_clip import (
    FourierCLIP,
    visualize_frequency_alignment
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {DEVICE}\n")


# =============================================================================
# Demo 1: Hybrid Fourier-Wavelet Attention
# =============================================================================

def demo_hybrid_attention():
    """Demonstrate HybridFourierWaveletAttention on synthetic data."""
    print("=" * 70)
    print("DEMO 1: Hybrid Fourier-Wavelet Attention")
    print("=" * 70)
    
    # Create test signal with both smooth (Fourier) and sharp (Wavelet) components
    L, D = 128, 256
    x = torch.randn(4, L, D).to(DEVICE)
    
    # Add smooth periodic component (Fourier handles this)
    t = torch.linspace(0, 4 * 3.14159, L).unsqueeze(0).unsqueeze(-1).to(DEVICE)
    x += 0.5 * torch.sin(5 * t).expand(4, L, D)
    
    # Add sharp discontinuities (Wavelet handles this)
    x[:, L//4, :] += 2.0  # Sharp spike
    x[:, 3*L//4, :] -= 1.5
    
    # Standard Fourier attention
    fourier_only = torch.fft.rfft(x, dim=1, norm='ortho')
    fourier_recon = torch.fft.irfft(fourier_only, n=L, dim=1, norm='ortho')
    
    # Hybrid Fourier-Wavelet attention
    hybrid_attn = HybridFourierWaveletAttention(
        dim=D,
        n_heads=8,
        n_freqs=64,
        top_k=32,
        n_wavelet_levels=3
    ).to(DEVICE)
    
    hybrid_out = hybrid_attn(x)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x[0, :, 0].cpu(), label='Input (smooth + spikes)', alpha=0.7)
    plt.title("Input Signal")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(fourier_recon[0, :, 0].cpu(), label='Fourier only', alpha=0.7)
    plt.title("Fourier Reconstruction (misses spikes)")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(hybrid_out[0, :, 0].detach().cpu(), label='Hybrid', alpha=0.7)
    plt.title("Hybrid Output (captures both)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("demo_hybrid_attention.png", dpi=150)
    print("✅ Saved visualization: demo_hybrid_attention.png\n")


# =============================================================================
# Demo 2: Adaptive Sparsity
# =============================================================================

def demo_adaptive_sparsity():
    """Demonstrate learnable frequency selection."""
    print("=" * 70)
    print("DEMO 2: Adaptive Sparsity (Learnable Frequency Selection)")
    print("=" * 70)
    
    n_freqs = 64
    adaptive_sparsity = AdaptiveSparsity(n_freqs, init_sparsity=0.5).to(DEVICE)
    
    # Fake spectrum
    spectrum = torch.randn(4, 8, n_freqs, 32, dtype=torch.complex64).to(DEVICE)
    
    # Apply learned mask
    sparse_spectrum = adaptive_sparsity(spectrum)
    
    # Get active frequencies
    active_freqs = adaptive_sparsity.get_active_frequencies(threshold=0.5)
    
    print(f"📊 Total frequencies: {n_freqs}")
    print(f"🎯 Active frequencies: {len(active_freqs)}")
    print(f"📈 Sparsity: {100 * (1 - len(active_freqs) / n_freqs):.1f}%")
    
    # Visualize learned importance
    weights = torch.sigmoid(adaptive_sparsity.importance).cpu().detach().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(n_freqs), weights, alpha=0.7)
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel("Frequency Index")
    plt.ylabel("Learned Importance")
    plt.title("Adaptive Frequency Selection (Higher = More Important)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo_adaptive_sparsity.png", dpi=150)
    print("✅ Saved visualization: demo_adaptive_sparsity.png\n")


# =============================================================================
# Demo 3: Phase Optimization
# =============================================================================

def demo_phase_optimization():
    """Compare different phase optimizers."""
    print("=" * 70)
    print("DEMO 3: Phase-Aware Optimization")
    print("=" * 70)
    
    # Simple phase optimization task: align complex vector to target
    def create_model():
        model = nn.Parameter(torch.randn(64, dtype=torch.complex64))
        return model
    
    target = torch.exp(1j * torch.linspace(0, 2 * 3.14159, 64))
    
    # Test 3 optimizers
    optimizers = {
        "Standard Adam": (create_model(), torch.optim.Adam),
        "PhaseSignSGD": (create_model(), PhaseSignSGD),
        "TruePhaseOpt": (create_model(), TruePhaseOptimizer),
    }
    
    losses_history = {name: [] for name in optimizers}
    
    # Train
    for name, (model, opt_class) in optimizers.items():
        opt = opt_class([model], lr=0.01)
        
        for step in range(200):
            opt.zero_grad()
            loss = F.mse_loss(torch.abs(model - target), torch.zeros(64))
            loss.backward()
            opt.step()
            
            losses_history[name].append(loss.item())
    
    # Visualize convergence
    plt.figure(figsize=(10, 5))
    for name, losses in losses_history.items():
        plt.plot(losses, label=name, alpha=0.8)
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Phase Optimizer Comparison")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("demo_phase_optimization.png", dpi=150)
    print("✅ Saved visualization: demo_phase_optimization.png\n")


# =============================================================================
# Demo 4: Fourier CLIP
# =============================================================================

def demo_fourier_clip():
    """Demonstrate frequency-domain CLIP."""
    print("=" * 70)
    print("DEMO 4: Fourier-Domain CLIP")
    print("=" * 70)
    
    # Load FashionMNIST
    tfm = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = tv.datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    
    # Simple text labels (class names)
    class_names = ["t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    
    # Build vocab
    vocab = sorted(set(' '.join(class_names).split()))
    stoi = {w: i+1 for i, w in enumerate(vocab)}
    stoi['<pad>'] = 0
    
    # Create model
    model = FourierCLIP(vocab_size=len(stoi), emb_dim=128, n_freqs=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for 1 epoch
    print("📚 Training FourierCLIP for 1 epoch...")
    model.train()
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        if batch_idx >= 50:  # Quick demo
            break
        
        imgs = imgs.to(DEVICE)
        
        # Create text tokens from labels
        texts = [class_names[label] for label in labels]
        tok = torch.zeros(len(texts), 8, dtype=torch.long).to(DEVICE)
        for i, text in enumerate(texts):
            words = text.split()
            for j, w in enumerate(words[:8]):
                tok[i, j] = stoi.get(w, 0)
        
        # Forward
        out = model(imgs, tok)
        loss = out['loss']
        
        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/50: loss={loss.item():.4f}")
    
    # Visualize frequency alignment
    model.eval()
    imgs_vis, labels_vis = next(iter(loader))
    imgs_vis = imgs_vis[:16].to(DEVICE)
    labels_vis = labels_vis[:16]
    
    texts_vis = [class_names[label] for label in labels_vis]
    tok_vis = torch.zeros(len(texts_vis), 8, dtype=torch.long).to(DEVICE)
    for i, text in enumerate(texts_vis):
        words = text.split()
        for j, w in enumerate(words[:8]):
            tok_vis[i, j] = stoi.get(w, 0)
    
    with torch.no_grad():
        img_freq = model.encode_image(imgs_vis)
        txt_freq = model.encode_text(tok_vis)
    
    visualize_frequency_alignment(img_freq, txt_freq, labels_vis)
    print("✅ FourierCLIP training complete!\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌊 Phase Transformer Enhancements Demo")
    print("="*70 + "\n")
    
    demo_hybrid_attention()
    demo_adaptive_sparsity()
    demo_phase_optimization()
    demo_fourier_clip()
    
    print("="*70)
    print("✅ All demos complete!")
    print("📊 Check the generated PNG files for visualizations")
    print("="*70)
