import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. COMPLEX-VALUED OPTICAL LAYERS
# ==========================================

class ComplexPhaseLinear(nn.Module):
    """
    The True Phase Linear Layer.
    Input:  Complex Tensor (Batch, Dim)
    Weight: Phase Rotation Matrix e^(i*theta)
    Output: Complex Tensor
    
    Operation: z_out = z_in @ e^(i*theta)
    Physics:   Phase Shift + Interference
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # We learn the Phase Angle (theta), not the real/imag numbers directly.
        # This forces the weight to stay on the unit circle (pure phase shifter).
        self.theta = nn.Parameter(torch.empty(out_features, in_features).uniform_(-math.pi, math.pi))
        
        # Optional: Learnable Amplitude (Transmission coefficient)
        # In pure phase optics, this is 1.0, but we allow slight attenuation learning.
        self.transmission = nn.Parameter(torch.ones(out_features, in_features) * 0.5)
        
        if bias:
            self.bias_mag = nn.Parameter(torch.zeros(out_features))
            self.bias_phase = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_mag', None)

    def forward(self, x):
        # x is (Batch, ..., In_Dim) in Complex64
        
        # 1. Construct the Optical Matrix (Euler's Formula)
        # W = A * (cos(theta) + i*sin(theta))
        weight = torch.polar(self.transmission, self.theta) # (Out, In)
        
        # 2. Complex Matrix Multiplication
        # This naturally performs Interference:
        # Real parts and Imag parts mix. Destructive interference happens automatically.
        out = F.linear(x, weight)
        
        # 3. Complex Bias
        if self.bias_mag is not None:
            b = torch.polar(self.bias_mag, self.bias_phase)
            out = out + b
            
        return out

class ComplexActivation(nn.Module):
    """
    Optical Non-Linearity.
    Standard ReLU doesn't make sense in Complex space.
    We use 'ModReLU' or 'Amplitude Gating'.
    
    Physics: A non-linear medium where transparency depends on light intensity.
    """
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, z):
        # z: Complex Tensor
        mag = z.abs()
        # ModReLU logic: (|z| + b) * (z / |z|)
        # We clamp magnitude, but keep phase strictly preserved.
        new_mag = F.relu(mag + self.bias)
        return torch.polar(new_mag, z.angle())

class HolographicEmbedding(nn.Module):
    """
    Maps Tokens -> Amplitude AND Phase.
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        # Magnitude Embedding
        self.mag_embed = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.mag_embed.weight, 0.1, 1.0)
        
        # Phase Embedding
        self.phase_embed = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.phase_embed.weight, -math.pi, math.pi)
        
        # Positional Phases (The Hologram)
        self.pos_phase = nn.Parameter(torch.randn(1, 1024, dim) * 0.1)

    def forward(self, x):
        # Get Magnitude and Base Phase
        mag = self.mag_embed(x)
        phase = self.phase_embed(x)
        
        # Add Positional Phase Rotation (Rotation depends on index)
        seq_len = x.size(1)
        pos_p = self.pos_phase[:, :seq_len, :]
        
        final_phase = phase + pos_p
        
        # Return Complex Tensor
        return torch.polar(mag, final_phase)

class PhaseResonanceAttention(nn.Module):
    """
    True Phase Attention.
    Instead of Dot Product of Vectors, we calculate Phase Coherence.
    
    Resonance = Re( Query * Conjugate(Key) )
    If phases are aligned: Output is Max Positive (Constructive).
    If phases are opposite: Output is Max Negative (Destructive).
    """
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Complex Projections
        self.q_proj = ComplexPhaseLinear(dim, dim)
        self.k_proj = ComplexPhaseLinear(dim, dim)
        self.v_proj = ComplexPhaseLinear(dim, dim)
        self.o_proj = ComplexPhaseLinear(dim, dim)
        
    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        
        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2) # Complex
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2) # Complex
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2) # Complex
        
        # RESONANCE CALCULATION
        # (a+bi)(c-di) = (ac+bd) + i(bc-ad)
        # We care about the Real part (ac+bd) which represents phase alignment.
        # Matmul of Complex numbers handles this:
        # attn_score = Q @ K.H (Hermitian Transpose)
        
        # Manual Complex Dot Product to be explicit:
        # We use conjugate of K to find phase difference.
        attn_logits = torch.matmul(q, k.conj().transpose(-2, -1))
        
        # We take the Magnitude of the resonance as the attention score
        # (Or the Real part if we want anti-alignment to be penalized)
        attn_scores = attn_logits.abs() / math.sqrt(self.head_dim)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Cast weights back to complex for multiplication (Phase 0)
        attn_weights_c = torch.polar(attn_weights, torch.zeros_like(attn_weights))
        
        # Weighted Sum (Interference)
        out = torch.matmul(attn_weights_c, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

class TruePhaseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = PhaseResonanceAttention(dim)
        self.norm1 = ComplexLayerNorm(dim)
        self.norm2 = ComplexLayerNorm(dim)
        
        self.mlp = nn.Sequential(
            ComplexPhaseLinear(dim, dim*4),
            ComplexActivation(dim*4),
            ComplexPhaseLinear(dim*4, dim)
        )
        
    def forward(self, x):
        # Residual connections in Complex space
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ComplexLayerNorm(nn.Module):
    """ LayerNorm applied to Magnitude, Phase preserved """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, z):
        mag = z.abs()
        phase = z.angle()
        
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + self.eps)
        
        norm_mag = norm_mag * self.scale + self.shift
        return torch.polar(norm_mag, phase)

class TrueHolographicSFPT(nn.Module):
    def __init__(self, vocab_size, layers=6, dim=256):
        super().__init__()
        print(f"--- INITIALIZING TRUE COMPLEX-VALUED PHASE NET ---")
        self.embed = HolographicEmbedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TruePhaseBlock(dim) for _ in range(layers)])
        self.norm = ComplexLayerNorm(dim)
        
        # Final Readout: Photodetector (Intensity Measurement)
        # We project Complex -> Real Class Logits
        # Ideally, we project to "Class Resonance Vectors" and measure overlap.
        self.readout = nn.Linear(dim, vocab_size) 

    def forward(self, x, dataset_idx=None, targets=None, **kwargs):
        # x: (B, L)
        B, L = x.shape
        # 1. Convert Token IDs to Complex Wave
        z = self.embed(x) # (B, L, D) Complex64
        
        # 2. Pass through Optical Stack
        for block in self.blocks:
            z = block(z)
            
        z = self.norm(z)
        
        # 3. Photodetection (Intensity)
        # We take the Magnitude (Intensity) of the final complex wave
        intensity = z.abs()
        
        # 4. Classify based on Intensity Pattern
        logits = self.readout(intensity)
        
        # Return format compatible with trainer
        # logits, aux, gate, _, _, think_stats
        return logits, torch.tensor(0.0).to(logits.device), None, None, None, {}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            logits = self(idx_cond)[0]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
