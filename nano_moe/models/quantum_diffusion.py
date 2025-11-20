# @title ⚛️ Quantum Diffusion ARC (The Thermodynamic Solver)
# Solves puzzles by "crystallizing" a solution from noise.
# Combines: Wave-Particle Dual Backbone + DDPM (Denoising Diffusion).

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import time
import json
import random
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# ==========================================
# 1. DIFFUSION UTILS (Time Embeddings)
# ==========================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ==========================================
# 2. THE BACKBONE (Quantum U-Net)
# ==========================================
# This replaces the standard U-Net. 
# It uses your Wave-Particle blocks to denoise.

class WaveBlock(nn.Module):
    """ FFT Lens for global structure denoising """
    def __init__(self, dim, shape=(30, 30)):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.view_as_complex(torch.randn(dim, shape[0], shape[1], 2) * 0.02))
    
    def forward(self, x):
        x_c = torch.complex(x, torch.zeros_like(x))
        x_freq = torch.fft.fft2(x_c, norm='ortho')
        x_filtered = x_freq * self.complex_weight
        return torch.fft.ifft2(x_filtered, norm='ortho').abs()

class ParticleBlock(nn.Module):
    """ Attention for local pixel-fixing """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_norm = self.norm(x)
        x_norm_flat = x_norm.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        attn_out, _ = self.attn(x_norm_flat, x_norm_flat, x_norm_flat)
        return x + attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

class QuantumDenoiser(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Input Embedding (Noisy Target + Conditional Input)
        # We concat the Noisy Grid (12 channels) and the Input Condition Grid (12 channels)
        # Actually, we embed indices to vectors first.
        self.grid_embed = nn.Embedding(12, dim) # For Condition
        
        # Input projection for the continuous noisy state
        self.noisy_proj = nn.Conv2d(dim, dim, 1) 
        
        # The Hybrid Layers
        self.layers = nn.ModuleList([
            WaveBlock(dim*2),      # See the global pattern
            ParticleBlock(dim*2),  # Fix local relations
            WaveBlock(dim*2),      # Refine global symmetry
            ParticleBlock(dim*2)   # Final pixel polish
        ])
        
        self.final_conv = nn.Conv2d(dim*2, dim, 1) # Output predicted noise

    def forward(self, x_noisy, x_cond, time_emb):
        # x_noisy: (B, Dim, H, W) - The latent we are denoising
        # x_cond: (B, H, W) - The input puzzle grid (Integers)
        
        # Embed condition
        cond_emb = self.grid_embed(x_cond).permute(0, 3, 1, 2) # (B, D, H, W)
        
        # Process Time
        t = self.time_mlp(time_emb) # (B, Dim)
        t = t.unsqueeze(-1).unsqueeze(-1) # (B, Dim, 1, 1)
        
        # Fuse: Noisy State + Condition
        # We concatenate them along channel dim
        x = torch.cat([x_noisy, cond_emb], dim=1) # (B, 2*Dim, H, W)
        
        # Add time info (broadcast)
        x = x + torch.cat([t, t], dim=1)
        
        for layer in self.layers:
            x = x + layer(x)
            
        return self.final_conv(x)

# ==========================================
# 3. THE DIFFUSION MANAGER
# ==========================================

class ARCDiffusion:
    def __init__(self, device='cuda', timesteps=50):
        self.device = device
        self.timesteps = timesteps
        
        # Linear Schedule
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion (Add noise)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, x_cond, t, t_index):
        # Reverse diffusion (Denoise one step)
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1. / self.extract(self.alphas, t, x.shape))
        
        # Predict noise using our Quantum Backbone
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, x_cond, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t * (1. - self.extract(self.alphas_cumprod, t, x.shape).prev) / (1. - self.extract(self.alphas_cumprod, t, x.shape))
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise

    def train_step(self, model, x_start, x_cond, optimizer):
        # x_start: Target Grid (B, H, W) -> need to embed first
        # x_cond: Input Grid (B, H, W)
        
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        
        # Embed Target to Continuous Space for Diffusion
        # We use the same embedding layer as the model for consistency
        target_emb = model.grid_embed(x_start).permute(0, 3, 1, 2)
        
        noise = torch.randn_like(target_emb)
        x_noisy = self.q_sample(target_emb, t, noise)
        
        noise_pred = model(x_noisy, x_cond, t)
        
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, model, x_cond):
        # Generate from noise
        B = x_cond.shape[0]
        H, W = x_cond.shape[1], x_cond.shape[2]
        Dim = model.grid_embed.embedding_dim
        
        # Start with pure noise
        img = torch.randn((B, Dim, H, W), device=self.device)
        
        history = [] # To visualize the "thought process"
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, x_cond, t, i)
            if i % 10 == 0:
                history.append(img.cpu())
                
        return img, history

# ==========================================
# 4. RUNNER
# ==========================================

def pad(grid):
    g = np.array(grid)
    h, w = g.shape
    padded = np.ones((30, 30), dtype=int) * 10
    padded[:h, :w] = g
    return torch.tensor(padded, dtype=torch.long)

def train_quantum_diffusion(cfg=None):
    # Setup
    if not os.path.exists("training.json"):
        url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training.json"
        urllib.request.urlretrieve(url, "training.json")
    with open("training.json", 'r') as f: tasks = json.load(f)
    
    task_id = random.choice(list(tasks.keys()))
    task = tasks[task_id]
    console.print(f"--- DIFFUSING SOLUTION FOR TASK {task_id} ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Init
    dim = 64
    model = QuantumDenoiser(dim=dim).to(device)
    diffusion = ARCDiffusion(device, timesteps=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 1. Train (Test-Time Adaptation)
    model.train()
    # Prepare training batch (Augmented)
    train_inps, train_outs = [], []
    for p in task['train']:
        i_g, o_g = np.array(p['input']), np.array(p['output'])
        # Add Identity + Rotations
        for k in range(4):
            train_inps.append(pad(np.rot90(i_g, k)))
            train_outs.append(pad(np.rot90(o_g, k)))
            
    t_in = torch.stack(train_inps).to(device)
    t_out = torch.stack(train_outs).to(device)
    
    console.print("Focusing lens (Training)...")
    for step in range(300):
        loss = diffusion.train_step(model, t_out, t_in, optimizer)
        if step % 50 == 0: console.print(f"Step {step} Loss: {loss:.4f}")
        
    # 2. Solve
    console.print("Diffusing (Inference)...")
    model.eval()
    test_inp = pad(task['test'][0]['input']).unsqueeze(0).to(device)
    
    # Run reverse diffusion
    final_emb, history = diffusion.sample(model, test_inp)
    
    # Decode: Continuous Vector -> Discrete Color (Nearest Neighbor)
    # We compute distance to embedding weights
    B, C, H, W = final_emb.shape
    flat_emb = final_emb.permute(0, 2, 3, 1).reshape(-1, C)
    weights = model.grid_embed.weight # (12, C)
    
    # Distance matrix
    dists = torch.cdist(flat_emb, weights) # (B*H*W, 12)
    pred_flat = torch.argmin(dists, dim=1)
    prediction = pred_flat.reshape(H, W).cpu().numpy()
    
    console.print("Quantum Diffusion Complete.")
    return prediction
