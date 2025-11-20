"""
Diffusion models (DDPM and Latent Diffusion) for nano_moe.

Includes:
- DDPM: Pixel-space denoising diffusion
- LDM: Latent diffusion with VAE
- Context U-Net architectures
- DDIM sampling
- Dual-image oscillation with 180° rotation (creative feature!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# U-Net Building Blocks
# =============================================================================

class ResidualConvBlock(nn.Module):
    """Residual convolutional block with optional skip connection."""
    
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = (x + x2) if self.same_channels else (x1 + x2)
            return out / 1.41421356237
        else:
            return self.conv2(self.conv1(x))


class UnetDown(nn.Module):
    """Downsampling block with pooling."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, up_in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
        self.res1 = ResidualConvBlock(out_ch + skip_ch, out_ch)
        self.res2 = ResidualConvBlock(out_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Align spatial sizes if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


class EmbedFC(nn.Module):
    """Fully connected embedding layer."""
    
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, x):
        return self.net(x.view(-1, x.size(-1)))


# =============================================================================
# Context U-Net (Pixel Space, 28×28)
# =============================================================================

class ContextUnetPixel(nn.Module):
    """U-Net for pixel-space DDPM with vector conditioning (e.g., text embeddings)."""
    
    def __init__(self, in_channels=1, n_feat=128, cond_dim=128):
        super().__init__()
        self.n_feat = n_feat
        
        # Encoder
        self.init = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # Time and context embeddings
        self.temb1 = EmbedFC(1, 2 * n_feat)
        self.temb2 = EmbedFC(1, n_feat)
        self.cemb1 = EmbedFC(cond_dim, 2 * n_feat)
        self.cemb2 = EmbedFC(cond_dim, n_feat)
        
        # Decoder
        self.up1 = UnetUp(up_in_ch=2*n_feat, skip_ch=n_feat, out_ch=n_feat)
        self.up2 = UnetUp(up_in_ch=n_feat, skip_ch=n_feat, out_ch=n_feat)
        
        # Output head
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )
    
    def forward(self, x, c_vec, t_scaled, context_mask):
        """
        Args:
            x: (B, C, 28, 28) noisy images
            c_vec: (B, D) conditioning vectors
            t_scaled: (B, 1) timestep in [0, 1]
            context_mask: (B,) binary mask for classifier-free guidance
        Returns:
            noise_pred: (B, C, 28, 28) predicted noise
        """
        B = x.size(0)
        
        # Encoder
        x0 = self.init(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        
        # Classifier-free guidance: mask context
        c_masked = c_vec * (1.0 - context_mask.float().unsqueeze(-1))
        
        # Embeddings
        c1 = self.cemb1(c_masked).view(B, 2 * self.n_feat, 1, 1)
        t1 = self.temb1(t_scaled).view(B, 2 * self.n_feat, 1, 1)
        c2 = self.cemb2(c_masked).view(B, self.n_feat, 1, 1)
        t2 = self.temb2(t_scaled).view(B, self.n_feat, 1, 1)
        
        # Decoder with conditioning
        d2 = d2 + c1 + t1
        u1 = self.up1(d2, d1)
        u1 = u1 + c2 + t2
        u2 = self.up2(u1, x0)
        
        return self.out(torch.cat([u2, x0], dim=1))


# =============================================================================
# Context U-Net (Latent Space, 8×8)
# =============================================================================

class ContextUnetLatent(nn.Module):
    """U-Net for latent diffusion with vector conditioning."""
    
    def __init__(self, in_channels=4, n_feat=128, cond_dim=128):
        super().__init__()
        self.n_feat = n_feat
        
        # Encoder on 8×8 latent
        self.init = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # Time and context embeddings
        self.temb1 = EmbedFC(1, 2 * n_feat)
        self.temb2 = EmbedFC(1, n_feat)
        self.cemb1 = EmbedFC(cond_dim, 2 * n_feat)
        self.cemb2 = EmbedFC(cond_dim, n_feat)
        
        # Decoder
        self.up1 = UnetUp(up_in_ch=2*n_feat, skip_ch=n_feat, out_ch=n_feat)
        self.up2 = UnetUp(up_in_ch=n_feat, skip_ch=n_feat, out_ch=n_feat)
        
        # Output head
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )
    
    def forward(self, x, c_vec, t_scaled, context_mask):
        """
        Args:
            x: (B, C, 8, 8) noisy latents
            c_vec: (B, D) conditioning vectors
            t_scaled: (B, 1) timestep in [0, 1]
            context_mask: (B,) binary mask
        Returns:
            noise_pred: (B, C, 8, 8) predicted noise
        """
        B = x.size(0)
        
        # Encoder
        x0 = self.init(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        
        # Classifier-free guidance
        c_masked = c_vec * (1.0 - context_mask.float().unsqueeze(-1))
        
        # Embeddings
        c1 = self.cemb1(c_masked).view(B, 2 * self.n_feat, 1, 1)
        t1 = self.temb1(t_scaled).view(B, 2 * self.n_feat, 1, 1)
        c2 = self.cemb2(c_masked).view(B, self.n_feat, 1, 1)
        t2 = self.temb2(t_scaled).view(B, self.n_feat, 1, 1)
        
        # Decoder with conditioning
        d2 = d2 + c1 + t1
        u1 = self.up1(d2, d1)
        u1 = u1 + c2 + t2
        u2 = self.up2(u1, x0)
        
        return self.out(torch.cat([u2, x0], dim=1))


# =============================================================================
# Noise Schedules
# =============================================================================

def ddpm_schedules(beta1, beta2, T):
    """
    Compute DDPM noise schedules.
    
    Args:
        beta1: Starting beta
        beta2: Ending beta
        T: Number of diffusion steps
    Returns:
        Dictionary of schedule tensors
    """
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1.0 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1.0 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1.0 - alphabar_t)
    mab_over_sqrtmab = (1.0 - alpha_t) / torch.clamp(sqrtmab, min=1e-12)
    
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab
    }


# =============================================================================
# DDPM (Pixel Space)
# =============================================================================

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for pixel space."""
    
    def __init__(self, nn_model, betas=(1e-4, 0.02), n_diffusion_steps=200,
                 device="cpu", drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.device = device
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.drop_prob = float(drop_prob)
        self.loss_mse = nn.MSELoss()
        
        # Register noise schedules as buffers
        sched = ddpm_schedules(betas[0], betas[1], self.n_diffusion_steps)
        for k, v in sched.items():
            self.register_buffer(k, v)
    
    def forward(self, x, c_vec):
        """
        Training forward pass.
        
        Args:
            x: (B, C, H, W) clean images
            c_vec: (B, D) conditioning vectors
        Returns:
            loss: MSE between predicted and true noise
        """
        B = x.size(0)
        
        # Random timesteps
        t_int = torch.randint(1, self.n_diffusion_steps, (B,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(x)
        x_t = self.sqrtab[t_int, None, None, None] * x + self.sqrtmab[t_int, None, None, None] * noise
        
        # Classifier-free guidance: randomly drop conditioning
        context_mask = torch.bernoulli(
            torch.full((B,), self.drop_prob, dtype=torch.float32, device=self.device)
        ).long()
        
        # Predict noise
        t_scaled = (t_int / self.n_diffusion_steps).float().view(-1, 1)
        pred = self.nn_model(x_t, c_vec, t_scaled, context_mask)
        
        return self.loss_mse(pred, noise)
    
    @torch.no_grad()
    def sample(self, n_samples, size, c_vec, guide_weight=2.0):
        """
        Sample from the model using classifier-free guidance.
        
        Args:
            n_samples: Number of samples
            size: (C, H, W) size of each sample
            c_vec: (B, D) or (1, D) conditioning vectors
            guide_weight: Guidance scale (higher = stronger conditioning)
        Returns:
            x: (B, C, H, W) generated samples
        """
        device = self.device
        x = torch.randn(n_samples, *size, device=device)
        
        if c_vec.size(0) == 1:
            c_vec = c_vec.repeat(n_samples, 1)
        
        for i in range(self.n_diffusion_steps, 0, -1):
            t_scaled = torch.full((n_samples, 1), i / self.n_diffusion_steps, device=device)
            
            # Duplicate for conditional and unconditional
            x_in = x.repeat(2, 1, 1, 1)
            c_in = torch.cat([c_vec, c_vec], dim=0)
            context_mask = torch.zeros(n_samples * 2, device=device).long()
            context_mask[n_samples:] = 1  # Mask second half (uncond)
            
            # Predict noise with guidance
            eps = self.nn_model(x_in, c_in, t_scaled.repeat(2, 1), context_mask)
            eps1, eps2 = eps[:n_samples], eps[n_samples:]
            eps = (1 + guide_weight) * eps1 - guide_weight * eps2
            
            # Denoise step
            z = torch.randn_like(x) if i > 1 else 0.0
            x = self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        
        return x
    
    @torch.no_grad()
    def dual_oscillation(self, size, class_a, class_b, guide_weight=2.0,
                         flip_every=50, n_classes=10):
        """
        🌀 Creative feature: Generate image that oscillates between two classes.
        Uses 180° rotation to encourage blending between concepts.
        
        Args:
            size: (C, H, W) output size
            class_a: First class ID
            class_b: Second class ID
            guide_weight: Guidance strength
            flip_every: Rotate and switch class every N steps
            n_classes: Total number of classes
        Returns:
            x: (1, C, H, W) oscillated image
        """
        device = self.device
        x = torch.randn(1, *size, device=device)
        current_class = int(class_a)
        
        for i in range(self.n_diffusion_steps, 0, -1):
            t_scaled = torch.full((1, 1), i / self.n_diffusion_steps, device=device)
            
            # Use current class as condition
            labels = torch.tensor([current_class], device=device, dtype=torch.long)
            c_vec = F.one_hot(labels, num_classes=n_classes).float()
            
            # Predict with guidance
            x_in = x.repeat(2, 1, 1, 1)
            c_in = c_vec.repeat(2, 1)
            context_mask = torch.tensor([0, 1], device=device, dtype=torch.long)
            
            eps = self.nn_model(x_in, c_in, t_scaled.repeat(2, 1), context_mask)
            eps1, eps2 = eps[:1], eps[1:]
            eps = (1 + guide_weight) * eps1 - guide_weight * eps2
            
            # Denoise
            z = torch.randn_like(x) if i > 1 else 0.0
            x = self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
            # 🔄 Magic: rotate 180° and switch class periodically
            if i % flip_every == 0 and i != self.n_diffusion_steps:
                x = torch.rot90(x, k=2, dims=(2, 3))  # 180° rotation
                current_class = class_b if current_class == class_a else class_a
        
        return x


# =============================================================================
# LDM (Latent Diffusion)
# =============================================================================

class LDM(nn.Module):
    """Latent Diffusion Model with DDIM sampling support."""
    
    def __init__(self, nn_model, betas=(1e-4, 0.02), n_diffusion_steps=200,
                 device="cpu", drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.device = device
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.drop_prob = float(drop_prob)
        self.loss_mse = nn.MSELoss()
        
        sched = ddpm_schedules(betas[0], betas[1], self.n_diffusion_steps)
        for k, v in sched.items():
            self.register_buffer(k, v)
    
    def forward(self, z, c_vec):
        """Training forward pass on latents."""
        B = z.size(0)
        t_int = torch.randint(1, self.n_diffusion_steps, (B,), device=self.device)
        noise = torch.randn_like(z)
        z_t = self.sqrtab[t_int, None, None, None] * z + self.sqrtmab[t_int, None, None, None] * noise
        
        context_mask = torch.bernoulli(
            torch.full((B,), self.drop_prob, dtype=torch.float32, device=self.device)
        ).long()
        
        t_scaled = (t_int / self.n_diffusion_steps).float().view(-1, 1)
        pred = self.nn_model(z_t, c_vec, t_scaled, context_mask)
        
        return self.loss_mse(pred, noise)
    
    @torch.no_grad()
    def sample(self, n_samples, size, c_vec, guide_weight=2.0):
        """DDPM sampling in latent space."""
        device = self.device
        x = torch.randn(n_samples, *size, device=device)
        
        if c_vec.size(0) == 1:
            c_vec = c_vec.repeat(n_samples, 1)
        
        for i in range(self.n_diffusion_steps, 0, -1):
            t_scaled = torch.full((n_samples, 1), i / self.n_diffusion_steps, device=device)
            
            x_in = x.repeat(2, 1, 1, 1)
            c_in = torch.cat([c_vec, c_vec], dim=0)
            context_mask = torch.zeros(n_samples * 2, device=device).long()
            context_mask[n_samples:] = 1
            
            pred = self.nn_model(x_in, c_in, t_scaled.repeat(2, 1), context_mask)
            pred1, pred2 = pred[:n_samples], pred[n_samples:]
            pred = (1 + guide_weight) * pred1 - guide_weight * pred2
            
            z = torch.randn_like(x) if i > 1 else 0.0
            x = self.oneover_sqrta[i] * (x - pred * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        
        return x
    
    @torch.no_grad()
    def sample_ddim(self, n_samples, size, c_vec, guide_weight=2.0, eta=0.0, ddim_steps=50):
        """
        Fast DDIM sampling (deterministic when eta=0).
        
        Args:
            n_samples: Number of samples
            size: (C, H, W) latent size
            c_vec: Conditioning
            guide_weight: Guidance scale
            eta: Stochasticity (0=deterministic)
            ddim_steps: Number of sampling steps (< n_diffusion_steps for speed)
        Returns:
            x: Generated latents
        """
        device = self.device
        S = int(self.n_diffusion_steps)
        ddim_steps = int(max(1, min(ddim_steps, S)))
        
        # Evenly spaced timesteps
        idx = torch.linspace(0, S - 1, steps=ddim_steps, device=device).long()
        t_seq = (S - idx).long()  # [S, S-1, ..., 1]
        
        x = torch.randn(n_samples, *size, device=device)
        if c_vec.size(0) == 1:
            c_vec = c_vec.repeat(n_samples, 1)
        
        for i in range(ddim_steps):
            t_cur = int(t_seq[i].item())
            t_next = int(t_seq[i + 1].item()) if i + 1 < ddim_steps else 0
            
            alpha_cur = self.alphabar_t[t_cur]
            alpha_next = self.alphabar_t[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            
            # DDIM sigma
            sigma = eta * torch.sqrt(
                ((1 - alpha_next) / (1 - alpha_cur)) * (1 - alpha_cur / alpha_next)
            ).clamp(min=0)
            
            # Predict noise with guidance
            t_scaled = (torch.tensor(t_cur, device=device, dtype=torch.float32) / S).repeat(n_samples, 1)
            x_in = x.repeat(2, 1, 1, 1)
            c_in = torch.cat([c_vec, c_vec], dim=0)
            cmask = torch.zeros(n_samples * 2, device=device, dtype=torch.long)
            cmask[n_samples:] = 1
            
            pred = self.nn_model(x_in, c_in, t_scaled.repeat(2, 1), cmask)
            pred1, pred2 = pred[:n_samples], pred[n_samples:]
            pred = (1 + guide_weight) * pred1 - guide_weight * pred2
            
            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_cur) * pred) / torch.sqrt(alpha_cur)
            x_dir = torch.sqrt(1 - alpha_next - sigma**2) * pred
            x = torch.sqrt(alpha_next) * x0_pred + x_dir + sigma * torch.randn_like(x)
        
        return x
    
    @torch.no_grad()
    def dual_oscillation(self, size, class_a, class_b, guide_weight=2.0,
                         flip_every=50, n_classes=10):
        """🌀 Creative oscillation in latent space with 180° rotation."""
        device = self.device
        x = torch.randn(1, *size, device=device)
        current_class = int(class_a)
        
        for i in range(self.n_diffusion_steps, 0, -1):
            t_scaled = torch.full((1, 1), i / self.n_diffusion_steps, device=device)
            
            labels = torch.tensor([current_class], device=device, dtype=torch.long)
            c_vec = F.one_hot(labels, num_classes=n_classes).float()
            
            x_in = x.repeat(2, 1, 1, 1)
            c_in = c_vec.repeat(2, 1)
            context_mask = torch.tensor([0, 1], device=device, dtype=torch.long)
            
            pred = self.nn_model(x_in, c_in, t_scaled.repeat(2, 1), context_mask)
            pred1, pred2 = pred[:1], pred[1:]
            pred = (1 + guide_weight) * pred1 - guide_weight * pred2
            
            z = torch.randn_like(x) if i > 1 else 0.0
            x = self.oneover_sqrta[i] * (x - pred * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
            # 🔄 Magic rotation
            if i % flip_every == 0 and i != self.n_diffusion_steps:
                x = torch.rot90(x, k=2, dims=(2, 3))
                current_class = class_b if current_class == class_a else class_a
        
        return x
