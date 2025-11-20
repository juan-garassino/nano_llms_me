"""
Variational Autoencoder (VAE) for latent diffusion in nano_moe.

Compresses 32×32 images to 8×8×4 latent representations for efficient diffusion.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for learning compressed latent representations."""
    
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder: 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, latent_channels * 2, 1),  # -> mu and logvar
        )
        
        # Decoder: 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh()  # [-1, 1] output
        )
    
    def encode(self, x):
        """
        Encode image to latent parameters.
        
        Args:
            x: (B, 1, 32, 32) input images
        Returns:
            mu: (B, C, 8, 8) mean
            logvar: (B, C, 8, 8) log variance
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent to image.
        
        Args:
            z: (B, C, 8, 8) latent codes
        Returns:
            x: (B, 1, 32, 32) reconstructed images
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full VAE forward pass.
        
        Args:
            x: (B, 1, 32, 32) input images
        Returns:
            recon: (B, 1, 32, 32) reconstructed images
            mu: (B, C, 8, 8) mean
            logvar: (B, C, 8, 8) log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
