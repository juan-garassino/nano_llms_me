"""
Diffusion Training Script for nano_moe.

Train DDPM or Latent Diffusion models with text conditioning.
Includes demo functions for:
- text_to_image generation
- image_to_image transformation
- dual_oscillation (creative 180° rotation feature!)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from tqdm import tqdm
from rich.console import Console
import os

from nano_moe.models.clip import EnhancedTextEncoder
from nano_moe.models.diffusion import ContextUnetPixel, ContextUnetLatent, DDPM, LDM
from nano_moe.models.vae import VAE

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Fashion class names for text conditioning
FASHION_CLASS_TEXTS = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot"
]


def train_vae(epochs=2, batch_size=128, lr=1e-3, dataset="fashion", device=DEVICE):
    """Train VAE for latent diffusion."""
    console.print("[blue]Training VAE...[/blue]")
    
    tfm = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = tv.datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    
    vae = VAE().to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for ep in range(1, epochs + 1):
        vae.train()
        pbar = tqdm(loader, desc=f"VAE Epoch {ep}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            
            recon_loss = F.mse_loss(recon, x, reduction='mean')
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = recon_loss + 0.001 * kl_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    os.makedirs("results/diffusion", exist_ok=True)
    torch.save(vae.state_dict(), "results/diffusion/vae.pt")
    console.print("[green]✓ VAE saved to results/diffusion/vae.pt[/green]")
    return vae


def train_diffusion(use_latent=False, epochs=2, batch_size=128, lr=1e-4, 
                     steps=200, n_feat=128, guide_weight=2.0, vae=None, device=DEVICE):
    """Train DDPM or Latent Diffusion."""
    console.print(f"[blue]Training {'Latent Diffusion' if use_latent else 'DDPM'}...[/blue]")
    
    tfm = T.Compose([
        T.Resize(32 if use_latent else 28), 
        T.ToTensor(), 
        T.Normalize((0.5,), (0.5,))
    ])
    ds = tv.datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create model
    if use_latent:
        if vae is None:
            vae = train_vae(epochs=2, device=device)
        vae.eval()
        unet = ContextUnetLatent(in_channels=4, n_feat=n_feat, cond_dim=10)
        diffusion = LDM(unet, n_diffusion_steps=steps, device=device)
    else:
        unet = ContextUnetPixel(in_channels=1, n_feat=n_feat, cond_dim=10)
        diffusion = DDPM(unet, n_diffusion_steps=steps, device=device)
    
    diffusion = diffusion.to(device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=lr)
    
    for ep in range(1, epochs + 1):
        diffusion.train()
        pbar = tqdm(loader, desc=f"Diffusion Epoch {ep}/{epochs}")
        
        for x, c in pbar:
            x, c = x.to(device), c.to(device)
            
            # Convert to one-hot conditioning
            c_vec = F.one_hot(c, num_classes=10).float()
            
            if use_latent:
                with torch.no_grad():
                    mu, _ = vae.encode(x)
                loss = diffusion(mu, c_vec)
            else:
                loss = diffusion(x, c_vec)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Sample
        diffusion.eval()
        with torch.no_grad():
            if use_latent:
                latents = diffusion.sample(
                    n_samples=16, 
                    size=(4, 8, 8), 
                    c_vec=F.one_hot(torch.arange(0, 10, device=device).repeat(2)[:16], 10).float(),
                    guide_weight=guide_weight
                )
                samples = vae.decode(latents)
            else:
                samples = diffusion.sample(
                    n_samples=16,
                    size=(1, 28, 28),
                    c_vec=F.one_hot(torch.arange(0, 10, device=device).repeat(2)[:16], 10).float(),
                    guide_weight=guide_weight
                )
        
        show_grid(samples, f"Epoch {ep}")
    
    os.makedirs("results/diffusion", exist_ok=True)
    torch.save(diffusion.state_dict(), "results/diffusion/diffusion.pt")
    console.print("[green]✓ Diffusion saved to results/diffusion/diffusion.pt[/green]")
    return diffusion, vae if use_latent else None


def text_to_image(query, txt_enc, diffusion, vae=None, n_samples=6, guide_weight=2.0, device=DEVICE):
    """Generate images from text query."""
    console.print(f'[yellow]Query: "{query}"[/yellow]')
    
    # Find closest class
    with torch.no_grad():
        q_emb = txt_enc([query])
        class_embs = txt_enc(FASHION_CLASS_TEXTS)
        sims = (q_emb @ class_embs.T).softmax(dim=-1)
        class_id = sims.argmax().item()
    
    console.print(f"  → Matched class: {FASHION_CLASS_TEXTS[class_id]} (id={class_id})")
    
    # Generate
    c_vec = F.one_hot(torch.tensor([class_id], device=device), 10).float()
    
    if vae is not None:
        latents = diffusion.sample(n_samples, (4, 8, 8), c_vec, guide_weight)
        imgs = vae.decode(latents)
    else:
        imgs = diffusion.sample(n_samples, (1, 28, 28), c_vec, guide_weight)
    
    show_grid(imgs, f'"{query}" → {FASHION_CLASS_TEXTS[class_id]}')
    return imgs


def image_to_image(img, diffusion, target_class, vae=None, guide_weight=2.0, device=DEVICE):
    """Transform image to target class (style transfer)."""
    console.print(f"[yellow]Transforming to class: {target_class}[/yellow]")
    
    # Use noisy version as starting point (image-to-image)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    c_vec = F.one_hot(torch.tensor([target_class], device=device), 10).float()
    
    # Add some noise and denoise
    if vae is not None:
        with torch.no_grad():
            z, _ = vae.encode(img.to(device))
            noise_level = diffusion.n_diffusion_steps // 2
            noise = torch.randn_like(z)
            alpha = diffusion.sqrtab[noise_level]
            z_noisy = alpha * z + (1 - alpha).sqrt() * noise
            
            # Partial denoising
            result = diffusion.sample(1, z.shape[1:], c_vec, guide_weight)
            result = vae.decode(result)
    else:
        # Pixel space
        result = diffusion.sample(1, img.shape[1:], c_vec, guide_weight)
    
    show_grid(result, f"Image→Image (class {target_class})")
    return result


def dual_oscillation(diffusion, class_a=5, class_b=7, vae=None, guide_weight=2.0, device=DEVICE):
    """
    🌀 Creative dual-image oscillation with 180° rotation!
    Generates image that blends two classes.
    """
    console.print(f"[yellow]🌀 Dual oscillation: {FASHION_CLASS_TEXTS[class_a]} ↔ {FASHION_CLASS_TEXTS[class_b]}[/yellow]")
    
    if vae is not None:
        result = diffusion.dual_oscillation(
            size=(4, 8, 8), 
            class_a=class_a, 
            class_b=class_b,
            guide_weight=guide_weight,
            n_classes=10
        )
        result = vae.decode(result)
    else:
        result = diffusion.dual_oscillation(
            size=(1, 28, 28),
            class_a=class_a,
            class_b=class_b,
            guide_weight=guide_weight,
            n_classes=10
        )
    
    show_grid(result, f"Oscillation: {FASHION_CLASS_TEXTS[class_a]} ↔ {FASHION_CLASS_TEXTS[class_b]}")
    return result


def show_grid(imgs, title="Samples", save_dir="results/diffusion"):
    """Display and save image grid."""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    imgs = imgs.detach().cpu()
    grid = tv.utils.make_grid(imgs, nrow=min(8, imgs.size(0)), normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("→", "to").replace("↔", "oscillate").replace('"', "").replace("/", "_")
    save_path = os.path.join(save_dir, f"{safe_title}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()  # Close to free memory in Colab
    
    console.print(f"[green]💾 Saved: {save_path}[/green]")


def main(use_latent=False, epochs=2, guide_weight=2.0):
    """Main diffusion training."""
    console.print("[bold magenta]Diffusion Training[/bold magenta]")
    
    # Train
    diffusion, vae = train_diffusion(
        use_latent=use_latent, 
        epochs=epochs, 
        guide_weight=guide_weight
    )
    
    # Load text encoder for text-to-image
    console.print("\n[bold green]Demo: Text→Image[/bold green]")
    from nano_moe.models.clip import generate_rich_captions
    _, fmnist_caps = generate_rich_captions()
    vocab = sorted(set(' '.join([cap for _, cap in fmnist_caps]).split()))
    txt_enc = EnhancedTextEncoder(vocab, emb_dim=128).to(DEVICE)
    
    queries = ["a stylish sneaker", "an elegant dress", "a clear bag"]
    for q in queries:
        text_to_image(q, txt_enc, diffusion, vae, guide_weight=guide_weight)
    
    # Image-to-image demo
    console.print("\n[bold green]Demo: Image→Image[/bold green]")
    # Load a sample image and transform it
    tfm = T.Compose([T.Resize(32 if use_latent else 28), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = tv.datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)
    sample_img, _ = ds[0]
    image_to_image(sample_img, diffusion, target_class=7, vae=vae, guide_weight=guide_weight)
    
    # Dual oscillation demo
    console.print("\n[bold green]🌀 Demo: Dual Oscillation[/bold green]")
    dual_oscillation(diffusion, class_a=5, class_b=7, vae=vae, guide_weight=guide_weight)
    
    console.print("[green]✅ Diffusion training complete![/green]")


if __name__ == "__main__":
    main(use_latent=False, epochs=2)
