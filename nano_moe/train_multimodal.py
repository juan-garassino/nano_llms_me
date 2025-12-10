"""
Multimodal Joint Training Script for nano_moe.

Train a unified model with 4 objectives:
1. CLIP contrastive (image ↔ text)
2. Image captioning (image → text)
3. Text-to-image diffusion (text → image)
4. Text-to-text LM (text → text)

Includes all demo functions:
- text_to_image
- image_to_text
- image_to_image (with text condition)
- text_to_text
- train_joint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms as T
from tqdm import tqdm
from rich.console import Console
import random
import os
import json

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Decoder-Only Language Model (for text-to-text)
# =============================================================================

class DecoderOnlyLM(nn.Module):
    """GPT-style decoder-only language model for text generation."""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, 
                 max_len=16, dropout=0.1, tie_weights=True):
        super().__init__()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.max_len = max_len
        
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_weights:
            self.out.weight = self.tok.weight
    
    def forward(self, y_inp):
        """Forward with causal masking."""
        B, L = y_inp.shape
        x = self.tok(y_inp) + self.pos[:, :L]
        causal = torch.triu(torch.ones(L, L, device=y_inp.device), diagonal=1).bool()
        h = self.enc(x, mask=causal)
        h = self.ln(h)
        return self.out(h)
    
    def loss(self, dec_tok):
        """Teacher forcing loss."""
        y_inp = dec_tok[:, :-1]
        y_tgt = dec_tok[:, 1:]
        logits = self.forward(y_inp)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_tgt.reshape(-1),
            ignore_index=self.pad_id
        )
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=16, temperature=1.0, top_k=0, top_p=1.0):
        """Autoregressive generation."""
        device = next(self.parameters()).device
        y = prompt_ids.clone().unsqueeze(0).to(device)
        
        for _ in range(max_new_tokens):
            if y.size(1) > self.max_len:
                y = y[:, -self.max_len:]
            
            logits = self.forward(y)[:, -1, :] / max(1e-6, float(temperature))
            probs = torch.softmax(logits, dim=-1)
            
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                filt = torch.zeros_like(probs).scatter_(1, topk_idx, topk_vals)
                probs = filt / (filt.sum(dim=-1, keepdim=True) + 1e-9)
            
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-9)
                idx_next = torch.multinomial(sorted_probs, num_samples=1)
                token = sorted_idx.gather(-1, idx_next)
            else:
                token = torch.multinomial(probs, num_samples=1)
            
            y = torch.cat([y, token], dim=1)
            if token.item() == self.eos_id:
                break
        
        return y.squeeze(0)


# =============================================================================
# Extended Multimodal Model with Text-to-Text
# =============================================================================

class ExtendedMultimodal(nn.Module):
    """
    Complete multimodal model with 4 capabilities:
    - CLIP (bi-directional image ↔ text)
    - Captioning (image → text)
    - Diffusion (text → image)
    - LM (text → text)
    """
    
    def __init__(self, vocab_size, emb_dim=128, max_len=16,
                 steps=200, drop_prob=0.1, use_latent=False, latent_ch=4):
        super().__init__()
        self.use_latent = use_latent
        self.pad_id = 0
        
        # Import models from nano_moe
        from nano_moe.models.clip import EnhancedImageEncoder, EnhancedTextEncoder, CaptionDecoder
        from nano_moe.models.diffusion import ContextUnetPixel, ContextUnetLatent, DDPM, LDM
        from nano_moe.models.vae import VAE
        
        # Vision-language
        self.img_enc = EnhancedImageEncoder(emb_dim)
        self.txt_enc = EnhancedTextEncoder([], emb_dim, max_len=max_len)
        self.txt_enc.vocab_size = vocab_size
        self.cap_dec = CaptionDecoder(vocab_size, d_model=emb_dim, max_len=max_len)
        
        # Decoder-only LM for text→text
        self.lm = DecoderOnlyLM(vocab_size, d_model=emb_dim, max_len=max_len)
        
        # Learnable temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))
        
        # Diffusion
        if use_latent:
            self.vae = VAE(latent_channels=latent_ch)
            self.unet = ContextUnetLatent(in_channels=latent_ch, n_feat=emb_dim, cond_dim=emb_dim)
            self.diff = LDM(self.unet, n_diffusion_steps=steps, drop_prob=drop_prob, device=DEVICE)
        else:
            self.unet = ContextUnetPixel(in_channels=1, n_feat=emb_dim, cond_dim=emb_dim)
            self.diff = DDPM(self.unet, n_diffusion_steps=steps, drop_prob=drop_prob, device=DEVICE)
    
    def contrastive_loss(self, img_emb, txt_emb):
        """CLIP loss."""
        temp = self.log_temp.exp()
        logits = img_emb @ txt_emb.T / temp
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2, temp
    
    def caption_loss(self, img_emb, dec_tok):
        """Image captioning loss."""
        y_inp, y_tgt = dec_tok[:, :-1], dec_tok[:, 1:]
        logits = self.cap_dec(img_emb, y_inp)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_tgt.reshape(-1),
            ignore_index=self.pad_id
        )
    
    def diffusion_loss(self, imgs, txt_emb):
        """Text-to-image diffusion loss."""
        if self.use_latent:
            with torch.no_grad():
                mu, _ = self.vae.encode(imgs)
            return self.diff(mu, txt_emb)
        else:
            return self.diff(imgs, txt_emb)
    
    def lm_loss(self, dec_tok):
        """Language modeling loss."""
        return self.lm.loss(dec_tok)
    
    def forward_losses(self, imgs, enc_tok, dec_tok,
                       lam_clip=1.0, lam_cap=1.0, lam_diff=1.0, lam_lm=1.0):
        """Compute all 4 losses jointly."""
        # Encodings
        img_emb = self.img_enc(imgs)
        txt_emb = self.txt_enc(enc_tok)
        
        # Four losses
        l_clip, temp = self.contrastive_loss(img_emb, txt_emb)
        l_cap = self.caption_loss(img_emb, dec_tok)
        l_diff = self.diffusion_loss(imgs, txt_emb)
        l_lm = self.lm_loss(dec_tok)
        
        total = lam_clip * l_clip + lam_cap * l_cap + lam_diff * l_diff + lam_lm * l_lm
        
        return {
            "total": total,
            "clip": l_clip,
            "cap": l_cap,
            "diff": l_diff,
            "lm": l_lm,
            "temp": temp,
            "img_emb": img_emb.detach(),
            "txt_emb": txt_emb.detach()
        }


# =============================================================================
# Simple Fashion Dataset with Captions
# =============================================================================

class FashionCaptionDataset(Dataset):
    """FashionMNIST with generated captions."""
    
    CLASS_NAMES = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    
    def __init__(self, split="train", img_size=32, max_len=16):
        self.max_len = max_len
        
        tfm = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        self.ds = tv.datasets.FashionMNIST(
            "./data", train=(split == "train"), download=True, transform=tfm
        )
        
        # Build vocabulary
        captions = []
        for _, y in self.ds:
            caption = f"a stylish {self.CLASS_NAMES[y]}"
            captions.append(caption)
        
        all_words = set()
        for cap in captions:
            all_words.update(cap.split())
        
        self.vocab = sorted(all_words)
        self.stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        for w in self.vocab:
            if w not in self.stoi:
                self.stoi[w] = len(self.stoi)
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, y = self.ds[idx]
        caption = f"a stylish {self.CLASS_NAMES[y]}"
        
        # Tokenize for encoding (no special tokens)
        enc_toks = [self.stoi.get(w, 0) for w in caption.split()]
        enc_toks = enc_toks[:self.max_len] + [0] * (self.max_len - len(enc_toks))
        
        # Tokenize for decoding (with BOS/EOS)
        dec_toks = [self.stoi["<bos>"]] + [self.stoi.get(w, 0) for w in caption.split()] + [self.stoi["<eos>"]]
        dec_toks = dec_toks[:self.max_len] + [0] * (self.max_len - len(dec_toks))
        
        return img, y, caption, torch.tensor(enc_toks, dtype=torch.long), torch.tensor(dec_toks, dtype=torch.long)


# =============================================================================
# Joint Training Function
# =============================================================================

def train_joint(epochs=2, batch_size=128, emb_dim=128, steps=200,
                lam_clip=1.0, lam_cap=1.0, lam_diff=1.0, lam_lm=1.0,
                lr=2e-4, max_len=16, use_latent=False, latent_ch=4,
                vae=None, img_size=28):
    """
    Train joint multimodal model.
    
    Args:
        epochs: Training epochs
        batch_size: Batch size
        emb_dim: Embedding dimension
        steps: Diffusion steps
        lam_*: Loss weights
        lr: Learning rate
        max_len: Max sequence length
        use_latent: Use latent diffusion (True) or pixel DDPM (False)
        latent_ch: Latent channels
        vae: Pre-trained VAE (required if use_latent=True)
        img_size: Image size
    Returns:
        model, train_ds, val_ds
    """
    console.print(f"[bold magenta]Joint Multimodal Training (mode={'LATENT' if use_latent else 'PIXEL'})[/bold magenta]")
    
    # Load datasets
    train_ds = FashionCaptionDataset(split="train", img_size=img_size, max_len=max_len)
    val_ds = FashionCaptionDataset(split="test", img_size=img_size, max_len=max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    
    # Create model
    model = ExtendedMultimodal(
        vocab_size=train_ds.vocab_size,
        emb_dim=emb_dim,
        max_len=max_len,
        steps=steps,
        use_latent=use_latent,
        latent_ch=latent_ch
    ).to(DEVICE)
    
    # Load pre-trained VAE if latent mode
    if use_latent:
        assert vae is not None, "VAE required for latent mode"
        model.vae.load_state_dict(vae.state_dict())
        model.vae.eval()
        for p in model.vae.parameters():
            p.requires_grad = False
    
    # Optimizer
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    
    best_val = float('inf')
    
    for ep in range(1, epochs + 1):
        # Training
        model.train()
        train_metrics = {"total": 0, "clip": 0, "cap": 0, "diff": 0, "lm": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for imgs, ys, caps, enc_tok, dec_tok in pbar:
            imgs, enc_tok, dec_tok = imgs.to(DEVICE), enc_tok.to(DEVICE), dec_tok.to(DEVICE)
            
            # Forward
            out = model.forward_losses(
                imgs, enc_tok, dec_tok,
                lam_clip=lam_clip, lam_cap=lam_cap, lam_diff=lam_diff, lam_lm=lam_lm
            )
            
            # Backward
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            opt.step()
            sched.step()
            
            # Track
            for k in train_metrics:
                train_metrics[k] += out[k].item()
            
            pbar.set_postfix(
                total=f"{out['total'].item():.3f}",
                clip=f"{out['clip'].item():.3f}",
                cap=f"{out['cap'].item():.3f}",
                diff=f"{out['diff'].item():.3f}",
                lm=f"{out['lm'].item():.3f}",
                temp=f"{out['temp'].item():.3f}"
            )
        
        # Log train
        n = len(train_loader)
        console.print(
            f"[cyan]Train: total={train_metrics['total']/n:.3f}, "
            f"clip={train_metrics['clip']/n:.3f}, cap={train_metrics['cap']/n:.3f}, "
            f"diff={train_metrics['diff']/n:.3f}, lm={train_metrics['lm']/n:.3f}[/cyan]"
        )
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, ys, caps, enc_tok, dec_tok in val_loader:
                imgs, enc_tok, dec_tok = imgs.to(DEVICE), enc_tok.to(DEVICE), dec_tok.to(DEVICE)
                out = model.forward_losses(
                    imgs, enc_tok, dec_tok,
                    lam_clip=lam_clip, lam_cap=lam_cap, lam_diff=lam_diff, lam_lm=lam_lm
                )
                val_loss += out["total"].item()
        
        val_loss /= len(val_loader)
        console.print(f"[yellow]Val: total={val_loss:.3f}[/yellow]")
        
        # Save best
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("results/multimodal", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "meta": {
                    "use_latent": use_latent,
                    "emb_dim": emb_dim,
                    "max_len": max_len,
                    "steps": steps,
                    "latent_ch": latent_ch,
                    "vocab_size": train_ds.vocab_size,
                    "stoi": train_ds.stoi,
                    "itos": train_ds.itos
                }
            }, "results/multimodal/nano_multimodal.pt")
            
            # Save training log
            with open("results/multimodal/training_log.json", "w") as f:
                json.dump({
                    "epoch": ep,
                    "best_val_loss": best_val,
                    "train_metrics": {k: v/n for k, v in train_metrics.items()},
                    "val_loss": val_loss
                }, f, indent=2)
            
            console.print("[green]✓ Saved best checkpoint to results/multimodal/[/green]")
    
    return model, train_ds, val_ds


# =============================================================================
# Demo Functions
# =============================================================================

@torch.no_grad()
def text_to_image(model, text, stoi, max_len=16, n=8, guide_w=2.0,
                  use_ddim=False, ddim_steps=50):
    """Generate images from text."""
    console.print(f'[yellow]Text→Image: "{text}"[/yellow]')
    
    # Tokenize
    ids = [stoi.get(t, 0) for t in text.lower().split()]
    ids = ids[:max_len] + [stoi.get("<pad>", 0)] * (max_len - len(ids))
    enc_tok = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Encode
    txt_emb = model.txt_enc.forward(enc_tok.repeat(n, 1))
    
    # Generate
    if model.use_latent:
        latent_size = (model.vae.latent_channels, 8, 8)
        if use_ddim:
            latents = model.diff.sample_ddim(n, latent_size, txt_emb, guide_w, ddim_steps=ddim_steps)
        else:
            latents = model.diff.sample(n, latent_size, txt_emb, guide_w)
        imgs = model.vae.decode(latents)
    else:
        imgs = model.diff.sample(n, (1, 28, 28), txt_emb, guide_w)
    
    # Display
    show_grid(imgs, f'"{ text}"')
    return imgs


@torch.no_grad()
def image_to_text(model, imgs, itos):
    """Generate captions from images."""
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    
    imgs = imgs.to(DEVICE)
    emb = model.img_enc(imgs)
    seq = model.cap_dec.greedy(emb, max_len=model.cap_dec.max_len)
    
    captions = []
    for row in seq:
        words = []
        for tid in row.tolist():
            if tid in (model.cap_dec.bos_id, model.cap_dec.eos_id, model.cap_dec.pad_id):
                continue
            words.append(itos.get(tid, ""))
        captions.append(" ".join([w for w in words if w]))
    
    for i, cap in enumerate(captions):
        console.print(f"  [{i}] {cap}")
    
    return captions


@torch.no_grad()
def image_to_image(model, src_img, stoi, text=None, w_img=0.7, w_txt=0.3,
                   strength=0.6, guide_w=2.0, use_ddim=True, ddim_steps=50):
    """Transform image with optional text condition."""
    console.print(f'[yellow]Image→Image (strength={strength}' + (f', text="{text}"' if text else '') + ')[/yellow]')
    
    # Prepare image
    if src_img.ndim == 3 and src_img.size(0) == 1:
        x0 = src_img.unsqueeze(0).to(DEVICE)
    else:
        x0 = src_img.unsqueeze(0).to(DEVICE)
    
    # Conditioning
    img_emb = model.img_enc(x0)
    cond = img_emb
    
    if text and w_txt > 0:
        ids = [stoi.get(t, 0) for t in text.lower().split()]
        ids = ids[:model.cap_dec.max_len] + [stoi.get("<pad>", 0)] * (model.cap_dec.max_len - len(ids))
        enc_tok = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        txt_emb = model.txt_enc.forward(enc_tok)
        cond = F.normalize(w_img * img_emb + w_txt * txt_emb, dim=-1)
    
    # Noise and denoise
    S = model.diff.n_diffusion_steps
    t_start = max(1, int(round(strength * (S - 1))))
    
    if model.use_latent:
        mu, _ = model.vae.encode(x0)
        z_t = model.diff.q_sample(mu, t_start)
        if use_ddim:
            z_out = model.diff.sample_ddim_i2i(z_t, t_start, cond, guide_w, ddim_steps=ddim_steps)
        else:
            z_out = model.diff.sample_i2i(z_t, t_start, cond, guide_w)
        imgs = model.vae.decode(z_out)
    else:
        x_t = model.diff.q_sample(x0, t_start)
        imgs = model.diff.sample_i2i(x_t, t_start, cond, guide_w)
    
    show_grid(imgs, f'I2I (str={strength}, guide={guide_w})')
    return imgs


@torch.no_grad()
def text_to_text(model, prompt, stoi, itos, max_len=16, max_new_tokens=12,
                 temperature=1.0, top_k=0, top_p=1.0):
    """Generate text continuation."""
    console.print(f'[yellow]Text→Text: "{prompt}"[/yellow]')
    
    toks = ["<bos>"] + prompt.lower().split()
    ids = [stoi.get(t, 0) for t in toks][:max_len]
    y0 = torch.tensor(ids, dtype=torch.long, device=DEVICE)
    
    out_ids = model.lm.generate(
        y0, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p
    ).tolist()
    
    words = []
    for tid in out_ids:
        if tid in (model.lm.bos_id, model.lm.pad_id):
            continue
        if tid == model.lm.eos_id:
            break
        words.append(itos.get(tid, ""))
    
    result = " ".join([w for w in words if w])
    console.print(f'  → "{result}"')
    return result


def show_grid(imgs, title="", save_dir="results/multimodal"):
    """Display and save image grid."""
    os.makedirs(save_dir, exist_ok=True)
    
    imgs = imgs.detach().cpu()
    grid = tv.utils.make_grid(imgs, nrow=min(8, imgs.size(0)), normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("→", "to").replace("↔", "oscillate").replace('"', "").replace("/", "_")
    save_path = os.path.join(save_dir, f"{safe_title}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()  # Close to free memory
    
    console.print(f"[green]💾 Saved: {save_path}[/green]")


# =============================================================================
# Main
# =============================================================================

def main(use_latent=False, epochs=2):
    """Main multimodal training."""
    console.print("[bold magenta]NanoMultimodal Training[/bold magenta]")
    
    img_size = 32 if use_latent else 28
    
    # Train VAE if latent mode
    vae = None
    if use_latent:
        from nano_moe.train_diffusion import train_vae
        vae = train_vae(epochs=2, img_size=img_size)
    
    # Joint training
    model, train_ds, val_ds = train_joint(
        epochs=epochs,
        use_latent=use_latent,
        vae=vae,
        img_size=img_size
    )
    
    # Demos
    stoi, itos = train_ds.stoi, train_ds.itos
    
    console.print("\n[bold green]Demo: Text→Image[/bold green]")
    text_to_image(model, "a stylish sneaker", stoi, n=6, guide_w=2.0, use_ddim=use_latent)
    
    console.print("\n[bold green]Demo: Image→Text[/bold green]")
    sample_imgs, *_ = next(iter(DataLoader(val_ds, batch_size=4)))
    image_to_text(model, sample_imgs, itos)
    
    console.print("\n[bold green]Demo: Image→Image[/bold green]")
    image_to_image(model, sample_imgs[0], stoi, text="a clear bag", strength=0.6, guide_w=2.5, use_ddim=use_latent)
    
    console.print("\n[bold green]Demo: Text→Text[/bold green]")
    text_to_text(model, "a stylish", stoi, itos, max_new_tokens=8, temperature=0.9, top_k=20)
    
    console.print("[green]✅ Multimodal training complete![/green]")


if __name__ == "__main__":
    main(use_latent=False, epochs=1)
