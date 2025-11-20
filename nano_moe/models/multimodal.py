"""
Multimodal model for joint vision-language training in nano_moe.

Combines:
- CLIP contrastive learning (image ↔ text alignment)
- Image captioning (image → text)
- Text-to-image generation (text → image via diffusion)

All three losses trained jointly in a single model!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import EnhancedImageEncoder, EnhancedTextEncoder, CaptionDecoder
from .diffusion import ContextUnetPixel, ContextUnetLatent, DDPM, LDM
from .vae import VAE


class NanoMultimodal(nn.Module):
    """
    Joint multimodal model with 3 objectives:
    1. CLIP-style contrastive loss
    2. Image captioning loss
    3. Text-to-image diffusion loss
    """
    
    def __init__(self, vocab_size, emb_dim=128, max_len=16,
                 steps=200, drop_prob=0.1, use_latent=False, latent_ch=4):
        super().__init__()
        self.use_latent = use_latent
        self.pad_id = 0
        
        # Vision-language encoders
        self.img_enc = EnhancedImageEncoder(emb_dim)
        self.txt_enc = EnhancedTextEncoder([], emb_dim, max_len=max_len)  # Vocab set later
        self.txt_enc.vocab_size = vocab_size
        
        # Caption decoder
        self.cap_dec = CaptionDecoder(vocab_size, d_model=emb_dim, max_len=max_len)
        
        # Learnable temperature for contrastive loss
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))
        
        # Diffusion (pixel or latent)
        if use_latent:
            self.vae = VAE(latent_channels=latent_ch)
            self.unet = ContextUnetLatent(in_ch=latent_ch, n_feat=emb_dim, cond_dim=emb_dim)
            self.diffusion = LDM(self.unet, n_diffusion_steps=steps, drop_prob=drop_prob)
        else:
            self.vae = None
            self.unet = ContextUnetPixel(in_channels=1, n_feat=emb_dim, cond_dim=emb_dim)
            self.diffusion = DDPM(self.unet, n_diffusion_steps=steps, drop_prob=drop_prob)
    
    def contrastive_loss(self, img_emb, txt_emb):
        """CLIP-style contrastive loss."""
        temp = self.log_temp.exp()
        logits = img_emb @ txt_emb.T / temp
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2, temp
    
    def caption_loss(self, img_emb, dec_tok):
        """Image captioning loss (teacher forcing)."""
        y_inp = dec_tok[:, :-1]
        y_tgt = dec_tok[:, 1:]
        
        logits = self.cap_dec(img_emb, y_inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_tgt.reshape(-1),
            ignore_index=self.pad_id
        )
        return loss
    
    def diffusion_loss(self, imgs, txt_emb):
        """Text-to-image diffusion loss."""
        if self.use_latent:
            with torch.no_grad():
                mu, _ = self.vae.encode(imgs)
            return self.diffusion(mu, txt_emb)
        else:
            return self.diffusion(imgs, txt_emb)
    
    def forward_losses(self, imgs, enc_tok, dec_tok,
                       lam_clip=1.0, lam_cap=1.0, lam_diff=1.0):
        """
        Compute all three losses jointly.
        
        Args:
            imgs: (B, 1, H, W) images
            enc_tok: (B, L) tokenized captions (no specials)
            dec_tok: (B, L) tokenized captions (with BOS/EOS)
            lam_*: Loss weights
        Returns:
            Dictionary with losses and embeddings
        """
        # Encode
        img_emb = self.img_enc(imgs)
        txt_emb = self.txt_enc(enc_tok)
        
        # Three losses
        l_clip, temp = self.contrastive_loss(img_emb, txt_emb)
        l_cap = self.caption_loss(img_emb, dec_tok)
        l_diff = self.diffusion_loss(imgs, txt_emb)
        
        total = lam_clip * l_clip + lam_cap * l_cap + lam_diff * l_diff
        
        return {
            "total": total,
            "clip": l_clip,
            "cap": l_cap,
            "diff": l_diff,
            "temp": temp,
            "img_emb": img_emb.detach(),
            "txt_emb": txt_emb.detach()
        }
    
    @torch.no_grad()
    def text_to_image(self, text, n_samples=4, guide_weight=2.0, use_ddim=False, ddim_steps=50):
        """Generate images from text."""
        # Tokenize and encode
        enc_tok = self.txt_enc.tokenize(text).unsqueeze(0).to(next(self.parameters()).device)
        txt_emb = self.txt_enc([text])
        
        # Generate in latent or pixel space
        if self.use_latent:
            if use_ddim:
                latents = self.diffusion.sample_ddim(
                    n_samples, (self.vae.latent_channels, 8, 8), txt_emb,
                    guide_weight=guide_weight, ddim_steps=ddim_steps
                )
            else:
                latents = self.diffusion.sample(
                    n_samples, (self.vae.latent_channels, 8, 8), txt_emb,
                    guide_weight=guide_weight
                )
            imgs = self.vae.decode(latents)
        else:
            imgs = self.diffusion.sample(
                n_samples, (1, 28, 28), txt_emb, guide_weight=guide_weight
            )
        
        return imgs
    
    @torch.no_grad()
    def image_to_text(self, imgs, itos):
        """Generate captions for images."""
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        
        emb = self.img_enc(imgs)
        seq = self.cap_dec.greedy(emb, max_len=self.cap_dec.max_len)
        
        # Decode tokens to text
        texts = []
        for row in seq:
            toks = []
            for tid in row.tolist():
                if tid in (self.cap_dec.bos_id, self.cap_dec.eos_id, self.cap_dec.pad_id):
                    continue
                toks.append(itos.get(tid, ""))
            texts.append(" ".join([t for t in toks if t]))
        
        return texts
