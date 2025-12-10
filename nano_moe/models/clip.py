"""
CLIP (Contrastive Language-Image Pre-training) components for nano_moe.

Includes:
- EnhancedImageEncoder: CNN-based vision encoder
- EnhancedTextEncoder: Transformer-based text encoder
- ContrastiveLoss: Learnable temperature contrastive loss
- Caption generation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# =============================================================================
# Caption Generation Utilities
# =============================================================================

def generate_rich_captions():
    """Generate diverse captions with templates and descriptors for MNIST family."""
    
    templates = [
        "a {descriptor} {item}",
        "an image of a {descriptor} {item}",
        "this is a {descriptor} {item}",
        "a {style} {item}",
        "a picture showing a {descriptor} {item}",
        "{item} with {descriptor} appearance",
        "handwritten {item}",
        "a clear {item}",
    ]
    
    descriptors = [
        "clear", "bold", "faint", "thick", "thin", "dark", "light",
        "handwritten", "printed", "stylized", "simple", "complex"
    ]
    
    styles = [
        "stylish", "elegant", "casual", "formal", "modern", "classic"
    ]
    
    # MNIST Digits (0-9)
    mnist_captions = []
    for i in range(10):
        base_items = [f"digit {i}", f"number {i}", f"numeral {i}"]
        for item in base_items:
            for template in templates:
                if "{descriptor}" in template:
                    for desc in descriptors:
                        caption = template.format(descriptor=desc, item=item)
                        mnist_captions.append((i, caption))
                elif "{style}" in template:
                    for style in styles:
                        caption = template.format(style=style, item=item)
                        mnist_captions.append((i, caption))
                else:
                    caption = template.format(item=item)
                    mnist_captions.append((i, caption))
    
    # FashionMNIST (0-9)
    fmnist_items = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    fmnist_captions = []
    for i, item in enumerate(fmnist_items):
        base_items = [item, f"fashion {item}", f"clothing {item}"]
        for base_item in base_items:
            for template in templates:
                if "{descriptor}" in template:
                    for desc in descriptors + styles:
                        caption = template.format(descriptor=desc, item=base_item)
                        fmnist_captions.append((i, caption))
                elif "{style}" in template:
                    for style in styles:
                        caption = template.format(style=style, item=base_item)
                        fmnist_captions.append((i, caption))
                else:
                    caption = template.format(item=base_item)
                    fmnist_captions.append((i, caption))
    
    return mnist_captions, fmnist_captions


# =============================================================================
# Vision Encoder
# =============================================================================

class EnhancedImageEncoder(nn.Module):
    """CNN-based image encoder for CLIP."""
    
    def __init__(self, emb_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) grayscale images
        Returns:
            z: (B, emb_dim) L2-normalized embeddings
        """
        h = self.cnn(x).view(x.size(0), -1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)


# =============================================================================
# Text Encoder
# =============================================================================

class EnhancedTextEncoder(nn.Module):
    """Transformer-based text encoder for CLIP."""
    
    def __init__(self, vocab, emb_dim=128, n_heads=4, n_layers=2, max_len=15):
        super().__init__()
        self.vocab = vocab
        self.stoi = {w: i+1 for i, w in enumerate(vocab)}  # 0=pad
        self.vocab_size = len(self.stoi) + 1
        self.max_len = max_len
        
        self.embed = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.01)
        self.dropout = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim*4,
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def tokenize(self, text):
        """Tokenize text into indices."""
        toks = [self.stoi.get(w.lower(), 0) for w in text.split()]
        toks = toks[:self.max_len] + [0] * (self.max_len - len(toks))
        return torch.tensor(toks, dtype=torch.long)
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings OR tensor of token ids (B, L)
        Returns:
            z: (B, emb_dim) L2-normalized embeddings
        """
        device = self.embed.weight.device
        
        # Handle both string lists and token tensors
        if isinstance(texts, torch.Tensor):
            toks = texts.to(device)
        else:
            toks = torch.stack([self.tokenize(t) for t in texts]).to(device)
        
        # Create attention mask for padding
        mask = (toks == 0)
        
        x = self.embed(toks) + self.pos_embed[:, :toks.size(1)]
        x = self.dropout(x)
        
        h = self.encoder(x, src_key_padding_mask=mask)
        
        # Mean pooling (excluding padding)
        mask_expanded = mask.unsqueeze(-1).expand_as(h)
        h_masked = h.masked_fill(mask_expanded, 0)
        pooled = h_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        
        z = self.fc(pooled)
        return F.normalize(z, dim=-1)


# =============================================================================
# Contrastive Loss
# =============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss with learnable temperature for CLIP training."""
    
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
    
    def forward(self, img_emb, txt_emb):
        """
        Args:
            img_emb: (B, D) L2-normalized image embeddings
            txt_emb: (B, D) L2-normalized text embeddings
        Returns:
            loss: Symmetric contrastive loss
            temp: Current temperature value
        """
        temperature = self.log_temp.exp()
        logits = img_emb @ txt_emb.T / temperature
        labels = torch.arange(len(img_emb), device=img_emb.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2, temperature


# =============================================================================
# Caption Decoder (for image captioning)
# =============================================================================

class CaptionDecoder(nn.Module):
    """Transformer decoder for image-to-text captioning."""
    
    def __init__(self, vocab_size, d_model=128, max_len=16, n_heads=4, n_layers=2):
        super().__init__()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.max_len = max_len
        
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.dec = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.mem_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, memory, y_inp):
        """
        Args:
            memory: (B, D) image embeddings
            y_inp: (B, L) input token ids
        Returns:
            logits: (B, L, V) output logits
        """
        B, L = y_inp.shape
        tgt = self.tok(y_inp) + self.pos[:, :L]
        causal = torch.triu(torch.ones(L, L, device=y_inp.device), diagonal=1).bool()
        mem = self.mem_proj(memory).unsqueeze(1)  # (B, 1, D)
        h = self.dec(tgt, mem, tgt_mask=causal)
        return self.out(h)
    
    @torch.no_grad()
    def greedy(self, memory, max_len=16):
        """Greedy decoding for inference."""
        B = memory.size(0)
        y = torch.full((B, 1), self.bos_id, dtype=torch.long, device=memory.device)
        
        for _ in range(max_len - 1):
            logits = self.forward(memory, y)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            y = torch.cat([y, nxt], dim=1)
            if (nxt == self.eos_id).all():
                break
        
        return y
