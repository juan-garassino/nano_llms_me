"""
Contrastive learning utilities for CLIP-style training in nano_moe.

Includes:
- Data loading with rich captions
- CLIP training loop
- Retrieval metrics (R@1, R@5, R@10)
- Image-to-text and text-to-image retrieval demos
"""

import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms as T
from tqdm import tqdm
from rich.console import Console

console = Console()


def build_clip_loaders(batch_size=256, val_split=0.1, img_size=32):
    """
    Build CLIP training loaders with rich captions.
    
    Args:
        batch_size: Batch size
        val_split: Validation split fraction
        img_size: Image size (32 for CLIP)
    Returns:
        train_loader, val_loader, vocab, train_pairs, val_pairs
    """
    from ..models.clip import generate_rich_captions
    
    # Transforms
    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    val_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    mnist = tv.datasets.MNIST("./data", train=True, download=True, transform=train_tf)
    fmnist = tv.datasets.FashionMNIST("./data", train=True, download=True, transform=train_tf)
    
    # Generate captions
    mnist_captions, fmnist_captions = generate_rich_captions()
    
    # Create (image, caption) pairs
    pairs = []
    for img, y in mnist:
        caps = [cap for cls, cap in mnist_captions if cls == y]
        caption = random.choice(caps)
        pairs.append((img, caption))
    
    for img, y in fmnist:
        caps = [cap for cls, cap in fmnist_captions if cls == y]
        caption = random.choice(caps)
        pairs.append((img, caption))
    
    # Split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
    
    # Dataset wrapper
    class CLIPDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            return self.pairs[idx]
    
    train_ds = CLIPDataset(train_pairs)
    val_ds = CLIPDataset(val_pairs)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Build vocabulary
    all_caps = [cap for _, cap in pairs]
    vocab = sorted(set(' '.join(all_caps).split()))
    
    return train_loader, val_loader, vocab, train_pairs, val_pairs


def compute_retrieval_metrics(img_emb, txt_emb, k=[1, 5, 10]):
    """
    Compute retrieval metrics (R@k).
    
    Args:
        img_emb: (N, D) image embeddings
        txt_emb: (N, D) text embeddings
        k: List of k values for recall@k
    Returns:
        Dictionary of metrics
    """
    batch_size = img_emb.size(0)
    
    # Image-to-text retrieval
    i2t_sims = img_emb @ txt_emb.T
    i2t_ranks = []
    for i in range(batch_size):
        sim = i2t_sims[i]
        sorted_indices = sim.argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)
    
    # Text-to-image retrieval
    t2i_sims = txt_emb @ img_emb.T
    t2i_ranks = []
    for i in range(batch_size):
        sim = t2i_sims[i]
        sorted_indices = sim.argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    # Compute recall@k
    metrics = {}
    for k_val in k:
        i2t_recall = sum(1 for r in i2t_ranks if r <= k_val) / len(i2t_ranks)
        t2i_recall = sum(1 for r in t2i_ranks if r <= k_val) / len(t2i_ranks)
        metrics[f'i2t_r@{k_val}'] = i2t_recall
        metrics[f't2i_r@{k_val}'] = t2i_recall
    
    return metrics


def train_clip(img_enc, txt_enc, criterion, train_loader, val_loader, 
               epochs=5, lr=1e-3, device="cuda", ckpt_path="nanoclip.pt"):
    """
    Train CLIP model.
    
    Args:
        img_enc: Image encoder
        txt_enc: Text encoder
        criterion: Contrastive loss
        train_loader: Training loader
        val_loader: Validation loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device
        ckpt_path: Checkpoint path
    Returns:
        img_enc, txt_enc, best_val_loss
    """
    img_enc = img_enc.to(device)
    txt_enc = txt_enc.to(device)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(
        list(img_enc.parameters()) + list(txt_enc.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        # Training
        img_enc.train()
        txt_enc.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"CLIP Epoch {epoch}/{epochs}")
        for imgs, texts in pbar:
            imgs = imgs.to(device)
            
            img_emb = img_enc(imgs)
            txt_emb = txt_enc(texts)
            
            loss, temp = criterion(img_emb, txt_emb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(img_enc.parameters()) + list(txt_enc.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{temp.item():.3f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        img_enc.eval()
        txt_enc.eval()
        val_loss = 0
        all_img_embs, all_txt_embs = [], []
        
        with torch.no_grad():
            for imgs, texts in val_loader:
                imgs = imgs.to(device)
                
                img_emb = img_enc(imgs)
                txt_emb = txt_enc(texts)
                
                loss, _ = criterion(img_emb, txt_emb)
                val_loss += loss.item()
                
                all_img_embs.append(img_emb.cpu())
                all_txt_embs.append(txt_emb.cpu())
        
        all_img_embs = torch.cat(all_img_embs).to(device)
        all_txt_embs = torch.cat(all_txt_embs).to(device)
        metrics = compute_retrieval_metrics(all_img_embs, all_txt_embs)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        console.print(
            f"[cyan]Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, I2T R@1={metrics['i2t_r@1']:.3f}, "
            f"T2I R@1={metrics['t2i_r@1']:.3f}[/cyan]"
        )
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "img_encoder": img_enc.state_dict(),
                "txt_encoder": txt_enc.state_dict(),
                "criterion": criterion.state_dict()
            }, ckpt_path)
            console.print(f"[green]✓ Saved best CLIP → {ckpt_path}[/green]")
    
    return img_enc, txt_enc, best_val_loss
