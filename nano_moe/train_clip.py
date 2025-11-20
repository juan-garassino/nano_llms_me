"""
CLIP Training Script for nano_moe.

Train vision-language models with contrastive learning.
Includes demo functions for:
- text_to_image retrieval
- image_to_text retrieval
"""

import torch
import matplotlib.pyplot as plt
from rich.console import Console

from nano_moe.models.clip import EnhancedImageEncoder, EnhancedTextEncoder, ContrastiveLoss
from nano_moe.training.contrastive import build_clip_loaders, train_clip

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text_to_image(query, img_enc, txt_enc, pairs, k=5, num_candidates=1000, device=DEVICE):
    """
    Retrieve images from text query.
    
    Args:
        query: Text string
        img_enc: Image encoder
        txt_enc: Text encoder
        pairs: List of (img, caption) pairs
        k: Number of results
        num_candidates: Number of candidates to search
        device: Device
    """
    console.print(f'[yellow]Query: "{query}"[/yellow]')
    
    with torch.no_grad():
        txt_emb = txt_enc([query])
        
        candidates = random.sample(pairs, min(num_candidates, len(pairs)))
        imgs = torch.stack([img for img, _ in candidates]).to(device)
        
        # Batch process
        batch_size = 100
        all_sims = []
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i+batch_size]
            img_embs = img_enc(batch_imgs)
            sims = (txt_emb @ img_embs.T).squeeze().cpu()
            all_sims.append(sims)
        
        all_sims = torch.cat(all_sims)
        topk = all_sims.topk(k)
    
    # Display results
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(topk.indices):
        plt.subplot(1, k, i+1)
        img = candidates[idx][0]
        if img.dim() == 3:
            img = img.squeeze(0)
        plt.imshow(img.cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"#{i+1}: {topk.values[i]:.3f}")
    plt.suptitle(f'Text→Image: "{query}"', fontsize=14)
    plt.tight_layout()
    plt.show()


def image_to_text(idx, img_enc, txt_enc, pairs, k=10, device=DEVICE):
    """
    Retrieve text captions for an image.
    
    Args:
        idx: Index in pairs
        img_enc: Image encoder
        txt_enc: Text encoder
        pairs: List of (img, caption) pairs
        k: Number of results
        device: Device
    """
    img, true_txt = pairs[idx]
    
    with torch.no_grad():
        img_emb = img_enc(img.unsqueeze(0).to(device))
        
        # Get diverse text candidates
        import random
        text_candidates = random.sample([t for _, t in pairs], min(1000, len(pairs)))
        txt_embs = txt_enc(text_candidates)
        
        sims = (img_emb @ txt_embs.T).squeeze().cpu()
        topk = sims.topk(k)
    
    # Display image
    plt.figure(figsize=(3, 3))
    if img.dim() == 3:
        img = img.squeeze(0)
    plt.imshow(img.cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Query Image")
    plt.show()
    
    # Print results
    console.print(f"[green]True caption:[/green] {true_txt}")
    console.print("[blue]Top predictions:[/blue]")
    for i in range(k):
        console.print(f"  {i+1}. {text_candidates[topk.indices[i]]} ({topk.values[i]:.3f})")


def main(epochs=5, batch_size=256, emb_dim=128, lr=1e-3):
    """Main CLIP training."""
    console.print("[bold magenta]NanoCLIP Training[/bold magenta]")
    
    # Load data
    train_loader, val_loader, vocab, train_pairs, val_pairs = build_clip_loaders(
        batch_size=batch_size
    )
    console.print(f"[blue]Vocab size: {len(vocab)}, Train: {len(train_pairs)}, Val: {len(val_pairs)}[/blue]")
    
    # Create models
    img_enc = EnhancedImageEncoder(emb_dim)
    txt_enc = EnhancedTextEncoder(vocab, emb_dim)
    criterion = ContrastiveLoss()
    
    # Train
    img_enc, txt_enc, best_loss = train_clip(
        img_enc, txt_enc, criterion, train_loader, val_loader,
        epochs=epochs, lr=lr, device=DEVICE
    )
    
    # Demo retrievals
    console.print("\n[bold green]Demo: Text→Image[/bold green]")
    queries = ["a stylish sneaker", "an elegant dress", "a clear digit 7"]
    import random
    for q in queries:
        text_to_image(q, img_enc, txt_enc, val_pairs, k=5, device=DEVICE)
    
    console.print("\n[bold green]Demo: Image→Text[/bold green]")
    for _ in range(3):
        idx = random.randint(0, len(val_pairs) - 1)
        image_to_text(idx, img_enc, txt_enc, val_pairs, k=5, device=DEVICE)
    
    console.print("[green]✅ CLIP training complete![/green]")


if __name__ == "__main__":
    main()
