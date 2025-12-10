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
import os
import json
import random

from nano_moe.models.clip import EnhancedImageEncoder, EnhancedTextEncoder, ContrastiveLoss
from nano_moe.training.contrastive import build_clip_loaders, train_clip

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text_to_image(query, img_enc, txt_enc, pairs, k=5, num_candidates=1000, device=DEVICE, save_dir="results/clip"):
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
        save_dir: Directory to save results
    """
    console.print(f'[yellow]Query: "{query}"[/yellow]')
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Display and save results
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
    
    # Save the figure
    safe_query = query.replace(" ", "_").replace("→", "to").replace('"', "")
    save_path = os.path.join(save_dir, f"text_to_image_{safe_query}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]💾 Saved: {save_path}[/green]")
    
    # Save retrieval results as JSON
    results = {
        "query": query,
        "results": [
            {
                "rank": i+1,
                "similarity": float(topk.values[i]),
                "caption": candidates[topk.indices[i]][1]
            }
            for i in range(k)
        ]
    }
    
    json_path = os.path.join(save_dir, f"text_to_image_{safe_query}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def image_to_text(idx, img_enc, txt_enc, pairs, k=10, device=DEVICE, save_dir="results/clip"):
    """
    Retrieve text captions for an image.
    
    Args:
        idx: Index in pairs
        img_enc: Image encoder
        txt_enc: Text encoder
        pairs: List of (img, caption) pairs
        k: Number of results
        device: Device
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    img, true_txt = pairs[idx]
    
    with torch.no_grad():
        img_emb = img_enc(img.unsqueeze(0).to(device))
        
        # Get diverse text candidates
        text_candidates = random.sample([t for _, t in pairs], min(1000, len(pairs)))
        txt_embs = txt_enc(text_candidates)
        
        sims = (img_emb @ txt_embs.T).squeeze().cpu()
        topk = sims.topk(k)
    
    # Display and save image
    plt.figure(figsize=(3, 3))
    if img.dim() == 3:
        img = img.squeeze(0)
    plt.imshow(img.cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Query Image")
    
    save_path = os.path.join(save_dir, f"image_to_text_query_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]💾 Saved query image: {save_path}[/green]")
    
    # Print and save results
    console.print(f"[green]True caption:[/green] {true_txt}")
    console.print("[blue]Top predictions:[/blue]")
    
    results = {
        "query_idx": idx,
        "true_caption": true_txt,
        "predictions": []
    }
    
    for i in range(k):
        pred_text = text_candidates[topk.indices[i]]
        similarity = float(topk.values[i])
        console.print(f"  {i+1}. {pred_text} ({similarity:.3f})")
        results["predictions"].append({
            "rank": i+1,
            "text": pred_text,
            "similarity": similarity
        })
    
    # Save results as JSON
    json_path = os.path.join(save_dir, f"image_to_text_results_{idx}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


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
    
    # Save trained models
    os.makedirs("results/clip", exist_ok=True)
    torch.save({
        "img_enc": img_enc.state_dict(),
        "txt_enc": txt_enc.state_dict(),
        "vocab": vocab,
        "best_loss": best_loss,
        "config": {
            "emb_dim": emb_dim,
            "vocab_size": len(vocab)
        }
    }, "results/clip/clip_model.pt")
    
    console.print("[green]✓ Saved CLIP model to results/clip/clip_model.pt[/green]")
    
    # Demo retrievals
    console.print("\n[bold green]Demo: Text→Image[/bold green]")
    queries = ["a stylish sneaker", "an elegant dress", "a clear digit 7"]
    all_results = {"text_to_image": [], "image_to_text": []}
    
    for q in queries:
        result = text_to_image(q, img_enc, txt_enc, val_pairs, k=5, device=DEVICE)
        all_results["text_to_image"].append(result)
    
    console.print("\n[bold green]Demo: Image→Text[/bold green]")
    for i in range(3):
        idx = random.randint(0, len(val_pairs) - 1)
        result = image_to_text(idx, img_enc, txt_enc, val_pairs, k=5, device=DEVICE)
        all_results["image_to_text"].append(result)
    
    # Save all demo results
    with open("results/clip/demo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    console.print("[green]💾 Saved all demo results to results/clip/demo_results.json[/green]")
    
    console.print("[green]✅ CLIP training complete![/green]")


if __name__ == "__main__":
    main()
