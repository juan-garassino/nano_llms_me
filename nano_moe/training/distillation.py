"""
Knowledge Distillation utilities for nano_moe.

Compress large teacher models into efficient student models using:
- Logit-based distillation with temperature
- Feature-based distillation
- EMA collapse prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
from rich.console import Console

console = Console()


def train_teacher(train_loader, epochs=2, lr=1e-3, freeze_conv=True, 
                  ckpt_path="vgg16_teacher_20class.pt", device="cuda"):
    """
    Train VGG16 teacher on MNIST + FashionMNIST (20 classes).
    
    Args:
        train_loader: DataLoader with (images, labels)
        epochs: Number of training epochs
        lr: Learning rate
        freeze_conv: Whether to freeze convolutional layers
        ckpt_path: Where to save checkpoint
        device: Device to train on
    Returns:
        Path to saved checkpoint
    """
    console.print(f"[blue]Training VGG16 teacher for {epochs} epochs...[/blue]")
    
    # Load pretrained VGG16
    teacher = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = teacher.classifier[6].in_features
    teacher.classifier[6] = nn.Linear(in_features, 20)
    
    # Optionally freeze conv layers
    if freeze_conv:
        for p in teacher.features.parameters():
            p.requires_grad_(False)
    
    teacher = teacher.to(device)
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, teacher.parameters()), 
        lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in range(1, epochs + 1):
        teacher.train()
        tot_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = teacher(x)
            loss = loss_fn(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tot_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
        
        avg_loss = tot_loss / len(train_loader)
        acc = correct / max(1, total)
        console.print(f"[cyan]Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}[/cyan]")
    
    torch.save(teacher.state_dict(), ckpt_path)
    console.print(f"[green]✓ Teacher saved → {ckpt_path}[/green]")
    return ckpt_path


def distill_student(student, teacher_ckpt, train_loader, epochs=1, alpha=0.5, 
                     beta=1.0, temperature=2.0, lam=0.99, lr=1e-3, 
                     device="cuda", out_ckpt=None):
    """
    Distill knowledge from teacher to student.
    
    Args:
        student: Student model (CNN or ViT)
        teacher_ckpt: Path to teacher checkpoint
        train_loader: DataLoader
        epochs: Number of epochs
        alpha: Weight for KD loss
        beta: Weight for feature loss
        temperature: Temperature for softmax
        lam: EMA decay rate
        lr: Learning rate
        device: Device
        out_ckpt: Output checkpoint path
    Returns:
        Path to saved student checkpoint
    """
    console.print(f"[blue]Distilling student for {epochs} epochs...[/blue]")
    
    # Load teacher
    teacher = vgg16(weights=None)
    in_features = teacher.classifier[6].in_features
    teacher.classifier[6] = nn.Linear(in_features, 20)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)
    
    # Student
    student = student.to(device)
    
    # Feature projector (teacher features → student space)
    projector = nn.Linear(512 * 7 * 7, 128).to(device)
    
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()), 
        lr=lr, weight_decay=0.05
    )
    
    ema_mean_probs = None
    teacher_feat_ma, student_feat_ma = None, None
    
    for ep in range(1, epochs + 1):
        student.train()
        projector.train()
        
        pbar = tqdm(train_loader, desc=f"Distill Epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Teacher forward (detached)
            with torch.no_grad():
                t_feat = teacher.features(x)
                t_feat = teacher.avgpool(t_feat)
                t_feat = torch.flatten(t_feat, 1).detach()
                
                # Moving average centering
                batch_mean = t_feat.mean(0)
                teacher_feat_ma = batch_mean if teacher_feat_ma is None else \
                                  (lam * teacher_feat_ma + (1 - lam) * batch_mean)
                t_feat_c = (t_feat - teacher_feat_ma).detach()
                
                t_logits = teacher(x).detach()
                t_probs = F.softmax(t_logits / temperature, dim=1).detach()
            
            # Student forward
            s_logits, s_rep = student(x)
            s_batch_mean = s_rep.mean(0).detach()
            student_feat_ma = s_batch_mean if student_feat_ma is None else \
                              (lam * student_feat_ma + (1 - lam) * s_batch_mean)
            s_rep_c = s_rep - student_feat_ma
            s_probs = F.softmax(s_logits / temperature, dim=1)
            
            # EMA collapse prevention
            if ema_mean_probs is None:
                ema_mean_probs = s_probs.detach().mean(0)
            else:
                ema_mean_probs = lam * ema_mean_probs + (1 - lam) * s_probs.detach().mean(0)
            ema_loss = F.mse_loss(s_probs.mean(0), ema_mean_probs)
            
            # Losses
            ce = F.cross_entropy(s_logits, y)
            feat_loss = F.mse_loss(
                F.normalize(s_rep_c, dim=1), 
                F.normalize(projector(t_feat_c), dim=1)
            )
            kd = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=1),
                t_probs,
                reduction="batchmean"
            ) * (temperature * temperature)
            
            loss = ce + beta * feat_loss + alpha * kd + 0.1 * ema_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            pbar.set_postfix(
                ce=f"{ce.item():.3f}",
                feat=f"{feat_loss.item():.3f}",
                kd=f"{kd.item():.3f}"
            )
    
    out_ckpt = out_ckpt or "student.pt"
    torch.save(student.state_dict(), out_ckpt)
    console.print(f"[green]✓ Student saved → {out_ckpt}[/green]")
    return out_ckpt
