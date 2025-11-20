#!/usr/bin/env python3
# reflective_transformer_full_pipeline.py
# Integrates Reflective Attention (iterative thinking loops) into the comprehensive
# training framework with monitoring, visualization, and full pipeline support.
#
# Quick examples:
#   python reflective_transformer_full_pipeline.py --mode train --use_reflective --max_iters 3
#   python reflective_transformer_full_pipeline.py --mode infer --load_path artifacts/reflective_gpt.pt
#   python reflective_transformer_full_pipeline.py  # No args = full pipeline + ZIP

import os, json, argparse, random, time, urllib.request, zipfile, shutil, math
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("=" * 50)
    class Panel:
        def __init__(self, text, title="", style=""): self.text=text; self.title=title
        def __str__(self): return f"[{self.title}] {self.text}"

console = Console()

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

try:
    import seaborn as sns
    sns.set_palette("husl")
except Exception:
    pass

plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# =============================================================================
# TRAINING MONITOR (Enhanced for Reflective Attention)
# =============================================================================

class TrainingMonitor:
    """Comprehensive training monitor with reflective attention tracking."""
    def __init__(self, experiment_name="reflective_experiment"):
        self.experiment_name = experiment_name
        self.base_dir = Path("training_outputs") / experiment_name
        self._setup_folders()
        self.metrics = defaultdict(list)
        self.attention_snapshots = []
        self.embedding_snapshots = []
        self.moe_snapshots = []
        self.reflective_snapshots = []  # NEW: Track thinking iterations
        self.step_count = 0
        self.epoch_count = 0

    def _setup_folders(self):
        folders = [
            self.base_dir,
            self.base_dir / "models",
            self.base_dir / "plots" / "training_curves",
            self.base_dir / "plots" / "attention_analysis",
            self.base_dir / "plots" / "embedding_evolution",
            self.base_dir / "plots" / "moe_analysis",
            self.base_dir / "plots" / "reflective_thinking",  # NEW
            self.base_dir / "plots" / "animations",
            self.base_dir / "tokenizers",
            self.base_dir / "logs",
            self.base_dir / "data",
        ]
        for f in folders: f.mkdir(parents=True, exist_ok=True)
        console.print(f"📁 Output structure at: {self.base_dir}")

    def log_training_step(self, epoch, step, train_loss, val_loss=None, lr=None, grad_norm=None,
                         avg_iterations=None, early_stop_ratio=None):
        self.metrics['epoch'].append(epoch)
        self.metrics['step'].append(self.step_count)
        self.metrics['train_loss'].append(float(train_loss))
        if val_loss is not None: self.metrics['val_loss'].append(float(val_loss))
        if lr is not None: self.metrics['learning_rate'].append(float(lr))
        if grad_norm is not None: self.metrics['grad_norm'].append(float(grad_norm))
        if avg_iterations is not None: self.metrics['avg_iterations'].append(float(avg_iterations))
        if early_stop_ratio is not None: self.metrics['early_stop_ratio'].append(float(early_stop_ratio))
        self.step_count += 1

    def capture_reflective_stats(self, stats):
        """Capture statistics from reflective attention blocks."""
        if stats and 'avg_iterations_per_block' in stats:
            self.reflective_snapshots.append({
                'step': self.step_count,
                'epoch': self.epoch_count,
                'avg_iterations': float(stats['avg_iterations_per_block']),
                'early_stop_ratio': float(stats.get('early_stop_ratio', 0.0)),
                'thinking_depth': float(stats.get('thinking_depth', 0.0))
            })

    def capture_model_state(self, model):
        if hasattr(model, 'blocks'):
            attn_data = []
            for i, block in enumerate(model.blocks):
                if hasattr(block, 'attn') and hasattr(block.attn, 'c_attn'):
                    with torch.no_grad():
                        w = block.attn.c_attn.weight.data.detach().cpu().numpy()
                    n_embd = w.shape[1]
                    q_w = w[:n_embd, :]
                    k_w = w[n_embd:2*n_embd, :]
                    corr = np.corrcoef(q_w.flatten(), k_w.flatten())[0, 1]
                    attn_data.append({'layer': i, 'qk_correlation': float(corr)})
                elif hasattr(block, 'reflective_attn'):
                    # Reflective attention block
                    attn_data.append({'layer': i, 'type': 'reflective'})
            if attn_data:
                self.attention_snapshots.append({
                    'step': self.step_count, 'epoch': self.epoch_count, 'data': attn_data
                })
        if hasattr(model, 'wte'):
            with torch.no_grad():
                emb = model.wte.weight.data.detach().cpu().numpy()
            self.embedding_snapshots.append({
                'step': self.step_count, 'epoch': self.epoch_count,
                'norm': float(np.linalg.norm(emb)),
                'mean': float(np.mean(emb)), 'std': float(np.std(emb)),
                'weights': emb.copy(),
            })

    def capture_moe_stats(self, stats):
        if stats and 'mean_routing_probs' in stats and stats['mean_routing_probs'] is not None:
            probs = stats['mean_routing_probs']
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            probs = np.asarray(probs)
            entropy = float(-(probs * np.log(probs + 1e-8)).sum())
            balance = float(1.0 - (np.std(probs) / (np.mean(probs) + 1e-8)))
            self.moe_snapshots.append({
                'step': self.step_count, 'epoch': self.epoch_count,
                'routing_probs': probs.tolist(), 'entropy': entropy, 'balance': balance,
            })

    def _plot_training_curves(self):
        if not self.metrics['train_loss']:
            return
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        epochs = np.array(self.metrics['epoch'])
        train_losses = np.array(self.metrics['train_loss'])

        # Loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, train_losses, linewidth=2, alpha=0.9, label='Training Loss')
        if self.metrics.get('val_loss'):
            vl = np.array(self.metrics['val_loss'])
            ax1.plot(epochs[-len(vl):], vl, linestyle='--', linewidth=2, alpha=0.9, label='Validation Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Progress')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Learning rate
        if self.metrics.get('learning_rate'):
            ax2 = fig.add_subplot(gs[0, 2])
            lrs = np.array(self.metrics['learning_rate'])
            ax2.plot(epochs[-len(lrs):], lrs, linewidth=2)
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('LR'); ax2.set_title('LR Schedule')
            ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

        # Gradient norms
        if self.metrics.get('grad_norm'):
            ax3 = fig.add_subplot(gs[1, 0])
            gns = np.array(self.metrics['grad_norm'])
            ax3.plot(epochs[-len(gns):], gns, linewidth=2, alpha=0.9)
            ax3.set_xlabel('Epoch'); ax3.set_ylabel('Grad Norm')
            ax3.set_title('Gradient Norms'); ax3.grid(True, alpha=0.3)

        # Smoothed loss
        ax4 = fig.add_subplot(gs[1, 1])
        if len(train_losses) > 4:
            w = max(3, min(20, len(train_losses)//10))
            sm = np.convolve(train_losses, np.ones(w)/w, mode='valid')
            ax4.plot(epochs[w-1:], sm, linewidth=3, alpha=0.9)
            ax4.set_xlabel('Epoch'); ax4.set_ylabel('Smoothed Loss')
            ax4.set_title(f'Loss (Moving Avg, window={w})'); ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5,0.5,'Insufficient steps',ha='center',va='center'); ax4.axis('off')

        # Stats box
        ax5 = fig.add_subplot(gs[1, 2])
        final_loss = float(train_losses[-1]) if len(train_losses) else 0.0
        min_loss = float(np.min(train_losses)) if len(train_losses) else 0.0
        improvement = float(train_losses[0] - final_loss) if len(train_losses) > 1 else 0.0
        stats_text = f'Final Loss: {final_loss:.4f}\nBest Loss: {min_loss:.4f}\n'
        stats_text += f'Improvement: {improvement:.4f}\nTotal Steps: {len(train_losses)}'
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax5.axis('off'); ax5.set_title('Training Stats')

        # Reflective iterations (NEW)
        if self.metrics.get('avg_iterations'):
            ax6 = fig.add_subplot(gs[2, 0])
            iters = np.array(self.metrics['avg_iterations'])
            ax6.plot(epochs[-len(iters):], iters, linewidth=2, alpha=0.9, color='purple')
            ax6.set_xlabel('Epoch'); ax6.set_ylabel('Avg Iterations')
            ax6.set_title('Thinking Depth'); ax6.grid(True, alpha=0.3)

        # Early stopping ratio (NEW)
        if self.metrics.get('early_stop_ratio'):
            ax7 = fig.add_subplot(gs[2, 1])
            esr = np.array(self.metrics['early_stop_ratio'])
            ax7.plot(epochs[-len(esr):], esr, linewidth=2, alpha=0.9, color='green')
            ax7.set_xlabel('Epoch'); ax7.set_ylabel('Early Stop Ratio')
            ax7.set_title('Adaptive Termination'); ax7.grid(True, alpha=0.3)

        out = self.base_dir / "plots" / "training_curves" / "comprehensive_training.png"
        plt.suptitle(f'{self.experiment_name} - Training Analysis', fontsize=16)
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        console.print(f"📊 Training curves saved to {out}")

    def _plot_reflective_thinking(self):
        """NEW: Visualize reflective thinking patterns."""
        if not self.reflective_snapshots:
            return
        
        steps = [s['step'] for s in self.reflective_snapshots]
        avg_iters = [s['avg_iterations'] for s in self.reflective_snapshots]
        early_stops = [s['early_stop_ratio'] for s in self.reflective_snapshots]
        thinking_depths = [s['thinking_depth'] for s in self.reflective_snapshots]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average iterations over time
        ax1 = axes[0, 0]
        ax1.plot(steps, avg_iters, linewidth=2, alpha=0.9, color='purple')
        ax1.fill_between(steps, avg_iters, alpha=0.3, color='purple')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Avg Iterations')
        ax1.set_title('Average Thinking Iterations per Block'); ax1.grid(True, alpha=0.3)

        # Early stopping ratio
        ax2 = axes[0, 1]
        ax2.plot(steps, early_stops, linewidth=2, alpha=0.9, color='green')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Early Stop Ratio')
        ax2.set_title('Adaptive Termination Rate'); ax2.legend(); ax2.grid(True, alpha=0.3)

        # Thinking depth (total iterations across all blocks)
        ax3 = axes[1, 0]
        ax3.plot(steps, thinking_depths, linewidth=2, alpha=0.9, color='orange')
        ax3.set_xlabel('Step'); ax3.set_ylabel('Total Thinking Depth')
        ax3.set_title('Cumulative Iterations per Forward Pass'); ax3.grid(True, alpha=0.3)

        # Distribution of iterations
        ax4 = axes[1, 1]
        ax4.hist(avg_iters, bins=30, alpha=0.85, color='purple', edgecolor='black')
        ax4.axvline(np.mean(avg_iters), color='r', linestyle='--', linewidth=2, label='Mean')
        ax4.set_xlabel('Avg Iterations'); ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Thinking Iterations'); ax4.legend(); ax4.grid(True, alpha=0.3)

        out = self.base_dir / "plots" / "reflective_thinking" / "thinking_analysis.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        console.print(f"🧠 Reflective thinking analysis saved to {out}")

    def _create_loss_animation(self):
        if not self.metrics['train_loss']:
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        train_losses = np.array(self.metrics['train_loss'])
        epochs = np.array(self.metrics['epoch'])
        def animate(i):
            ax.clear()
            ax.plot(epochs[:i+1], train_losses[:i+1], linewidth=3, alpha=0.9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(f'Training Progress - Step {i+1}/{len(train_losses)}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(epochs) if len(epochs) else 1)
            if len(train_losses):
                ax.set_ylim(min(train_losses)*0.9, max(train_losses)*1.1)
        frames = min(len(train_losses), 100)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=150, repeat=True)
        out = self.base_dir / "plots" / "animations" / "loss_evolution.gif"
        anim.save(out, writer='pillow', fps=5); plt.close()
        console.print(f"🎬 Loss animation saved to {out}")

    def _plot_attention(self):
        if not self.attention_snapshots:
            return
        steps = [s['step'] for s in self.attention_snapshots]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.ravel()
        
        # Filter only standard attention layers
        standard_layers = []
        for s in self.attention_snapshots:
            for d in s['data']:
                if 'qk_correlation' in d:
                    standard_layers.append(d['layer'])
                    break
            if standard_layers:
                break
        
        if standard_layers:
            n_layers = max(standard_layers) + 1
            for layer in range(n_layers):
                corrs = []
                for s in self.attention_snapshots:
                    for d in s['data']:
                        if d.get('layer') == layer and 'qk_correlation' in d:
                            corrs.append(d['qk_correlation'])
                            break
                if corrs:
                    ax1.plot(steps[:len(corrs)], corrs, linewidth=2, alpha=0.9, label=f'L{layer}')
            ax1.set_xlabel('Step'); ax1.set_ylabel('Q-K Corr')
            ax1.set_title('Query-Key Alignment Evolution')
            ax1.legend(); ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No standard attention layers', ha='center', va='center')
            ax1.axis('off')
        
        ax2.text(0.5, 0.5, 'Reflective Attention\n(see reflective_thinking plots)', 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        out = self.base_dir / "plots" / "attention_analysis" / "attention_comprehensive.png"
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        console.print(f"🔍 Attention analysis saved to {out}")

    def _plot_embeddings(self):
        if not self.embedding_snapshots:
            return
        steps = [s['step'] for s in self.embedding_snapshots]
        norms = [s['norm'] for s in self.embedding_snapshots]
        means = [s['mean'] for s in self.embedding_snapshots]
        stds  = [s['std']  for s in self.embedding_snapshots]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.ravel()
        
        ax1.plot(steps, norms, linewidth=2)
        ax1.set_title('Embedding Frobenius Norm'); ax1.set_xlabel('Step')
        ax1.set_ylabel('Norm'); ax1.grid(True, alpha=0.3)
        
        ax2.plot(steps, means, linewidth=2, label='Mean')
        ax2.plot(steps, stds, linewidth=2, label='Std')
        ax2.legend(); ax2.set_title('Embedding Stats'); ax2.grid(True, alpha=0.3)

        w = self.embedding_snapshots[-1].get('weights', None)
        if w is not None:
            ax3.hist(w.flatten(), bins=60, alpha=0.85)
            ax3.set_title('Latest Embedding Distribution'); ax3.grid(True, alpha=0.3)
            token_norms = np.linalg.norm(w, axis=1)
            ax4.plot(token_norms, alpha=0.9)
            ax4.set_title('Per-Token L2 Norms'); ax4.set_xlabel('Token')
            ax4.set_ylabel('L2'); ax4.grid(True, alpha=0.3)
        else:
            ax3.axis('off'); ax4.axis('off')

        out = self.base_dir / "plots" / "embedding_evolution" / "embedding_analysis.png"
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        console.print(f"📐 Embedding analysis saved to {out}")

    def _plot_moe(self):
        if not self.moe_snapshots:
            return
        steps = [s['step'] for s in self.moe_snapshots]
        nE = len(self.moe_snapshots[0]['routing_probs'])
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.ravel()
        
        for e in range(nE):
            usage = [s['routing_probs'][e] for s in self.moe_snapshots]
            ax1.plot(steps, usage, linewidth=2, alpha=0.9, label=f'E{e}')
        ax1.set_title('Expert Usage Evolution'); ax1.set_xlabel('Step')
        ax1.set_ylabel('Prob'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ent = [s['entropy'] for s in self.moe_snapshots]
        ax2.plot(steps, ent, linewidth=2, alpha=0.9)
        ax2.set_title('Routing Entropy'); ax2.set_xlabel('Step'); ax2.grid(True, alpha=0.3)

        latest = self.moe_snapshots[-1]['routing_probs']
        ax3.pie(latest, labels=[f'E{i}' for i in range(len(latest))], 
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('Current Expert Distribution')

        bal = [s['balance'] for s in self.moe_snapshots]
        ax4.plot(steps, bal, linewidth=2, alpha=0.9)
        ax4.set_title('Load Balance (1=best)'); ax4.set_xlabel('Step')
        ax4.grid(True, alpha=0.3)

        out = self.base_dir / "plots" / "moe_analysis" / "moe_comprehensive.png"
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        console.print(f"🔀 MoE analysis saved to {out}")

    def _save_logs(self):
        with open(self.base_dir / "logs" / "training_metrics.json", 'w') as f:
            json.dump({k: list(v) for k, v in self.metrics.items()}, f, indent=2)
        if self.attention_snapshots:
            with open(self.base_dir / "logs" / "attention_evolution.json", 'w') as f:
                json.dump(self.attention_snapshots, f, indent=2)
        if self.embedding_snapshots:
            light = [{k: v for k, v in s.items() if k != 'weights'} 
                    for s in self.embedding_snapshots]
            with open(self.base_dir / "logs" / "embedding_evolution.json", 'w') as f:
                json.dump(light, f, indent=2)
        if self.moe_snapshots:
            with open(self.base_dir / "logs" / "moe_evolution.json", 'w') as f:
                json.dump(self.moe_snapshots, f, indent=2)
        if self.reflective_snapshots:
            with open(self.base_dir / "logs" / "reflective_thinking.json", 'w') as f:
                json.dump(self.reflective_snapshots, f, indent=2)
        console.print(f"💾 Logs saved to {self.base_dir / 'logs'}")

    def generate_all(self):
        console.print("🎨 Generating visualizations and logs...")
        self._plot_training_curves()
        self._create_loss_animation()
        self._plot_reflective_thinking()  # NEW
        self._plot_attention()
        self._plot_embeddings()
        self._plot_moe()
        self._save_logs()
        console.print("✅ Visualizations complete")

    def snapshot_tokenizer(self, src_prefix: str):
        m = Path(src_prefix + ".model")
        if m.exists():
            dst = self.base_dir / "tokenizers" / m.name
            try:
                shutil.copy2(m, dst)
            except Exception:
                pass

    def zip_everything(self):
        summary = {
            "experiment": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "total_steps": int(self.step_count),
            "total_epochs": int(self.epoch_count),
            "final_loss": self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            "avg_thinking_depth": float(np.mean(self.metrics.get('avg_iterations', [0]))),
        }
        with open(self.base_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        readme = f"""# {self.experiment_name} - Results

This experiment uses Reflective Attention for iterative cognitive processing.

## Contents
- `models/`: Model checkpoints
- `plots/`: All visualizations including reflective thinking analysis
- `logs/`: JSON logs of all metrics
- `tokenizers/`: Tokenizer files
- `data/`: Data artifacts

## Reflective Attention Features
- Adaptive iteration termination
- Energy-based feedback
- Low-rank relational transformations
"""
        with open(self.base_dir / "README.md", 'w') as f:
            f.write(readme)

        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_name = f"reflective_results_{ts}.zip"
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in self.base_dir.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to('.'))
        size_mb = os.path.getsize(zip_name) / (1024*1024)
        console.print(Panel(f"Created ZIP: {zip_name} ({size_mb:.1f} MB)", title="Package"))
        return zip_name

# =============================================================================
# TOKENIZER (Simple BPE)
# =============================================================================

def _get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for p in zip(ids, ids[1:]): counts[p] = counts.get(p, 0) + 1
    return counts

def _merge(ids, pair, idx):
    newids = []; i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]:
            newids.append(idx); i += 2
        else:
            newids.append(ids[i]); i += 1
    return newids

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = self._build_vocab()
    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for (a,b),i in self.merges.items():
            vocab[i] = vocab[a] + vocab[b]
        return vocab
    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    def save(self, prefix):
        d = os.path.dirname(prefix)
        if d: os.makedirs(d, exist_ok=True)
        with open(prefix + ".model", "w") as f:
            for (a,b),i in self.merges.items():
                f.write(f"{a} {b}\n")
    def load(self, path):
        merges = {}; idx = 256
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        a = int(parts[0]); b = int(parts[1])
                    except ValueError:
                        continue
                    merges[(a, b)] = idx; idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()

class BasicTokenizer(Tokenizer):
    def train(self, text, vocab_size):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges = {}; vocab = {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = _get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get); idx = 256 + i
            ids = _merge(ids, pair, idx)
            merges[pair] = idx; vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges; self.vocab = vocab
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            if not stats: break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

def train_or_load_tokenizer(kind, vocab_size, text, prefix):
    model_path = prefix + ".model"
    if os.path.exists(model_path):
        tok = BasicTokenizer(); tok.load(model_path)
        vs = max(tok.vocab.keys()) + 1
        console.print(Panel(f"Loaded tokenizer: {model_path} (size={vs})", title="Tokenizer"))
        return tok, model_path, vs
    tok = BasicTokenizer(); tok.train(text, vocab_size); tok.save(prefix)
    vs = max(tok.vocab.keys()) + 1
    console.print(Panel(f"Saved tokenizer: {prefix}.model (size={vs})", title="Tokenizer"))
    return tok, model_path, vs

# =============================================================================
# DATA
# =============================================================================

def load_tiny_shakespeare(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare_input.txt")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, path)
        except:
            # Fallback content
            with open(path, "w", encoding="utf-8") as f:
                f.write("To be, or not to be, that is the question.\n" * 100)
    text = open(path, "r", encoding="utf-8").read()
    split = int(0.9 * len(text))
    return text[:split], text[split:]

def get_batch_tokens(ids, block_size, batch_size, device):
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(ids[i:i+block_size]) for i in ix]).long()
    y = torch.stack([torch.tensor(ids[i+1:i+1+block_size]) for i in ix]).long()
    return x.to(device), y.to(device)

# =============================================================================
# MoE COMPONENTS
# =============================================================================

class TopKRouter(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_experts, k=1, temp=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), 
                                nn.Linear(hidden_dim, num_experts))
        self.k = k; self.temp = temp
    def forward(self, x):
        probs = F.softmax(self.net(x) / self.temp, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals))
        sp = probs * mask
        sp = sp / (sp.sum(dim=-1, keepdim=True) + 1e-9)
        return sp, probs

def entropy_mean(p, eps=1e-9):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1).mean()

class MoEFFN(nn.Module):
    def __init__(self, in_dim, num_experts=4, k=1, router_hidden=128, dropout=0.1,
                 entropy_penalty=0.0, routing_mode="uniform"):
        super().__init__()
        hidden = 4 * in_dim
        self.expert_fc = nn.ModuleList([nn.Linear(in_dim, hidden) for _ in range(num_experts)])
        self.expert_proj = nn.ModuleList([nn.Linear(hidden, in_dim) for _ in range(num_experts)])
        self.router = TopKRouter(in_dim, router_hidden, num_experts, k)
        self.drop = nn.Dropout(dropout)
        self.entropy_penalty = entropy_penalty
        self.routing_mode = routing_mode
        self.num_experts = num_experts
    def forward(self, x):
        B, T, C = x.shape; xf = x.reshape(B*T, C)
        sp, dp = self.router(xf); out = 0.0
        for e in range(self.num_experts):
            h = F.gelu(self.expert_fc[e](xf))
            h = self.expert_proj[e](h)
            out = out + sp[:, e].unsqueeze(-1) * h
        y = self.drop(out.view(B, T, C))
        aux = torch.tensor(0.0, device=x.device)
        if self.routing_mode == "specialize":
            aux = -self.entropy_penalty * entropy_mean(dp)
        return y, aux, {"mean_routing_probs": dp.mean(0)}

# =============================================================================
# REFLECTIVE ATTENTION (from Document 1)
# =============================================================================

@dataclass
class ReflectiveAttentionCfg:
    dim: int
    num_heads: int = 8
    max_iters: int = 5
    num_relation_types: int = 8
    rank: int = 32
    dropout: float = 0.1
    adaptive_threshold: float = 0.5
    reflection_dim_multiplier: int = 4
    use_energy_feedback: bool = True

class ReflectiveAttentionBlock(nn.Module):
    """Reflective Thinking Block with iterative processing."""
    
    def __init__(self, cfg: ReflectiveAttentionCfg, block_size: int):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.max_iters = cfg.max_iters
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.rank = cfg.rank
        self.use_energy_feedback = cfg.use_energy_feedback
        
        # Core projections
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim)
        
        # Low-rank operator components
        self.u_proj = nn.Linear(self.head_dim, self.head_dim * self.rank)
        self.w_proj = nn.Linear(self.head_dim, self.rank * self.head_dim)
        self.b_proj = nn.Linear(self.head_dim, self.head_dim)
        
        # Learned relation embeddings
        self.relation_embeds = nn.Embedding(cfg.num_relation_types, self.head_dim)
        
        # Adaptive iteration gate
        self.iter_gate = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 4),
            nn.GELU(),
            nn.Linear(cfg.dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Reflection network
        self.reflection = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * cfg.reflection_dim_multiplier),
            nn.GELU(),
            nn.Linear(cfg.dim * cfg.reflection_dim_multiplier, cfg.dim)
        )
        
        # Energy feedback
        if self.use_energy_feedback:
            self.energy_feedback = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim // 2),
                nn.GELU(),
                nn.Linear(cfg.dim // 2, cfg.dim)
            )
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                           .view(1, 1, block_size, block_size))
    
    def _compute_low_rank_operators(self, q, k, v):
        b, h, n, d = q.shape
        q_flat = q.transpose(1, 2).contiguous().view(b * n, d)
        k_flat = k.transpose(1, 2).contiguous().view(b * n, d)
        
        u_flat = self.u_proj(q_flat)
        w_flat = self.w_proj(k_flat)
        
        U = u_flat.view(b, n, h, d, self.rank).transpose(1, 2)
        W = w_flat.view(b, n, h, self.rank, d).transpose(1, 2)
        
        v_expanded = v.unsqueeze(-1)
        inner = torch.matmul(W, v_expanded).squeeze(-1)
        
        return U, inner
    
    def _relation_conditioning(self, q, k):
        b, h, n, d = q.shape
        qk_mean = (q + k) / 2
        qk_mean_pool = qk_mean.mean(dim=2)
        
        rel_logits = torch.einsum('bhd,rd->bhr', qk_mean_pool, 
                                 self.relation_embeds.weight)
        rel_probs = F.softmax(rel_logits, dim=-1)
        
        rel_embed = torch.einsum('bhr,rd->bhd', rel_probs, 
                                self.relation_embeds.weight)
        return rel_embed.unsqueeze(2)
    
    def forward(self, x, energies=None):
        b, n, d = x.shape
        h = x.clone()
        
        iteration_stats = {
            'iterations_used': torch.zeros(b, device=x.device),
            'gate_values': [],
            'energy_feedback_applied': []
        }
        
        # Energy feedback integration
        if energies is not None and self.use_energy_feedback:
            energy_signal = self.energy_feedback(energies)
            h = h + self.dropout(energy_signal)
        
        for t in range(self.max_iters):
            # Project Q, K, V
            q = self.q_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention with causal masking
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_scores = attn_scores.masked_fill(self.bias[:,:,:n,:n] == 0, float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Standard attention
            usual_out = torch.matmul(attn_probs, v)
            
            # Low-rank relational component
            U, inner = self._compute_low_rank_operators(q, k, v)
            weighted_inner = torch.matmul(attn_probs, inner)
            relational_part = torch.matmul(U, weighted_inner.unsqueeze(-1)).squeeze(-1)
            
            # Bias and relation conditioning
            b_term = self.b_proj(q)
            rel_embed = self._relation_conditioning(q, k)
            
            # Combine components
            out = usual_out + relational_part + b_term + rel_embed
            out = out.transpose(1, 2).contiguous().view(b, n, d)
            out = self.o_proj(out)
            
            # Residual
            h = h + self.dropout(out)
            
            # Reflection step
            h_reflected = self.reflection(self.norm(h))
            h = h + self.dropout(h_reflected)
            
            # Adaptive gating
            gate_val = self.iter_gate(h.mean(dim=1)).squeeze(-1)
            iteration_stats['gate_values'].append(gate_val.mean().item())
            
            # Track iterations
            iteration_stats['iterations_used'] += (t + 1) * (gate_val >= 0.5).float()
            
            # Early stopping
            if t >= 1 and (gate_val < 0.5).all():
                iteration_stats['iterations_used'][gate_val < 0.5] = t + 1
                break
        
        avg_iterations = iteration_stats['iterations_used'].mean().item()
        iteration_stats.update({
            'avg_iterations': avg_iterations,
            'early_stop_ratio': (iteration_stats['iterations_used'] < self.max_iters).float().mean().item()
        })
        
        return h, iteration_stats

class ReflectiveTransformerBlock(nn.Module):
    """Complete transformer block with Reflective Attention."""
    
    def __init__(self, cfg: ReflectiveAttentionCfg, block_size: int, 
                 use_moe: bool = False, moe_config=None):
        super().__init__()
        self.reflective_attn = ReflectiveAttentionBlock(cfg, block_size)
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        
        if use_moe and moe_config:
            self.ffn = MoEFFN(cfg.dim, **moe_config)
            self.use_moe = True
        else:
            hidden_dim = 4 * cfg.dim
            self.ffn = nn.Sequential(
                nn.Linear(cfg.dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, cfg.dim),
                nn.Dropout(cfg.dropout)
            )
            self.use_moe = False
    
    def forward(self, x, energies=None):
        h, stats = self.reflective_attn(self.ln1(x), energies)
        x = x + h
        
        if self.use_moe:
            ff_out, aux_loss, moe_stats = self.ffn(self.ln2(x))
            x = x + ff_out
            stats.update(moe_stats)
            stats['aux_loss'] = aux_loss
        else:
            x = x + self.ffn(self.ln2(x))
        
        return x, stats

# =============================================================================
# STANDARD COMPONENTS
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, n, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head; self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3*n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout = dropout
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, use_moe=False, 
                 moe_config=None):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, True)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd, True)
        
        if use_moe and moe_config:
            self.ffn = MoEFFN(n_embd, **moe_config)
            self.use_moe = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.GELU(), 
                nn.Linear(4 * n_embd, n_embd)
            )
            self.use_moe = False
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            y, aux, stats = self.ffn(self.ln2(x))
            return x + y, aux, stats
        else:
            return x + self.ffn(self.ln2(x)), torch.tensor(0.0, device=x.device), {}

# =============================================================================
# REFLECTIVE GPT MODEL
# =============================================================================

@dataclass
class GPTCfg:
    block_size: int = 128
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    # Reflective attention
    use_reflective: bool = False
    max_iters: int = 5
    num_relation_types: int = 8
    rank: int = 32
    adaptive_threshold: float = 0.5
    reflection_dim_multiplier: int = 4
    use_energy_feedback: bool = True
    # MoE
    use_moe: bool = False
    routing_mode: str = "uniform"
    num_experts: int = 4
    topk: int = 1
    router_hidden: int = 128
    moe_dropout: float = 0.1
    entropy_penalty: float = 0.0

class ReflectiveNanoGPT(nn.Module):
    """NanoGPT with optional Reflective Attention blocks."""
    
    def __init__(self, cfg: GPTCfg):
        super().__init__()
        self.cfg = cfg
        
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        
        blocks = []
        for i in range(cfg.n_layer):
            # Use reflective attention in later layers
            use_reflective_here = cfg.use_reflective and i >= (cfg.n_layer // 2)
            
            if use_reflective_here:
                ref_cfg = ReflectiveAttentionCfg(
                    dim=cfg.n_embd,
                    num_heads=cfg.n_head,
                    max_iters=cfg.max_iters,
                    num_relation_types=cfg.num_relation_types,
                    rank=cfg.rank,
                    dropout=cfg.dropout,
                    adaptive_threshold=cfg.adaptive_threshold,
                    reflection_dim_multiplier=cfg.reflection_dim_multiplier,
                    use_energy_feedback=cfg.use_energy_feedback
                )
                moe_cfg = {
                    'num_experts': cfg.num_experts,
                    'k': cfg.topk,
                    'router_hidden': cfg.router_hidden,
                    'dropout': cfg.moe_dropout,
                    'entropy_penalty': cfg.entropy_penalty,
                    'routing_mode': cfg.routing_mode
                } if cfg.use_moe else None
                block = ReflectiveTransformerBlock(ref_cfg, cfg.block_size, 
                                                  cfg.use_moe, moe_cfg)
            else:
                moe_cfg = {
                    'num_experts': cfg.num_experts,
                    'k': cfg.topk,
                    'router_hidden': cfg.router_hidden,
                    'dropout': cfg.moe_dropout,
                    'entropy_penalty': cfg.entropy_penalty,
                    'routing_mode': cfg.routing_mode
                } if cfg.use_moe else None
                block = GPTBlock(cfg.n_embd, cfg.n_head, cfg.block_size, 
                               cfg.dropout, cfg.use_moe, moe_cfg)
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = LayerNorm(cfg.n_embd, True)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        
        aux_total = torch.tensor(0.0, device=idx.device)
        thinking_stats = {
            'avg_iterations': [],
            'early_stop_ratio': [],
            'aux_losses': []
        }
        
        energies = None
        
        for i, block in enumerate(self.blocks):
            if isinstance(block, ReflectiveTransformerBlock):
                x, stats = block(x, energies)
                thinking_stats['avg_iterations'].append(stats['avg_iterations'])
                thinking_stats['early_stop_ratio'].append(stats['early_stop_ratio'])
                if 'aux_loss' in stats:
                    aux_total = aux_total + stats['aux_loss']
                    thinking_stats['aux_losses'].append(stats['aux_loss'])
            else:
                x, aux, stats = block(x)
                aux_total = aux_total + aux
                if aux.item() > 0:
                    thinking_stats['aux_losses'].append(aux)
            
            # Compute energies after first block
            if i == 0:
                energies = -torch.norm(x, dim=-1, keepdim=True).expand(-1, -1, self.cfg.n_embd)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Aggregate stats
        avg_iterations = (sum(thinking_stats['avg_iterations']) / 
                         max(1, len(thinking_stats['avg_iterations'])))
        avg_early_stop = (sum(thinking_stats['early_stop_ratio']) / 
                         max(1, len(thinking_stats['early_stop_ratio'])))
        
        stats = {
            'avg_iterations_per_block': avg_iterations,
            'early_stop_ratio': avg_early_stop,
            'total_aux_loss': aux_total,
            'thinking_depth': avg_iterations * len(self.blocks)
        }
        
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + aux_total
            
            # Energy regularization
            if self.cfg.use_energy_feedback:
                energy_reg = torch.mean(logits ** 2) * 0.01
                loss = loss + energy_reg
        
        return logits, loss, stats
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================================================================
# TRAINING WITH MONITORING
# =============================================================================

def train_lm(model, train_ids, val_ids, block_size, epochs, steps, batch, lr, 
            device, title, monitor: TrainingMonitor=None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_val = float('inf')

    for ep in range(1, epochs+1):
        model.train()
        losses = []; gnorms = []; t0 = time.time()
        all_stats = []
        
        for st in range(steps):
            xb, yb = get_batch_tokens(train_ids, block_size, batch, device)
            opt.zero_grad()
            _, loss, stats = model(xb, yb)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gnorms.append(float(total_norm.item()))
            opt.step()
            losses.append(float(loss.item()))
            all_stats.append(stats)
            
            if monitor and (st % 10 == 0):
                monitor.capture_model_state(model)
                if stats:
                    monitor.capture_reflective_stats(stats)
                    if 'mean_routing_probs' in stats:
                        monitor.capture_moe_stats(stats)
        
        # Validation
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch_tokens(val_ids, block_size, batch, device)
            _, vloss, _ = model(xb, yb)
            v = float(vloss.item())
        
        tr = float(sum(losses)/len(losses))
        lr_now = float(sched.get_last_lr()[0])
        gn = float(sum(gnorms)/len(gnorms)) if gnorms else 0.0
        
        # Average reflective stats
        avg_iters = np.mean([s.get('avg_iterations_per_block', 0) for s in all_stats])
        avg_early = np.mean([s.get('early_stop_ratio', 0) for s in all_stats])
        
        if monitor:
            monitor.log_training_step(ep, steps, tr, v, lr_now, gn, avg_iters, avg_early)
            monitor.epoch_count = ep
        
        console.print(Panel(
            f"Epoch {ep}/{epochs} train={tr:.3f} val={v:.3f} grad={gn:.3f} "
            f"iters={avg_iters:.2f} early_stop={avg_early:.2%} time={time.time()-t0:.1f}s",
            title=title
        ))
        
        if v < best_val:
            best_val = v
            save_path = (monitor.base_dir / "models" / f"best_{title.replace(' ', '_').lower()}.pt" 
                        if monitor else Path("artifacts") / f"best_{title.replace(' ', '_').lower()}.pt")
            payload = {
                "kind": "reflective_gpt",
                "cfg": asdict(model.cfg),
                "state_dict": model.state_dict()
            }
            torch.save(payload, save_path)
        
        sched.step()
    
    return model

# =============================================================================
# MAIN CLI
# =============================================================================

def main(argv=None):
    if argv is None:
        argv = []

    DEFAULT_EPOCHS = int(os.environ.get("EPOCHS", 50))
    DEFAULT_STEPS = int(os.environ.get("STEPS", 50))
    DEFAULT_BATCH = int(os.environ.get("BATCH", 64))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "infer"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--steps_per_epoch", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    # Reflective
    ap.add_argument("--use_reflective", action="store_true")
    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--use_energy_feedback", action="store_true", default=True)
    # MoE
    ap.add_argument("--use_moe", action="store_true")
    ap.add_argument("--routing_mode", default="uniform", choices=["uniform", "specialize"])
    ap.add_argument("--num_experts", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--router_hidden", type=int, default=128)
    ap.add_argument("--entropy_penalty", type=float, default=0.0)
    # Files
    ap.add_argument("--tok_prefix", default="artifacts/tokenizers/reflective_bpe")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--save_path", default="artifacts/reflective_gpt.pt")
    ap.add_argument("--load_path", default="artifacts/reflective_gpt.pt")
    ap.add_argument("--sample_start", default="\n")
    ap.add_argument("--sample_tokens", type=int, default=100)
    ap.add_argument("--exp_name", default="reflective_experiment")
    args = ap.parse_args(argv)

    os.makedirs("artifacts", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(Panel(f"Using device: {device}", title="Device"))

    # Monitor
    monitor = TrainingMonitor(args.exp_name)

    # Data
    train_txt, val_txt = load_tiny_shakespeare("./data")
    tok, _, vs = train_or_load_tokenizer("basic", args.vocab_size, 
                                        train_txt + val_txt, args.tok_prefix)
    train_ids, val_ids = tok.encode(train_txt), tok.encode(val_txt)
    monitor.snapshot_tokenizer(args.tok_prefix)

    if args.mode == "train":
        cfg = GPTCfg(
            block_size=args.block_size,
            vocab_size=vs,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.1,
            use_reflective=args.use_reflective,
            max_iters=args.max_iters,
            rank=args.rank,
            use_energy_feedback=args.use_energy_feedback,
            use_moe=args.use_moe,
            routing_mode=args.routing_mode,
            num_experts=args.num_experts,
            topk=args.topk,
            router_hidden=args.router_hidden,
            entropy_penalty=args.entropy_penalty
        )
        
        model = ReflectiveNanoGPT(cfg)
        title = f"Reflective-GPT ({'reflective' if args.use_reflective else 'standard'})"
        
        train_lm(model, train_ids, val_ids, args.block_size, args.epochs,
                args.steps_per_epoch, args.batch_size, args.lr, device, title, monitor)
        
        # Save final model
        torch.save({"kind": "reflective_gpt", "cfg": asdict(cfg), 
                   "state_dict": model.state_dict()}, args.save_path)
        torch.save({"kind": "reflective_gpt", "cfg": asdict(cfg),
                   "state_dict": model.state_dict()}, 
                  monitor.base_dir / "models" / Path(args.save_path).name)
        
        console.print(Panel(f"Model saved to {args.save_path}", title="Save"))
        
        # Generate visualizations
        monitor.generate_all()
        zip_path = monitor.zip_everything()
        console.print(Panel(f"Results packaged: {zip_path}", title="Complete"))

    elif args.mode == "infer":
        payload = torch.load(args.load_path, map_location=device)
        cfg = GPTCfg(**payload["cfg"])
        model = ReflectiveNanoGPT(cfg)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        
        start_ids = tok.encode(args.sample_start)
        idx = torch.tensor([start_ids], device=device)
        
        console.print(Panel("Generating with reflective thinking...", title="Generation"))
        sample = model.generate(idx, args.sample_tokens, temperature=0.8, top_k=200)[0].tolist()
        text = tok.decode(sample)
        
        console.print(Panel(text, title="Generated Text", border_style="green"))

# =============================================================================
# FULL PIPELINE (Educational Demo)
# =============================================================================

def run_full_pipeline():
    console.rule("[bold magenta]Reflective Transformer - Full Pipeline")
    
    DEFAULT_EPOCHS = int(os.environ.get("EPOCHS", 30))
    DEFAULT_STEPS = int(os.environ.get("STEPS", 30))
    DEFAULT_BATCH = int(os.environ.get("BATCH", 64))
    
    # 1) Train standard GPT (baseline)
    console.print(Panel("Stage 1: Training Standard GPT (Baseline)", 
                       style="bold yellow"))
    main([
        "--mode", "train",
        "--epochs", str(DEFAULT_EPOCHS),
        "--steps_per_epoch", str(DEFAULT_STEPS),
        "--batch_size", str(DEFAULT_BATCH),
        "--n_layer", "4",
        "--n_embd", "256",
        "--save_path", "artifacts/standard_gpt.pt",
        "--exp_name", "standard_gpt_baseline"
    ])
    
    # 2) Train Reflective GPT (no MoE)
    console.print(Panel("Stage 2: Training Reflective GPT", 
                       style="bold cyan"))
    main([
        "--mode", "train",
        "--use_reflective",
        "--max_iters", "3",
        "--epochs", str(DEFAULT_EPOCHS),
        "--steps_per_epoch", str(DEFAULT_STEPS),
        "--batch_size", str(DEFAULT_BATCH),
        "--n_layer", "4",
        "--n_embd", "256",
        "--save_path", "artifacts/reflective_gpt.pt",
        "--exp_name", "reflective_gpt"
    ])
    
    # 3) Train Reflective GPT + MoE
    console.print(Panel("Stage 3: Training Reflective GPT with MoE", 
                       style="bold green"))
    main([
        "--mode", "train",
        "--use_reflective",
        "--use_moe",
        "--routing_mode", "specialize",
        "--max_iters", "3",
        "--num_experts", "4",
        "--epochs", str(DEFAULT_EPOCHS),
        "--steps_per_epoch", str(DEFAULT_STEPS),
        "--batch_size", str(DEFAULT_BATCH),
        "--n_layer", "4",
        "--n_embd", "256",
        "--entropy_penalty", "0.001",
        "--save_path", "artifacts/reflective_moe_gpt.pt",
        "--exp_name", "reflective_moe_gpt"
    ])
    
    # 4) Inference comparison
    console.print(Panel("Stage 4: Generating Samples from All Models", 
                       style="bold magenta"))
    
    for model_name, path in [
        ("Standard GPT", "artifacts/standard_gpt.pt"),
        ("Reflective GPT", "artifacts/reflective_gpt.pt"),
        ("Reflective MoE GPT", "artifacts/reflective_moe_gpt.pt")
    ]:
        if os.path.exists(path):
            console.print(f"\n[bold]Sampling from {model_name}:[/bold]")
            main([
                "--mode", "infer",
                "--load_path", path,
                "--sample_start", "To be, or not to be,",
                "--sample_tokens", "80",
                "--exp_name", model_name.replace(" ", "_").lower()
            ])
    
    console.rule("[bold green]Pipeline Complete!")
    console.print(Panel(
        "✅ All stages completed!\n\n"
        "Models trained:\n"
        "  • Standard GPT (baseline)\n"
        "  • Reflective GPT (iterative thinking)\n"
        "  • Reflective MoE GPT (thinking + expert routing)\n\n"
        "Check training_outputs/ for comprehensive visualizations:\n"
        "  • Training curves with thinking depth metrics\n"
        "  • Reflective thinking analysis (iterations, early stopping)\n"
        "  • MoE routing patterns\n"
        "  • Attention and embedding evolution\n"
        "  • Animated training progress\n\n"
        "All results packaged in timestamped ZIP files.",
        title="Summary",
        border_style="green"
    ))

# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_models():
    """Compare standard vs reflective models."""
    console.print(Panel("Model Comparison Tool", style="bold blue"))
    
    models_to_compare = [
        ("Standard GPT", "artifacts/standard_gpt.pt"),
        ("Reflective GPT", "artifacts/reflective_gpt.pt"),
        ("Reflective MoE", "artifacts/reflective_moe_gpt.pt")
    ]
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tok = BasicTokenizer()
    tok_path = "artifacts/tokenizers/reflective_bpe.model"
    if os.path.exists(tok_path):
        tok.load(tok_path)
    
    for name, path in models_to_compare:
        if not os.path.exists(path):
            continue
        
        payload = torch.load(path, map_location=device)
        cfg = GPTCfg(**payload["cfg"])
        model = ReflectiveNanoGPT(cfg)
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Measure inference time
        test_input = torch.randint(0, cfg.vocab_size, (1, 64), device=device)
        
        with torch.no_grad():
            t0 = time.time()
            for _ in range(10):
                logits, _, stats = model(test_input)
            avg_time = (time.time() - t0) / 10
        
        avg_iters = stats.get('avg_iterations_per_block', 0)
        thinking_depth = stats.get('thinking_depth', 0)
        
        results.append({
            'name': name,
            'params': total_params,
            'time_ms': avg_time * 1000,
            'avg_iters': avg_iters,
            'thinking_depth': thinking_depth,
            'reflective': cfg.use_reflective,
            'moe': cfg.use_moe
        })
    
    # Display comparison table
    if results:
        console.print("\n[bold]Model Comparison:[/bold]\n")
        table_data = []
        for r in results:
            table_data.append([
                r['name'],
                f"{r['params']:,}",
                f"{r['time_ms']:.2f}",
                f"{r['avg_iters']:.2f}" if r['reflective'] else "N/A",
                f"{r['thinking_depth']:.2f}" if r['reflective'] else "N/A",
                "✓" if r['reflective'] else "✗",
                "✓" if r['moe'] else "✗"
            ])
        
        headers = ["Model", "Parameters", "Time (ms)", "Avg Iterations", 
                  "Thinking Depth", "Reflective", "MoE"]
        
        # Simple text table
        console.print("  ".join(f"{h:^20}" for h in headers))
        console.print("-" * (20 * len(headers) + 2 * (len(headers) - 1)))
        for row in table_data:
            console.print("  ".join(f"{cell:^20}" for cell in row))
        console.print()

# =============================================================================
# ENTRYPOINT
# =============================================================================

def _sanitize_argv(argv):
    """Remove Jupyter/Colab notebook flags."""
    out = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("-f", "--f"):
            skip = True
            continue
        if a.startswith("-f=") or a.startswith("--f="):
            continue
        out.append(a)
    return out

if __name__ == "__main__":
    import sys
    argv = _sanitize_argv(sys.argv[1:])
    
    if not argv or ("--mode" not in argv):
        # No arguments = run full educational pipeline
        run_full_pipeline()
        compare_models()
    else:
        # Run specific mode
        main(argv)