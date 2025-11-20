# nano_moe 🧬

**Physics-Inspired Neuro-Symbolic AI for Abstract Reasoning**

`nano_moe` is a research framework that pushes the boundaries of neural network design by replacing standard attention mechanisms with **Sparse Fourier Phase Transformers (SFPT)** and integrating **symbolic program synthesis** for abstract reasoning tasks like ARC-AGI.

---

## 🎯 What Makes This Special?

Most AI today is built on **arithmetic** (matrix multiplication). This project is built on **physics** (wave interference, phase shifts, frequency decomposition).

### The Core Insight

> **In the Fourier domain, complex geometric transformations (rotations, reflections) become simple phase shifts.**

This means our models can learn **global symmetries** with $O(1)$ complexity, whereas standard CNNs or Vision Transformers need deep layers to "emerge" these patterns.

---

## 🏗️ Architecture

### 1. **Sparse Fourier Phase Transformer (SFPT)**
- **Frequency-domain attention** using DFT/FFT
- **Top-k sparse selection** of frequency modes
- **Phase-based reasoning** for geometric transformations
- Supports both **image** (via patch projection) and **text** (via BPE tokenization) inputs

### 2. **Hybrid Phase-Symbolic Reasoning**
- **Phase Operators**: Learnable transformations in frequency space
- **Program Synthesizer**: Composes operators into executable "programs"
- **Differentiable search**: Learns both operators and composition via backpropagation
- Designed specifically for **ARC-AGI** (Abstract Reasoning Corpus)

### 3. **Mixture of Experts (MoE) with Reflective Attention**
- **Continuous Attention Backbone**: S4-inspired state-space model for sequence modeling
- **Reflective Reasoning Block**: Iterative "thinking" mechanism with energy-based feedback
- **Sparse Expert Routing**: Top-k gating with load balancing and usage tracking

---

## 🚀 Features

### Training Methods
- ✅ **Standard Supervised Learning** (Cross-Entropy)
- ✅ **GRPO** (Group Relative Policy Optimization) - RL for text generation
- ✅ **Distillation** (teacher-student knowledge transfer)
- ✅ **Evolutionary Strategies** (OpenAI-ES, CMA-ES, Genetic Algorithms)
- ✅ **Meta-Learning** for ARC (few-shot program synthesis)

### Inference Strategies
- ✅ **System-2 Reasoning** (multi-pass refinement)
- ✅ **Active Inference** (particle-based generation with free energy minimization)
- ✅ **Adaptive Test-Time Compute** (entropy-aware retry mechanism)
- ✅ **Speculative Decoding** (draft-verifier acceleration)

### Model Compression
- ✅ **Pruning** (L1 unstructured global pruning)
- ✅ **Quantization** (int8 dynamic, fp16)

---

## 📦 Installation

```bash
# Clone the repository
git clone <repo-url>
cd nano_llms_me

# Install dependencies
pip install torch torchvision transformers datasets rich matplotlib seaborn

# (Optional) Fix NumPy version mismatch if you encounter warnings
pip install "numpy<2"
```

---

## 🎮 Quick Start

### Run Demos

```bash
# Train SFPT on Tiny Shakespeare (text generation)
make demo-shakespeare

# Train Hybrid Phase-Symbolic model on ARC-AGI (abstract reasoning)
make demo-arc

# Evolutionary Strategies demo (ES, CMA-ES, GA)
make demo-evolution
```

### Interactive Notebooks

```bash
jupyter notebook notebooks/Train_Shakespeare.ipynb  # SFPT for text
jupyter notebook notebooks/Train_ARC.ipynb          # Phase-Symbolic for ARC
```

---

## 🧪 Example: Training SFPT on Text

```python
from nano_moe.config import TrainingConfig
from nano_moe.data.text import get_text_loaders
from nano_moe.models.phase import SparseFourierPhaseTransformer
from nano_moe.training.trainer import train_epoch, eval_model

# Configuration
cfg = TrainingConfig()
cfg.dataset_type = "text"
cfg.model_type = "sfpt"

# Load data
loaders, vocab_size = get_text_loaders("tinyshakespeare", batch_size=32, seq_len=128)

# Initialize model
model = SparseFourierPhaseTransformer(
    vocab_size=vocab_size,
    dim=256,
    depth=6,
    n_heads=8,
    n_freqs=64,
    top_k=32
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
train_epoch(model, loaders, optimizer, criterion, device, ["tinyshakespeare"], tracker, epoch=1)
```

---

## 🧠 Example: Evolving ARC Operator Libraries

```python
from nano_moe.training.evolution import genetic_algorithm

# Define genetic operators
def init_population():
    return [random_operator_sequence() for _ in range(8)]

def fitness(library):
    # Evaluate on ARC tasks
    return consistency_score(library, arc_tasks)

# Evolve
best_library = genetic_algorithm(
    init_population_fn=init_population,
    fitness_fn=fitness,
    mutate_fn=mutate_operator,
    crossover_fn=crossover_libraries,
    population_size=50,
    generations=100
)
```

---

## 📊 Project Structure

```
nano_moe/
├── config.py              # TrainingConfig dataclass
├── main.py                # Entry point for standard training
├── train_arc.py           # Specialized training for ARC-AGI
├── inference.py           # Advanced inference strategies
├── models/
│   ├── attention.py       # ContinuousAttention (S4-inspired)
│   ├── reflection.py      # ReflectiveAttentionBlock
│   ├── moe.py             # Mixture of Experts with routing
│   ├── phase.py           # Sparse Fourier Phase Transformer
│   └── phase_symbolic.py  # Hybrid Phase-Symbolic for ARC
├── data/
│   ├── loaders.py         # Image dataset loaders (MNIST, CIFAR)
│   ├── text.py            # BPE text loaders (Shakespeare, WikiText)
│   └── arc.py             # ARC-AGI dataset loader
├── training/
│   ├── trainer.py         # Training and evaluation loops
│   ├── tracker.py         # Experiment tracking and plotting
│   ├── monitor.py         # TrainingMonitor with rich logging
│   ├── optimizer.py       # PhaseSignSGD (specialized for phase params)
│   ├── rl.py              # GRPO and Reward Model
│   ├── pruning.py         # Pruning and quantization utilities
│   └── evolution.py       # Evolutionary Strategies (ES, CMA-ES, GA)
└── utils/                 # Utility functions

notebooks/
├── Train_ARC.ipynb        # Interactive ARC training
└── Train_Shakespeare.ipynb # Interactive text generation

demo_evolution.py          # Evolutionary Strategies demo
Makefile                   # Quick demo commands
```

---

## 🎯 Use Cases

### 1. **ARC-AGI Challenge**
The `HybridPhaseSymbolicARC` model is specifically designed to solve abstract reasoning puzzles by:
- Learning a library of geometric operators (rotations, flips, color inversions)
- Composing them into "programs" that transform input grids to output grids
- Using meta-learning to generalize from 2-3 demonstrations

### 2. **Text Generation with Physics Priors**
The SFPT model can capture long-range dependencies in text using frequency-domain attention, potentially outperforming standard Transformers on:
- Music generation (periodic patterns)
- Code generation (structured syntax)
- Mathematical notation (symbolic reasoning)

### 3. **Hyperparameter Optimization**
Use CMA-ES to optimize learning rates, dropout, attention heads, etc. without relying on expensive grid search.

### 4. **Neural Architecture Search**
Use Genetic Algorithms to evolve the MoE expert configurations, operator libraries, or even model depth/width.

---

## 🔬 Research Foundations

This project synthesizes ideas from:
- **Fourier Neural Operators** (Li et al., 2020) - frequency-domain learning
- **S4 (Structured State Spaces)** (Gu et al., 2022) - continuous-time sequence models
- **Perceiver** (Jaegle et al., 2021) - cross-attention for arbitrary inputs
- **Program Synthesis** (Lake et al., 2015) - learning symbolic transformations
- **Active Inference** (Friston, 2010) - free energy minimization
- **Evolution Strategies** (Salimans et al., 2017) - gradient-free optimization

---

## 🚧 Known Limitations

1. **NumPy Version Conflict**: PyTorch was compiled with NumPy 1.x, but NumPy 2.x is installed. Install `numpy<2` to resolve.
2. **ARC Operator Library**: Currently randomly initialized. Needs "seeding" with geometric priors or synthetic pre-training.
3. **Phase Optimization**: The current `PhaseSignSGD` is simplified. True sign-based optimization may improve convergence.
4. **SFPT for Sharp Edges**: Fourier methods struggle with discontinuities. May need wavelet hybrids for crisp image generation.

---

## 🛠️ Future Work

- [ ] Implement true PhaseSignSGD with phase-specific learning rates
- [ ] Add seeded operator library with geometric priors
- [ ] Integrate Mamba-style SSMs for ultra-long sequences
- [ ] Add KV-cache and Flash Attention for inference speedup
- [ ] Create Hugging Face model hub integration
- [ ] Benchmark on ARC-AGI leaderboard

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{nano_moe2025,
  title={nano_moe: Physics-Inspired Neuro-Symbolic AI},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nano_moe}
}
```

---

## 🤝 Contributing

This is a research prototype. Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT architecture and ES research
- **Google DeepMind** for ARC-AGI dataset
- **Hugging Face** for transformers library
- **PyTorch** team for the deep learning framework

---

**Built with ❤️ for the future of AI reasoning**
