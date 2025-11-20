# nano_moe 🌐✨

**A Complete AI Research Lab: From Abstract Reasoning to Multimodal Generation**

`nano_moe` is a comprehensive research framework that combines physics-inspired neural architectures, symbolic reasoning, evolutionary optimization, and state-of-the-art multimodal AI into a single unified package.

---

## 🎯 Vision: The Full AI Lab

This isn't just another deep learning framework — it's a **complete AI research laboratory** covering:

| Domain | Capabilities | Models |
|--------|--------------|--------|
| 🧠 **Abstract Reasoning** | Geometric transformations, program synthesis | Phase-Symbolic ARC |
| 📝 **Language** | Text generation, sequence modeling | SFPT, Decoder-only LM |
| 🖼️ **Vision-Language** | Image-text alignment, retrieval | CLIP |
| 🎨 **Generation** | Text-to-image, image-to-image | DDPM, Latent Diffusion |
| 💬 **Multimodal** | Joint vision-language-generation | NanoMultimodal |
| 🧬 **Optimization** | Gradient-free search, meta-learning | ES, CMA-ES, GA |

---

## 🌟 What Makes This Special?

### 1. **Physics-Inspired Architecture**
Most AI is built on **arithmetic**. We build on **physics**:
- **Fourier domain reasoning**: Complex transformations become simple phase shifts
- **Frequency-based attention**: Global patterns learned in $O(1)$ complexity
- **Wave interference**: Natural mechanism for superposition and composition

### 2. **Unified Multimodal System**
Train a **single model** that can:
- Generate images from text (diffusion)
- Generate text from images (captioning)
- Align vision and language (CLIP)
- Generate text continuations (LM)
- Transform images with text guidance (I2I)

### 3. **Creative Generation Features**
- **🌀 Dual Oscillation**: Generate images that blend two concepts using 180° rotation during denoising
- **DDIM Sampling**: Fast deterministic generation
- **Classifier-Free Guidance**: Improved conditional generation

### 4. **Gradient-Free Optimization**
- **Evolution Strategies**: Optimize neural networks without backprop
- **CMA-ES**: Hyperparameter tuning
- **Genetic Algorithms**: Evolve discrete structures (operator libraries)

---

## 🏗️ Core Architectures

### **Sparse Fourier Phase Transformer (SFPT)**
```python
# Frequency-domain attention for images and text
- DFT/FFT-based reasoning
- Top-k sparse mode selection
- Phase-based geometric transformations
- Supports: Images (patches), Text (BPE tokens)
```

### **Hybrid Phase-Symbolic Reasoning**
```python
# Program synthesis for abstract reasoning (ARC-AGI)
- Learnable phase operators
- Differentiable program composition
- Meta-learning from few demonstrations
- Geometric transformation library
```

### **NanoMultimodal (4-in-1 Model)**
```python
# Joint training with 4 objectives:
1. CLIP contrastive (image ↔ text)
2. Image captioning (image → text)
3. Text-to-image diffusion (text → image)
4. Language modeling (text → text)
```

### **Mixture of Experts + Reflective Attention**
```python
# Sparse expert routing with iterative reasoning
- S4-inspired continuous attention
- Energy-based thinking mechanism
- Load-balanced top-k gating
```

---

## 🚀 Features

### Training Methods
- ✅ **Supervised Learning** (Cross-Entropy)
- ✅ **GRPO** (Group Relative Policy Optimization)
- ✅ **Knowledge Distillation** (Teacher-Student)
- ✅ **Contrastive Learning** (CLIP-style)
- ✅ **Diffusion Training** (DDPM, LDM, DDIM)
- ✅ **Evolutionary Strategies** (OpenAI-ES, CMA-ES, GA)
- ✅ **Meta-Learning** (Few-shot program synthesis)

### Inference Strategies
- ✅ **System-2 Reasoning** (Multi-pass refinement)
- ✅ **Active Inference** (Free energy minimization)
- ✅ **Adaptive Test-Time Compute** (Entropy-aware retry)
- ✅ **Speculative Decoding** (Draft-verifier acceleration)
- ✅ **DDIM Sampling** (Fast deterministic generation)
- ✅ **Classifier-Free Guidance** (Conditional scaling)

### Model Compression
- ✅ **Pruning** (L1 unstructured global)
- ✅ **Quantization** (int8, fp16)

---

## 📦 Installation

```bash
# Clone the repository
git clone <repo-url>
cd nano_llms_me

# Install dependencies
pip install torch torchvision transformers datasets rich matplotlib seaborn einops

# Fix NumPy version mismatch (if needed)
pip install "numpy<2"
```

---

## 🎮 Quick Start

### Run Demos

```bash
# Abstract Reasoning (ARC-AGI)
make demo-arc

# Text Generation (Tiny Shakespeare)
make demo-shakespeare

# Vision-Language (CLIP)
make demo-clip

# Text-to-Image (Diffusion + Oscillation)
make demo-diffusion

# Full Multimodal (All capabilities)
make demo-multimodal

# Evolutionary Strategies
make demo-evolution
```

### Interactive Notebooks

```bash
# Text generation with SFPT
jupyter notebook notebooks/Train_Shakespeare.ipynb

# Abstract reasoning with Phase-Symbolic
jupyter notebook notebooks/Train_ARC.ipynb

# Multimodal training & inference (with oscillation!)
jupyter notebook notebooks/Train_Multimodal.ipynb
```

---

## 🌐 Example: Multimodal Training

```python
from nano_moe.train_multimodal import train_joint, text_to_image, image_to_text

# Train joint model (CLIP + Captioning + Diffusion + LM)
model, train_ds, val_ds = train_joint(
    epochs=2,
    batch_size=128,
    use_latent=True,  # Use VAE + Latent Diffusion
    vae=pretrained_vae
)

# Generate images from text
imgs = text_to_image(model, "a stylish sneaker", train_ds.stoi, n=8, guide_w=2.0)

# Generate captions from images
captions = image_to_text(model, imgs, train_ds.itos)

# Transform images with text guidance
result = image_to_image(model, imgs[0], train_ds.stoi, text="an elegant dress", strength=0.6)
```

---

## 🌀 Special Feature: Dual Oscillation

Generate images that **blend two concepts** using creative 180° rotation during denoising:

```python
# Generate image oscillating between two classes
result = model.diff.dual_oscillation(
    size=(4, 8, 8),        # Latent size
    class_a=5,             # Sandal
    class_b=7,             # Sneaker
    guide_weight=2.0,
    flip_every=50,         # Rotate every 50 steps
    n_classes=10
)

# Decode latent to image
img = model.vae.decode(result)
# Result: Creative blend of sandal + sneaker!
```

This creates **artistic hybrid visualizations** that standard diffusion can't produce.

---

## 🎨 Example: Text-to-Image with CLIP Bridge

```python
from nano_moe.train_diffusion import train_diffusion, text_to_image

# Train diffusion model
diffusion, vae = train_diffusion(
    use_latent=True,      # Latent diffusion (faster)
    epochs=3,
    guide_weight=2.5
)

# Load CLIP text encoder for conditioning
from nano_moe.models.clip import EnhancedTextEncoder
txt_enc = EnhancedTextEncoder(vocab, emb_dim=128)

# Generate images from text
query = "a stylish sneaker"
imgs = text_to_image(query, txt_enc, diffusion, vae, n_samples=6, guide_weight=2.5)
```

---

## 🧠 Example: Evolving ARC Solutions

```python
from nano_moe.training.evolution import genetic_algorithm

# Evolve operator library for ARC
def fitness(library):
    # Test on ARC validation set
    return evaluate_arc_library(library, validation_tasks)

best_library = genetic_algorithm(
    init_population_fn=init_random_operators,
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
├── config.py                    # TrainingConfig dataclass
├── main.py                      # Entry point for standard training
├── train_arc.py                 # ARC-AGI meta-learning
├── train_clip.py                # Vision-language training
├── train_diffusion.py           # Text-to-image generation
├── train_multimodal.py          # Joint multimodal training
├── inference.py                 # Advanced inference strategies
│
├── models/
│   ├── attention.py             # ContinuousAttention (S4-inspired)
│   ├── reflection.py            # ReflectiveAttentionBlock
│   ├── moe.py                   # Mixture of Experts
│   ├── phase.py                 # Sparse Fourier Phase Transformer
│   ├── phase_symbolic.py        # Hybrid Phase-Symbolic (ARC)
│   ├── clip.py                  # CLIP encoders + contrastive loss
│   ├── vae.py                   # Variational Autoencoder
│   ├── diffusion.py             # DDPM, LDM, dual oscillation
│   └── multimodal.py            # Unified multimodal model
│
├── data/
│   ├── loaders.py               # Image datasets (MNIST, CIFAR)
│   ├── text.py                  # BPE text loaders
│   └── arc.py                   # ARC-AGI dataset
│
├── training/
│   ├── trainer.py               # Training loops
│   ├── tracker.py               # Experiment tracking
│   ├── monitor.py               # Rich logging
│   ├── optimizer.py             # PhaseSignSGD
│   ├── rl.py                    # GRPO, Reward Model
│   ├── pruning.py               # Compression utilities
│   ├── evolution.py             # ES, CMA-ES, GA
│   ├── distillation.py          # Knowledge distillation
│   └── contrastive.py           # CLIP training utilities
│
└── utils/                       # Helper functions

notebooks/
├── Train_ARC.ipynb              # Interactive ARC demo
├── Train_Shakespeare.ipynb      # Interactive text generation
└── Train_Multimodal.ipynb       # Interactive multimodal demo

demo_evolution.py                # Evolutionary strategies demo
Makefile                         # Quick command shortcuts
```

---

## 🎯 Capabilities Matrix

| Task | Input | Output | Model | Demo |
|------|-------|--------|-------|------|
| **Text Generation** | Text prompt | Continuation | SFPT | `make demo-shakespeare` |
| **Abstract Reasoning** | Input/output grids | Transform program | Phase-Symbolic | `make demo-arc` |
| **Vision-Language Retrieval** | Image or text | Matching text/image | CLIP | `make demo-clip` |
| **Text-to-Image** | Text description | Generated image | DDPM/LDM | `make demo-diffusion` |
| **Image-to-Text** | Image | Caption | Caption Decoder | `make demo-multimodal` |
| **Image-to-Image** | Image + text | Transformed image | Guided Diffusion | `make demo-multimodal` |
| **Text-to-Text** | Text prompt | Continuation | Decoder LM | `make demo-multimodal` |
| **Dual Oscillation** | Two class IDs | Blended image | Rotation + Diffusion | `make demo-diffusion` |
| **Hyperparameter Search** | Objective function | Optimal params | CMA-ES | `make demo-evolution` |
| **Architecture Search** | Task + constraints | Evolved structure | Genetic Algorithm | `make demo-evolution` |

---

## 🔬 Research Foundations

This project synthesizes cutting-edge research:

**Core Architectures:**
- [Fourier Neural Operators](https://arxiv.org/abs/2010.08895) (Li et al., 2020)
- [S4: Structured State Spaces](https://arxiv.org/abs/2111.00396) (Gu et al., 2022)
- [CLIP](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [DDPM](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Latent Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)

**Training Methods:**
- [Evolution Strategies](https://arxiv.org/abs/1703.03864) (Salimans et al., 2017)
- [GRPO](https://arxiv.org/abs/2402.03300) (Shao et al., 2024)
- [Active Inference](https://www.nature.com/articles/nrn2787) (Friston, 2010)

**Tasks:**
- [ARC-AGI](https://github.com/fchollet/ARC-AGI) (Chollet, 2019)

---

## 🚧 Known Limitations

1. **NumPy Version**: PyTorch compiled with NumPy 1.x, install `numpy<2` to resolve warnings
2. **ARC Operator Library**: Needs geometric priors or synthetic pre-training for better initialization
3. **SFPT Sharp Edges**: Fourier methods struggle with discontinuities; may need wavelet hybrids
4. **q_sample/sample_i2i**: Methods exist but need full integration testing for image-to-image workflows

---

## 🛠️ Future Work

- [ ] Integrate ControlNet for fine-grained image control
- [ ] Add LoRA/QLoRA for efficient fine-tuning
- [ ] Implement Flash Attention for faster training
- [ ] Add KV-cache for inference speedup
- [ ] Create Hugging Face model hub integration
- [ ] Benchmark on ARC-AGI leaderboard
- [ ] Add Stable Diffusion XL architecture
- [ ] Integrate Mamba-2 for ultra-long sequences

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{nano_moe2025,
  title={nano_moe: A Complete AI Research Lab},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nano_moe},
  note={Physics-inspired neural architectures, multimodal generation, and evolutionary optimization}
}
```

---

## 🤝 Contributing

This is an active research project. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution:**
- 🎨 New generation models (ControlNet, AnimateDiff)
- 🧠 New reasoning architectures (Mamba, RWKV)
- 🔧 Performance optimizations (Flash Attention, quantization)
- 📊 Benchmarks and evaluations
- 📚 Documentation and tutorials

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT, CLIP, and Evolution Strategies research
- **Google DeepMind** for ARC-AGI dataset
- **CompVis/Stability AI** for Latent Diffusion Models
- **Hugging Face** for transformers ecosystem
- **PyTorch** team for the deep learning framework

---

## 🌟 Highlights

> **"The first framework to combine Fourier-domain reasoning, symbolic program synthesis, and multimodal generation in a single unified architecture."**

**Key Innovations:**
- 🌀 **Dual Oscillation**: Creative image blending via 180° rotation
- 🧬 **Gradient-Free Optimization**: Evolve neural networks without backprop
- 🎨 **4-in-1 Multimodal**: Single model for CLIP + Caption + Diffusion + LM
- 🧠 **Phase-Symbolic Reasoning**: Frequency domain + program synthesis for ARC

---

**Built with ❤️ for the future of AI**

*From abstract reasoning to creative generation — all in one package.*
