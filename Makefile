.PHONY: demo-arc demo-shakespeare demo-evolution demo-clip demo-diffusion demo-multimodal clean

# Run the ARC training demo
demo-arc:
	@echo "🚀 Running ARC Training Demo..."
	python3 -m nano_moe.train_arc

# Run the Tiny Shakespeare training demo
demo-shakespeare:
	@echo "🎭 Running Tiny Shakespeare Training Demo..."
	python3 -m nano_moe.main

# Run the Evolutionary Strategies demo
demo-evolution:
	@echo "🧬 Running Evolutionary Strategies Demo..."
	python3 demo_evolution.py

# Run the CLIP training demo (vision-language)
demo-clip:
	@echo "🖼️ Running CLIP Training Demo..."
	python3 -m nano_moe.train_clip

# Run the Diffusion training demo (text-to-image)
demo-diffusion:
	@echo "🎨 Running Diffusion Training Demo..."
	python3 -m nano_moe.train_diffusion

# Run the Multimodal joint training demo (CLIP + Diffusion + Captioning + LM)
demo-multimodal:
	@echo "🌐 Running Multimodal Training Demo..."
	python3 -m nano_moe.train_multimodal

# Run the Phase Transformer Enhancements demo (Hybrid Wavelet, Adaptive Sparsity, FourierCLIP)
demo-phase:
	@echo "🌊 Running Phase Transformer Enhancements Demo..."
	python3 demo_phase_enhancements.py

# Benchmark SFPT vs Standard Transformer
benchmark:
	@echo "🏆 Running SFPT vs Transformer Benchmark..."
	python3 benchmark_sfpt_vs_transformer.py

# Clean up training outputs
clean:
	@echo "🧹 Cleaning up..."
	rm -rf training_outputs/
	rm -rf integrated_results/
	rm -f *.log
