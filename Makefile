.PHONY: demo-arc demo-shakespeare demo-evolution clean

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

# Clean up training outputs
clean:
	@echo "🧹 Cleaning up..."
	rm -rf training_outputs/
	rm -rf integrated_results/
	rm -f *.log
