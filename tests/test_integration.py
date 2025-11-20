import sys
import os
import torch

# Ensure we can import nano_moe
sys.path.append(os.getcwd())

from nano_moe.config import TrainingConfig
from nano_moe.datasets.loaders import get_multimnist_loaders
from nano_moe.main import main

def test_universal_phase():
    print("\n=== Testing Universal SFPT Integration ===")
    cfg = TrainingConfig()
    cfg.model_type = "universal_phase"
    cfg.epochs = 1
    cfg.dataset_names = ["tinyshakespeare"]
    cfg.batch_size = 2
    cfg.seq_len = 32
    
    try:
        main(cfg)
        print("✅ Universal SFPT Integration Test Passed")
    except Exception as e:
        print(f"❌ Universal SFPT Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()

def test_true_phase():
    print("\n=== Testing True Holographic SFPT Integration ===")
    cfg = TrainingConfig()
    cfg.model_type = "true_phase"
    cfg.epochs = 1
    cfg.dataset_names = ["tinyshakespeare"]
    cfg.batch_size = 2
    cfg.seq_len = 32
    
    try:
        main(cfg)
        print("✅ True Holographic SFPT Integration Test Passed")
    except Exception as e:
        print(f"❌ True Holographic SFPT Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_universal_phase()
    test_true_phase()
    test_quantum_diffusion()

def test_quantum_diffusion():
    print("\n=== Testing Quantum Diffusion ARC Integration ===")
    cfg = TrainingConfig()
    cfg.model_type = "quantum_diffusion"
    # Quantum Diffusion uses its own internal loop, so epochs/batch_size in cfg might be ignored or used differently
    # But we pass them anyway.
    
    try:
        main(cfg)
        print("✅ Quantum Diffusion ARC Integration Test Passed")
    except Exception as e:
        print(f"❌ Quantum Diffusion ARC Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
