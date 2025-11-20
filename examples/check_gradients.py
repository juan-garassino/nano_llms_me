import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from nano_moe.models.phase import ThetaLinear, ThetaParam

def check_gradients():
    print("🧪 Checking Gradient Flow for ThetaParam...")
    
    # Setup simple model
    dim = 64
    batch_size = 16
    
    # Standard Linear vs ThetaLinear
    std_linear = nn.Linear(dim, dim)
    theta_linear = ThetaLinear(dim, dim, A=1.0)
    
    # Inputs
    x = torch.randn(batch_size, dim, requires_grad=True)
    target = torch.randn(batch_size, dim)
    
    # Forward & Backward - Standard
    out_std = std_linear(x)
    loss_std = ((out_std - target)**2).mean()
    loss_std.backward()
    
    # Forward & Backward - Theta
    # Reset input grad
    x.grad = None
    out_theta = theta_linear(x)
    loss_theta = ((out_theta - target)**2).mean()
    loss_theta.backward()
    
    # Check Gradients
    print("\n📊 Gradient Norms:")
    
    std_grad_norm = std_linear.weight.grad.norm().item()
    print(f"  Standard Linear Weight Grad: {std_grad_norm:.6f}")
    
    # ThetaParam has 'theta' parameter
    theta_grad_norm = theta_linear.theta_weight.theta.grad.norm().item()
    print(f"  ThetaLinear Theta Grad:      {theta_grad_norm:.6f}")
    
    # Check if gradients are vanishing
    if theta_grad_norm < 1e-6:
        print("⚠️ WARNING: Theta gradients are very small (Vanishing?)")
    else:
        print("✅ Theta gradients look healthy")
        
    # Check distribution of gradients
    theta_grads = theta_linear.theta_weight.theta.grad.flatten().numpy()
    
    print(f"\n📈 Theta Gradient Stats:")
    print(f"  Mean: {np.mean(theta_grads):.6f}")
    print(f"  Std:  {np.std(theta_grads):.6f}")
    print(f"  Max:  {np.max(np.abs(theta_grads)):.6f}")

if __name__ == "__main__":
    check_gradients()
