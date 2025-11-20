import torch
import torch.nn as nn
import math

class S4Kernel(nn.Module):
    """State-Space Model kernel for O(L) sequence modeling"""
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.01)))
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """O(L) sequence processing via SSM"""
        B, L, D = u.shape
        dt = torch.exp(self.log_dt).clamp(1e-4, 0.1)
        
        # Discretize: A_bar = I + dt*A, B_bar = dt*B
        A_bar = torch.eye(self.d_state, device=u.device) + dt * self.A
        B_bar = dt * self.B
        
        # Sequential scan (simplified for compatibility)
        x = torch.zeros(B, self.d_state, device=u.device)
        outputs = []
        
        for t in range(L):
            x = A_bar @ x.unsqueeze(-1) + (B_bar @ u[:, t].unsqueeze(-1))
            x = x.squeeze(-1)
            y = (self.C @ x.unsqueeze(-1)).squeeze(-1) + self.D * u[:, t]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

class NeuralOperatorKernel(nn.Module):
    """Neural operator acting on function spaces"""
    def __init__(self, dim: int, num_modes: int = 16):
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        
        # Fourier layer for global interactions
        self.fourier_weight = nn.Parameter(
            torch.randn(dim, dim, num_modes, 2) * 0.02
        )
        
        # Local MLP
        self.local_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply continuous kernel operator"""
        B, L, D = x.shape
        
        # Global: Fourier transform O(L log L)
        # Handle short sequences gracefully
        actual_modes = min(self.num_modes, L // 2 + 1)
        
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')
        out_ft = torch.zeros_like(x_ft)
        
        for i in range(min(actual_modes, x_ft.size(1))):
            weight = torch.view_as_complex(self.fourier_weight[:, :, i])
            out_ft[:, i] = torch.einsum('bd,de->be', x_ft[:, i], weight)
        
        x_global = torch.fft.irfft(out_ft, n=L, dim=1, norm='ortho')
        
        # Local
        x_local = self.local_mlp(x)
        
        return x_global + x_local

class ContinuousAttention(nn.Module):
    """Attention 2.0: Continuous operator + SSM"""
    def __init__(self, dim: int, d_state: int = 64, num_modes: int = 16):
        super().__init__()
        self.to_latent = nn.Linear(dim, dim)
        self.operator = NeuralOperatorKernel(dim, num_modes=num_modes)
        self.ssm = S4Kernel(dim, d_state=d_state)
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """O(L) continuous attention"""
        f = self.to_latent(x)
        f = self.norm(f)
        
        # Neural operator
        kf = self.operator(f)
        
        # SSM memory
        hf = self.ssm(kf)
        
        # Context gate
        g = self.gate(x)
        out = g * hf
        
        return self.to_out(out)
