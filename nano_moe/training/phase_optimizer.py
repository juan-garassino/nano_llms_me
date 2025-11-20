"""
Advanced phase-aware optimization for complex-valued neural networks.

Handles the unique challenges of optimizing phase parameters:
- 2π periodicity (phase wrapping)
- Complex manifold geometry
- Coupling between magnitude and phase
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math


class TruePhaseOptimizer(Optimizer):
    """
    Geodesic optimizer for complex-valued parameters.
    
    Accounts for:
    - Phase periodicity (geodesic descent on S^1)
    - Adaptive per-frequency learning rates
    - Decoupled magnitude/phase updates
    """
    
    def __init__(self, params, lr=1e-3, phase_lr_mult=0.5, momentum=0.9,
                 phase_momentum=0.95, weight_decay=0, dampening=0):
        """
        Args:
            params: Model parameters
            lr: Base learning rate
            phase_lr_mult: Phase learning rate multiplier (typically < 1)
            momentum: Momentum for magnitude updates
            phase_momentum: Momentum for phase updates (higher for smoother phase)
            weight_decay: L2 regularization
            dampening: Dampening for momentum
        """
        defaults = dict(
            lr=lr,
            phase_lr_mult=phase_lr_mult,
            momentum=momentum,
            phase_momentum=phase_momentum,
            weight_decay=weight_decay,
            dampening=dampening
        )
        super().__init__(params, defaults)
    
    def _unwrap_phase(self, phase):
        """Unwrap phase to avoid discontinuities at ±π."""
        return torch.atan2(torch.sin(phase), torch.cos(phase))
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            phase_lr = lr * group['phase_lr_mult']
            momentum = group['momentum']
            phase_momentum = group['phase_momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Check if parameter is complex
                if torch.is_complex(p):
                    # Separate magnitude and phase
                    mag = torch.abs(p)
                    phase = torch.angle(p)
                    
                    # Gradients w.r.t. magnitude and phase
                    grad_mag = torch.real(grad * torch.conj(p / (mag + 1e-8)))
                    grad_phase = torch.imag(grad * torch.conj(p / (mag + 1e-8)))
                    
                    # Weight decay on magnitude only
                    if weight_decay != 0:
                        grad_mag = grad_mag.add(mag, alpha=weight_decay)
                    
                    # Momentum for magnitude
                    param_state = self.state[p]
                    if 'momentum_buffer_mag' not in param_state:
                        buf_mag = param_state['momentum_buffer_mag'] = torch.clone(grad_mag).detach()
                    else:
                        buf_mag = param_state['momentum_buffer_mag']
                        buf_mag.mul_(momentum).add_(grad_mag, alpha=1 - dampening)
                    
                    # Momentum for phase (higher to avoid rapid phase changes)
                    if 'momentum_buffer_phase' not in param_state:
                        buf_phase = param_state['momentum_buffer_phase'] = torch.clone(grad_phase).detach()
                    else:
                        buf_phase = param_state['momentum_buffer_phase']
                        buf_phase.mul_(phase_momentum).add_(grad_phase, alpha=1 - dampening)
                    
                    # Update magnitude
                    new_mag = mag - lr * buf_mag
                    new_mag = torch.clamp(new_mag, min=1e-8)  # Prevent collapse
                    
                    # Update phase with geodesic descent on S^1
                    new_phase = phase - phase_lr * buf_phase
                    new_phase = self._unwrap_phase(new_phase)  # Keep in [-π, π]
                    
                    # Reconstruct complex parameter
                    p.copy_(new_mag * torch.exp(1j * new_phase))
                
                else:
                    # Standard real-valued update
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                    
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    p.add_(buf, alpha=-lr)
        
        return loss


class AdaptivePhaseOptimizer(Optimizer):
    """
    Phase optimizer with per-frequency adaptive learning rates.
    
    Low frequencies (global patterns) get smaller learning rates.
    High frequencies (details) get larger learning rates.
    """
    
    def __init__(self, params, lr=1e-3, base_phase_lr=1e-4, 
                 freq_decay=0.9, momentum=0.9):
        """
        Args:
            params: Model parameters
            lr: Base magnitude learning rate
            base_phase_lr: Base phase learning rate
            freq_decay: Decay factor for frequency-dependent LR
            momentum: Momentum coefficient
        """
        defaults = dict(
            lr=lr,
            base_phase_lr=base_phase_lr,
            freq_decay=freq_decay,
            momentum=momentum
        )
        super().__init__(params, defaults)
    
    def _frequency_lr_schedule(self, freq_idx, total_freqs, base_lr, decay):
        """
        Compute frequency-dependent learning rate.
        
        Low frequencies → smaller LR (stable global patterns)
        High frequencies → larger LR (adaptable details)
        """
        normalized_freq = freq_idx / total_freqs
        lr_mult = 1.0 + (1.0 - decay) * normalized_freq
        return base_lr * lr_mult
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with frequency-adaptive phase LR."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            base_phase_lr = group['base_phase_lr']
            freq_decay = group['freq_decay']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if not torch.is_complex(p):
                    # Standard update for real parameters
                    grad = p.grad
                    param_state = self.state[p]
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    
                    p.add_(buf, alpha=-lr)
                    continue
                
                # Complex parameter - assume frequency dimension
                grad = p.grad
                mag = torch.abs(p)
                phase = torch.angle(p)
                
                grad_mag = torch.real(grad * torch.conj(p / (mag + 1e-8)))
                grad_phase = torch.imag(grad * torch.conj(p / (mag + 1e-8)))
                
                # Get frequency dimension (assume last dim or 2nd dim)
                if len(p.shape) >= 2:
                    freq_dim = p.shape[-2] if p.shape[-1] < p.shape[-2] else p.shape[-1]
                else:
                    freq_dim = p.shape[0]
                
                # Frequency-adaptive phase learning rate
                phase_lrs = torch.tensor([
                    self._frequency_lr_schedule(i, freq_dim, base_phase_lr, freq_decay)
                    for i in range(freq_dim)
                ], device=p.device)
                
                # Expand to match parameter shape
                for _ in range(len(p.shape) - 1):
                    phase_lrs = phase_lrs.unsqueeze(-1)
                
                # Update with adaptive LR
                new_mag = mag - lr * grad_mag
                new_mag = torch.clamp(new_mag, min=1e-8)
                
                new_phase = phase - phase_lrs * grad_phase
                new_phase = torch.atan2(torch.sin(new_phase), torch.cos(new_phase))
                
                p.copy_(new_mag * torch.exp(1j * new_phase))
        
        return loss


class PhaseSignSGD(Optimizer):
    """
    Sign-based phase optimizer (improved version).
    
    Uses only the sign of phase gradients for more stable updates.
    Inspired by signSGD but adapted for complex manifolds.
    """
    
    def __init__(self, params, lr=1e-3, phase_lr=1e-4, momentum=0.9):
        defaults = dict(lr=lr, phase_lr=phase_lr, momentum=momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Step using sign of phase gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            phase_lr = group['phase_lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if torch.is_complex(p):
                    grad = p.grad
                    mag = torch.abs(p)
                    phase = torch.angle(p)
                    
                    grad_mag = torch.real(grad * torch.conj(p / (mag + 1e-8)))
                    grad_phase = torch.imag(grad * torch.conj(p / (mag + 1e-8)))
                    
                    # Sign-based phase update (more robust)
                    phase_sign = torch.sign(grad_phase)
                    
                    param_state = self.state[p]
                    if 'phase_momentum' not in param_state:
                        param_state['phase_momentum'] = torch.zeros_like(phase_sign)
                    
                    phase_mom = param_state['phase_momentum']
                    phase_mom.mul_(momentum).add_(phase_sign, alpha=1 - momentum)
                    
                    # Update
                    new_mag = mag - lr * grad_mag
                    new_mag = torch.clamp(new_mag, min=1e-8)
                    
                    new_phase = phase - phase_lr * phase_mom
                    new_phase = torch.atan2(torch.sin(new_phase), torch.cos(new_phase))
                    
                    p.copy_(new_mag * torch.exp(1j * new_phase))
                else:
                    # Standard SGD for real params
                    p.add_(p.grad, alpha=-lr)
        
        return loss
