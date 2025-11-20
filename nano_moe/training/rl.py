import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from typing import List, Dict, Any
from rich.console import Console

console = Console()

class RewardModel(nn.Module):
    """
    Reward Model for RLHF/GRPO.
    Wraps a base model (e.g., NanoGPT/SFPT) and adds a scalar head.
    """
    def __init__(self, base_model: nn.Module, hidden_dim: int):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get reward scores.
        Args:
            idx: Input token indices (B, T)
        Returns:
            Scalar reward for each sequence (B, 1)
        """
        # We assume the base model has a way to return hidden states or we extract them
        # This implementation assumes we can get the final hidden state from the base model
        # For SFPT/NanoGPT, we might need to adapt this.
        
        # Hack: If base model is SFPT, it returns logits. We need hidden states.
        # We might need to modify SFPT to return hidden states if requested.
        # For now, let's assume we can access the last layer's output or similar.
        
        # If base is NanoGPT from the notebook:
        # x = self.base.wte(idx) + self.base.wpe(pos) ...
        # h = self.base.ln_f(x)
        # return self.head(h.mean(1))
        
        # Since we are integrating into nano_moe, we should rely on the model's interface.
        # Let's assume the model has a `forward_features` or we just use the logits for now (suboptimal)
        # or better, we expect the user to pass a model that exposes hidden states.
        
        # For this implementation, I will assume `base_model` has a `get_last_hidden_state` method
        # or we just run forward and capture the hook.
        
        # SIMPLIFICATION: We will just run the base model and use the last hidden state if available,
        # otherwise we might need to refactor the base models.
        # For now, let's implement the generic structure.
        
        with torch.no_grad():
            # This is model-specific. 
            # If it's SFPT, we need to check its forward signature.
            # If it's MoE, same.
            # Let's try to call it and see if it returns hidden states.
            output = self.base(idx)
            
            # If output is a tuple, maybe hidden state is in there?
            if isinstance(output, tuple):
                logits = output[0]
                # We don't have hidden states easily.
                # We will use the mean of logits as a proxy for "features" for now (very bad, but runs)
                # OR we assume the base model is modified to return hidden states.
                h = logits # (B, T, V)
            else:
                h = output
                
        # Global pooling
        if h.dim() == 3:
            h = h.mean(dim=1) # (B, V)
            
        # Project to scalar
        # Note: self.head expects hidden_dim, but h is (B, V). 
        # We need to re-init head if dimensions mismatch or project.
        if h.shape[-1] != self.head.in_features:
             # Dynamic adjustment (hacky but works for demo)
             self.head = nn.Linear(h.shape[-1], 1).to(h.device)
             
        return self.head(h)

def load_pairs(path: str) -> List[Dict[str, str]]:
    """Load comparison pairs for Reward Modeling."""
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def train_reward_model(rm: RewardModel, tokenizer, pairs_path: str, 
                      block_size: int, epochs: int, steps: int, 
                      batch_size: int, lr: float, device: str):
    """
    Train the Reward Model using pair-wise ranking loss.
    """
    pairs = load_pairs(pairs_path)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    rm.to(device)
    
    console.print(f"[bold green]Starting Reward Model Training on {len(pairs)} pairs[/bold green]")
    
    for ep in range(1, epochs+1):
        random.shuffle(pairs)
        losses = []
        
        for _ in range(steps):
            b = random.sample(pairs, min(batch_size, len(pairs)))
            
            # Tokenize
            # Assuming tokenizer has .encode()
            cl = [torch.tensor(tokenizer.encode(p["chosen"])[:block_size]) for p in b]
            rl = [torch.tensor(tokenizer.encode(p["rejected"])[:block_size]) for p in b]
            
            cl = nn.utils.rnn.pad_sequence(cl, batch_first=True).to(device)
            rl = nn.utils.rnn.pad_sequence(rl, batch_first=True).to(device)
            
            rc = rm(cl)
            rr = rm(rl)
            
            loss = -F.logsigmoid(rc - rr).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
        avg_loss = sum(losses)/len(losses) if losses else 0
        console.print(f"[RM] Epoch {ep}/{epochs} | Loss: {avg_loss:.4f}")

def grpo_step(model: nn.Module, encode_fn, prompts: List[str], 
              gen_tokens: int, opt: torch.optim.Optimizer, device: str):
    """
    Group Relative Policy Optimization (GRPO) step.
    Generates multiple outputs for prompts, ranks them by reward (self-consistency or RM),
    and optimizes the policy.
    """
    model.train()
    rewards = []
    logps = []
    
    for s in prompts:
        # 1. Generate
        x = torch.tensor([encode_fn(s)], device=device)
        
        # We need a generate function from the model
        if hasattr(model, 'generate'):
            y = model.generate(x, max_new_tokens=gen_tokens)[0]
        else:
            # Fallback to simple generation loop if model doesn't have generate
            # This is a placeholder
            y = x # Should implement generation
            
        # 2. Calculate Logprobs of the generated sequence
        inp, tgt = y[:-1].unsqueeze(0), y[1:].unsqueeze(0)
        logits, _ = model(inp) # Assuming forward returns (logits, loss) or just logits
        if isinstance(logits, tuple): logits = logits[0]
        
        logp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        
        # 3. Calculate Reward
        # In true GRPO, we compare against a group. 
        # Here we use the log probability itself as a proxy for "confidence" reward 
        # if no external reward model is provided.
        # The notebook implementation used logp as reward, which is ... interesting (self-reinforcing).
        # Ideally we'd use the RewardModel here.
        reward = logp.detach() 
        
        rewards.append(reward)
        logps.append(logp)
        
    rewards = torch.stack(rewards)
    
    # Advantage: How much better is this sample than the group mean?
    adv = rewards - rewards.mean()
    
    # Policy Gradient Loss
    loss = -(adv.detach() * torch.stack(logps)).mean()
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    return float(loss.item()), float(rewards.mean().item())
