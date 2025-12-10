"""
Simple GPT-style Transformer for text generation.
A reliable, standard implementation for language modeling tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SimpleTransformer(nn.Module):
    """
    Simple GPT-style transformer for language modeling.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm like GPT
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (common practice)
        self.head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, dataset_idx=None, return_attention=False, classification=True):
        """
        Forward pass.
        
        Args:
            x: Input token ids (B, L)
            dataset_idx: Dataset index (unused, for compatibility)
            return_attention: Whether to return attention (unused, for compatibility)
            classification: Whether this is classification (False for text generation)
            
        Returns:
            logits: Output logits (B, L, vocab_size) for text, (B, vocab_size) for classification
            aux_loss: Auxiliary loss (0 for compatibility)
            gate: Gate values (None for compatibility)
            attention: Attention weights (None for compatibility)
            hidden: Hidden states (None for compatibility)  
            think_stats: Thinking statistics (None for compatibility)
        """
        B, L = x.shape
        
        # Embeddings
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
        x_emb = self.dropout(x_emb)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Transformer
        hidden = self.transformer(x_emb, mask=causal_mask)
        
        # Output
        hidden = self.ln_f(hidden)
        logits = self.head(hidden)
        
        # For compatibility with trainer expectations
        aux_loss = torch.tensor(0.0, device=x.device)
        gate = None
        attention = None
        think_stats = None
        
        # Return format expected by trainer
        return logits, aux_loss, gate, attention, hidden, think_stats
    
    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None, device='cpu'):
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Starting token sequence (list or tensor)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            device: Device to run on
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        if isinstance(start_tokens, list):
            tokens = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
        else:
            tokens = start_tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # Crop to max_len if needed
            if tokens.size(1) > self.max_len:
                tokens = tokens[:, -self.max_len:]
            
            # Forward pass
            logits, _, _, _, _, _ = self.forward(tokens, classification=False)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens.squeeze(0)


# Alias for compatibility
GPTTransformer = SimpleTransformer