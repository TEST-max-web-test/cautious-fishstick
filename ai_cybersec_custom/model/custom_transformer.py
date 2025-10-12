import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple, Optional

"""
SIMPLIFIED TRANSFORMER - OPTIMIZED FOR SMALL DATASETS

Removes complex features like MoE that hurt learning on small data.
Focuses on core transformer functionality.
"""

class SimpleFFN(nn.Module):
    """Simple feedforward network - no MoE complexity"""
    
    def __init__(self, hidden_size: int, ff_expansion: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * ff_expansion)
        self.fc2 = nn.Linear(hidden_size * ff_expansion, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleAttention(nn.Module):
    """Simplified multi-head attention"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Transpose for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    """Simple transformer block with just attention + FFN"""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_expansion: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = SimpleAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = SimpleFFN(hidden_size, ff_expansion)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """
    Simplified transformer optimized for small datasets.
    
    Key differences from CustomTransformer:
    - No MoE (too complex for small data)
    - Simple LayerNorm instead of RMSNorm
    - No RoPE (positional encoding via learned embeddings)
    - Simpler FFN with GELU
    - No aux_loss to track
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, num_heads: int = 4, ff_expansion: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)  # Max sequence length
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_expansion)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T] token IDs
        
        Returns:
            logits: [B, T, vocab_size]
            aux_loss: Always 0.0 (for compatibility with training code)
        """
        B, T = x.shape
        
        # Token embeddings + positional embeddings
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Return 0 aux_loss for compatibility
        aux_loss = torch.tensor(0.0, device=x.device)
        return logits, aux_loss
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# For backward compatibility with existing code
CustomTransformer = SimpleTransformer


if __name__ == "__main__":
    # Test the model
    model = SimpleTransformer(vocab_size=2000, hidden_size=128, num_layers=3, num_heads=4)
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Test forward pass
    x = torch.randint(0, 2000, (2, 64))
    logits, aux_loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Aux loss: {aux_loss.item()}")