# MINIMAL WORKING TRANSFORMER
# Copy this ENTIRE file to: ai_cybersec_custom/model/custom_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleTransformer(nn.Module):
    """Minimal working transformer - no fancy features, just works"""
    
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int = 128,
        num_layers: int = 4, 
        num_heads: int = 4, 
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Return dummy aux_loss for compatibility
        aux_loss = torch.tensor(0.0, device=x.device)
        
        return logits, aux_loss
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_expansion: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FFN(hidden_size, ff_expansion, dropout)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create causal mask
        B, T, D = x.shape
        # Upper triangular matrix (True = mask out, False = keep)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(x.device)
        
        # Attention with residual
        attn_out, _ = self.attn(
            self.ln1(x), 
            self.ln1(x), 
            self.ln1(x),
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.drop(attn_out)
        
        # FFN with residual
        x = x + self.drop(self.ffn(self.ln2(x)))
        
        return x


class FFN(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, hidden_size: int, expansion: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.fc2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# Alias for compatibility
CustomTransformer = SimpleTransformer


if __name__ == "__main__":
    # Test
    model = SimpleTransformer(vocab_size=2000, hidden_size=128, num_layers=4, num_heads=4)
    print(f"Parameters: {model.num_parameters:,}")
    x = torch.randint(0, 2000, (2, 64))
    logits, aux_loss = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
    print("✅ Model working!")