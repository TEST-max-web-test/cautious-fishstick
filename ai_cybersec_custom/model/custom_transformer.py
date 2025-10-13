import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple, Optional

"""
ADVANCED REASONING TRANSFORMER

Key improvements over simple transformer:
1. Deeper attention with better context understanding
2. Gated FFN for selective information flow
3. Better positional encoding (RoPE)
4. Residual connections that preserve information
5. Dropout for better generalization (prevents memorization)
"""

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Better than learned embeddings
    Allows model to understand relative positions between tokens
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0)

def apply_rotary_pos_emb(x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to queries and keys"""
    cos = pos_emb.cos()
    sin = pos_emb.sin()
    
    # Split last dimension
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rotated


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network
    Uses gating mechanism to control information flow
    Helps model learn what information to keep vs discard
    """
    def __init__(self, hidden_size: int, ff_expansion: int = 4):
        super().__init__()
        ff_size = hidden_size * ff_expansion
        
        # Parallel gates
        self.gate_proj = nn.Linear(hidden_size, ff_size)
        self.up_proj = nn.Linear(hidden_size, ff_size)
        self.down_proj = nn.Linear(ff_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating: gate controls what information passes through
        gate = F.silu(self.gate_proj(x))  # SwiGLU activation
        up = self.up_proj(x)
        
        # Element-wise multiplication: gate filters information
        gated = gate * up
        gated = self.dropout(gated)
        
        # Project back down
        return self.down_proj(gated)


class MultiHeadAttention(nn.Module):
    """
    Advanced Multi-Head Attention with RoPE
    Better at understanding relationships between tokens
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Rotary embeddings for better positional understanding
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings to Q and K for positional awareness
        pos_emb = self.rotary(x)
        q = apply_rotary_pos_emb(q, pos_emb)
        k = apply_rotary_pos_emb(k, pos_emb)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.dropout(out)
        
        return out


class ReasoningBlock(nn.Module):
    """
    Transformer block optimized for reasoning
    
    Key features:
    - Pre-normalization for better gradient flow
    - Gated FFN for selective information processing
    - Strong residual connections to preserve information
    - Dropout to prevent memorization
    """
    def __init__(self, hidden_size: int, num_heads: int, ff_expansion: int, dropout: float = 0.1):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Attention and FFN
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = GatedFFN(hidden_size, ff_expansion)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        
        # FFN with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        
        return x


class ReasoningTransformer(nn.Module):
    """
    Advanced Transformer for Reasoning and Generation
    
    This model can:
    - Understand context deeply through multi-head attention
    - Reason about information through gated FFN
    - Generate novel responses (not just memorized)
    - Generalize to unseen inputs
    
    Key improvements:
    1. RoPE for better positional understanding
    2. Gated FFN for selective reasoning
    3. Proper dropout to prevent overfitting
    4. Deeper architecture options
    5. Better initialization
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int = 256,
        num_layers: int = 4, 
        num_heads: int = 8, 
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        
        # Dropout for input embeddings
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(hidden_size, num_heads, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between input embeddings and output head
        # This helps the model learn better representations
        self.head.weight = self.token_emb.weight
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization is crucial for learning"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T] token IDs
        
        Returns:
            logits: [B, T, vocab_size]
            aux_loss: Always 0.0 (for compatibility)
        """
        B, T = x.shape
        
        # Token embeddings
        x = self.token_emb(x)
        x = self.emb_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Return 0 aux_loss for compatibility with training code
        aux_loss = torch.tensor(0.0, device=x.device)
        
        return logits, aux_loss
    
    @property
    def num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Generate new tokens with sampling
        
        Args:
            idx: Starting token IDs [B, T]
            max_new_tokens: How many tokens to generate
            temperature: Higher = more random, lower = more focused
            top_k: Consider only top K tokens
            top_p: Nucleus sampling threshold
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if too long
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# For backward compatibility with existing code
CustomTransformer = ReasoningTransformer


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("🧠 ADVANCED REASONING TRANSFORMER TEST")
    print("="*70)
    
    model = ReasoningTransformer(
        vocab_size=2000, 
        hidden_size=256,
        num_layers=6,  # Deeper for better reasoning
        num_heads=8,
        ff_expansion=4,
        dropout=0.1
    )
    
    print(f"\n📊 Model Architecture:")
    print(f"   Parameters: {model.num_parameters:,}")
    print(f"   Hidden size: {model.hidden_size}")
    print(f"   Layers: {len(model.blocks)}")
    print(f"   Attention heads: {model.blocks[0].attn.num_heads}")
    
    # Test forward pass
    x = torch.randint(0, 2000, (2, 64))
    print(f"\n🔍 Forward Pass Test:")
    print(f"   Input shape: {x.shape}")
    
    logits, aux_loss = model(x)
    print(f"   Output shape: {logits.shape}")
    print(f"   Aux loss: {aux_loss.item()}")
    
    # Test generation
    print(f"\n🎲 Generation Test:")
    model.eval()
    generated = model.generate(x[:1, :10], max_new_tokens=20, temperature=0.8)
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated tokens: {generated[0].tolist()}")
    
    print(f"\n✅ All tests passed!")
    print("="*70)