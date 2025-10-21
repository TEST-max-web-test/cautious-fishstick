#!/usr/bin/env python3
"""
MODERN TRANSFORMER ARCHITECTURE
Implements techniques from GPT-3/4, LLaMA, and other SOTA models:
- Rotary Positional Embeddings (RoPE) - better position encoding
- Pre-Layer Normalization - more stable training
- SwiGLU activation - better than GELU
- Proper weight initialization
- GPU optimized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Used in LLaMA, GPT-NeoX, and other modern LLMs
    Better than absolute position embeddings for length extrapolation
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Pre-compute for efficiency
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to q and k tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in LLaMA - more efficient than LayerNorm
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    """
    SwiGLU activation from "GLU Variants Improve Transformer"
    Used in LLaMA and PaLM - better than GELU
    """
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size * expansion, bias=False)
        self.w2 = nn.Linear(hidden_size * expansion, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, hidden_size * expansion, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Multi-head attention with RoPE and Flash Attention compatibility
    Implements Grouped Query Attention (GQA) for efficiency
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, num_kv_heads: int = None):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Grouped Query Attention (used in LLaMA 2, Mistral)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, cos, sin, mask=None, use_cache=False, past_kv=None):
        B, T, C = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV caching for faster inference
        if use_cache:
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
            past_kv = (k, v)
        
        # Expand KV heads for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Scaled Dot-Product Attention with improved numerical stability
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.dropout(out)
        
        if use_cache:
            return out, past_kv
        return out


class TransformerBlock(nn.Module):
    """
    Modern Transformer block with:
    - Pre-layer normalization (more stable)
    - RoPE attention with GQA
    - SwiGLU FFN
    - Gradient checkpointing support
    """
    def __init__(self, hidden_size: int, num_heads: int, expansion: int, dropout: float, num_kv_heads: int = None):
        super().__init__()
        
        # Pre-norm for stability
        self.ln1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttentionWithRoPE(hidden_size, num_heads, dropout, num_kv_heads)
        
        self.ln2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, expansion)
        
        # Dropout is applied inside attn and ffn now
    
    def forward(self, x, cos, sin, mask, use_cache=False, past_kv=None):
        # Pre-norm + attention + residual
        if use_cache:
            attn_out, past_kv = self.attn(self.ln1(x), cos, sin, mask, use_cache, past_kv)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x), cos, sin, mask)
        
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.ln2(x))
        
        if use_cache:
            return x, past_kv
        return x


class ModernTransformer(nn.Module):
    """
    Modern Transformer Language Model
    Incorporates best practices from GPT-3/4, LLaMA 2, Mistral:
    - Grouped Query Attention (GQA)
    - RoPE positional embeddings
    - RMSNorm
    - SwiGLU activation
    - KV caching
    - Gradient checkpointing
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int = None,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # GQA: reduce KV heads for efficiency (like LLaMA 2)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        
        # RoPE instead of learned positional embeddings
        self.rope = RotaryPositionalEmbedding(hidden_size // num_heads, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks with GQA
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_expansion, dropout, self.num_kv_heads)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.ln_f = RMSNorm(hidden_size)
        
        # Output head (weight tied with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Scale embeddings (like GPT-2)
        with torch.no_grad():
            self.token_emb.weight.data.normal_(mean=0.0, std=0.02)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, past_kvs: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        
        # Embeddings
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Get RoPE embeddings
        cos, sin = self.rope(x, T)
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # Transformer blocks with optional gradient checkpointing
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                if use_cache:
                    x, new_kv = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), x, cos, sin, mask, use_cache, past_kv
                    )
                    new_kvs.append(new_kv)
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), x, cos, sin, mask
                    )
            else:
                if use_cache:
                    x, new_kv = block(x, cos, sin, mask, use_cache, past_kv)
                    new_kvs.append(new_kv)
                else:
                    x = block(x, cos, sin, mask)
        
        # Final norm and output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kvs
        return logits, torch.tensor(0.0, device=x.device)  # Dummy aux loss
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with proper sampling
        """
        self.eval()
        generated = []
        
        for _ in range(max_new_tokens):
            # Truncate if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Get last token logits
            
            # Apply repetition penalty
            if generated:
                for token_id in set(generated):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            # Temperature
            logits = logits / temperature
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(next_token.item())
        
        return input_ids


# Alias for compatibility
CustomTransformer = ModernTransformer


if __name__ == "__main__":
    # Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModernTransformer(
        vocab_size=2000,
        hidden_size=256,
        num_layers=6,
        num_heads=8
    ).to(device)
    
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Device: {device}")
    
    x = torch.randint(0, 2000, (2, 64), device=device)
    logits, _ = model(x)
    
    print(f"Input: {x.shape}, Output: {logits.shape}")
    print("✅ Modern Transformer working!")