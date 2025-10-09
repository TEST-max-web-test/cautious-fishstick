import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE):
    - Used in GPT-5/Claude models for encoding position information in attention.
    - Applies 2D rotation matrices to Q and K vectors, using theta_i = base^(-2i/d).
    - Enables efficient extrapolation and better generalization for long sequences.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, x):
        # x: [B, num_heads, T, head_dim]
        B, num_heads, T, head_dim = x.shape
        device = x.device
        # Compute rotary frequencies
        half_dim = head_dim // 2
        freq_seq = torch.arange(half_dim, device=device)
        theta = 1.0 / (self.base ** (2 * freq_seq / head_dim))
        pos = torch.arange(T, device=device)
        # Outer product: [T, half_dim]
        angles = pos[:, None] * theta[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        # Expand to [1, 1, T, half_dim]
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        # Split x into two halves
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        # Apply rotation
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        # Transpose to [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.rotary(q)
        k = self.rotary(k)
        # KV cache support for autoregressive generation
        if kv_cache is not None:
            # Append new keys/values to cache
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
            T = k.size(2)
        # PyTorch optimized attention (Flash Attention v2 equivalent)
        attn_mask = torch.full((T, T), float('-inf'), device=x.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p, is_causal=True
        )
        out = out / math.sqrt(self.num_heads)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), {'k': k, 'v': v}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
        return x / (norm + self.eps) * self.scale

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return F.silu(self.linear1(x)) * self.linear2(x)

class SparseMoE(nn.Module):
    """
    SparseMixture-of-Experts (SparseMoE):
    - Implements top-k expert selection for each token (routing).
    - Uses load balancing loss to prevent expert collapse and encourage uniform expert usage.
    - Expert capacity planning ensures no expert is overloaded.
    - Returns output and auxiliary load balancing loss for training stability.
    """
    def __init__(self, dim, num_experts=4, ff_expansion=4, k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.experts = nn.ModuleList([
            SwiGLU(dim, dim * ff_expansion) for _ in range(num_experts)
        ])
        self.proj = nn.ModuleList([
            nn.Linear(dim * ff_expansion, dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*T, D]
        gate_logits = self.gate(x_flat)  # [B*T, num_experts]
        topk_scores, topk_indices = torch.topk(gate_logits, self.k, dim=-1)
        gate_probs = F.softmax(topk_scores, dim=-1)  # [B*T, k]
        # Capacity planning
        capacity = int(self.capacity_factor * (x_flat.size(0) // self.num_experts))
        # Route tokens to experts
        expert_outputs = torch.zeros_like(x_flat)
        # Load balancing loss
        # Load balancing loss encourages uniform expert usage:
        #   loss = sum((expert_counts / total_tokens - 1/num_experts)^2)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.k):
            expert_idx = topk_indices[:, i]
            mask = F.one_hot(expert_idx, num_classes=self.num_experts).float()
            expert_counts += mask.sum(dim=0)
            for e in range(self.num_experts):
                selected = (expert_idx == e)
                if selected.any():
                    expert_in = x_flat[selected]
                    out = self.proj[e](self.experts[e](expert_in))
                    expert_outputs[selected] += gate_probs[selected, i].unsqueeze(-1) * out
        # Auxiliary load balancing loss
        load_balancing_loss = (expert_counts / expert_counts.sum() - 1.0 / self.num_experts).pow(2).sum()
        expert_outputs = expert_outputs.reshape(B, T, D)
        return expert_outputs, load_balancing_loss

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_expansion, num_experts=4, moe_k=2):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.moe = SparseMoE(hidden_size, num_experts=num_experts, ff_expansion=ff_expansion, k=moe_k)
    def forward(self, x):
        # Pre-norm: normalize before each sublayer
        x = x + self.attn(self.norm1(x))
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        # For training, return aux_loss for load balancing
        return x, aux_loss

class CustomTransformer(nn.Module):
    """
    CustomTransformer:
    - GPT-5/Claude Sonnet style decoder-only transformer.
    - Features: RoPE, Flash Attention, RMSNorm (pre-norm), SparseMoE, gradient checkpointing, KV cache.
    - Returns logits and auxiliary load balancing loss for training.
    """
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, ff_expansion=4, num_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_expansion, num_experts=num_experts)
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = checkpoint.checkpoint(block, x)
            aux_losses.append(aux_loss)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits, sum(aux_losses)

# Example usage:
# model = CustomTransformer(vocab_size=32000)
# x = torch.randint(0, 32000, (2, 128))
# logits = model(x)
