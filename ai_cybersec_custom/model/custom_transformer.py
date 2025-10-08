import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Rotary positional embedding (simplified)
        seq_len = x.size(-2)
        freqs = torch.arange(0, self.dim, 2, device=x.device).float() / self.dim
        angles = torch.arange(seq_len, device=x.device).float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return x * emb.unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        # Transpose to [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.rotary(q)
        k = self.rotary(k)
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, self.num_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
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

class MoE(nn.Module):
    def __init__(self, dim, num_experts=4, ff_expansion=4):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(dim, dim * ff_expansion) for _ in range(num_experts)
        ])
        self.proj = nn.ModuleList([
            nn.Linear(dim * ff_expansion, dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [B, T, num_experts]
        out = 0
        for i, (expert, proj) in enumerate(zip(self.experts, self.proj)):
            expert_out = proj(expert(x))
            out = out + expert_out * gate_scores[..., i].unsqueeze(-1)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_expansion, num_experts=4):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = RMSNorm(hidden_size)
        self.moe = MoE(hidden_size, num_experts=num_experts, ff_expansion=ff_expansion)
        self.norm2 = RMSNorm(hidden_size)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class CustomTransformer(nn.Module):
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
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits

# Example usage:
# model = CustomTransformer(vocab_size=32000)
# x = torch.randint(0, 32000, (2, 128))
# logits = model(x)
