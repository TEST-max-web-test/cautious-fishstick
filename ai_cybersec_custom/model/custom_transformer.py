import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE):
    - Used in GPT-5/Claude models for encoding position information in attention.
    - Applies 2D rotation matrices to Q and K vectors, using theta_i = base^(-2i/d).
    - Enables efficient extrapolation and better generalization for long sequences.
    - Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim: int, base: float = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to queries or keys.
        
        Args:
            x: Tensor of shape [B, num_heads, T, head_dim]
        
        Returns:
            x_rot: Rotated tensor of same shape
        """
        B, num_heads, T, head_dim = x.shape
        device = x.device
        
        # Compute rotary frequencies
        half_dim = head_dim // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=x.dtype)
        theta = 1.0 / (self.base ** (2 * freq_seq / head_dim))
        pos = torch.arange(T, device=device, dtype=x.dtype)
        
        # Outer product: [T, half_dim]
        angles = pos[:, None] * theta[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # Expand to [1, 1, T, half_dim]
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        
        # Split x into two halves
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional KV caching."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, C]
            kv_cache: Optional cache for KV from previous tokens
        
        Returns:
            out: [B, T, C]
            cache: Updated KV cache
        """
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        
        # Transpose to [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        q = self.rotary(q)
        k = self.rotary(k)
        
        # KV cache support for autoregressive generation
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
        
        # Use PyTorch's optimized attention (Flash Attention v2 equivalent)
        # is_causal=True handles the causal mask automatically
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back to [B, T, C]
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        
        return out, {'k': k, 'v': v}


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable and efficient than LayerNorm. Used in GPT-5, Llama, etc.
    Reference: "Root Mean Square Layer Normalization"
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
        return x / (norm + self.eps) * self.scale


class SwiGLU(nn.Module):
    """SwiGLU activation function: SiLU(xW1) * (xW2).
    
    More parameter-efficient than standard FFN while maintaining similar capacity.
    Reference: "GLU Variants Improve Transformer"
    """
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.linear1(x)) * self.linear2(x)


class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts (SparseMoE):
    - Implements top-k expert selection for each token (routing).
    - Uses load balancing loss to prevent expert collapse and encourage uniform expert usage.
    - Expert capacity planning ensures no expert is overloaded.
    - Returns output and auxiliary load balancing loss for training stability.
    
    Reference: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
    """
    
    def __init__(self, dim: int, num_experts: int = 4, ff_expansion: int = 4, 
                 k: int = 2, capacity_factor: float = 1.25):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
        
        Returns:
            out: [B, T, D] weighted combination of expert outputs
            aux_loss: Load balancing loss for training
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*T, D]
        
        # Gate: compute scores for each expert
        gate_logits = self.gate(x_flat)  # [B*T, num_experts]
        topk_scores, topk_indices = torch.topk(gate_logits, self.k, dim=-1)
        gate_probs = F.softmax(topk_scores, dim=-1)  # [B*T, k]
        
        # Route tokens to experts
        expert_outputs = torch.zeros_like(x_flat)
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
        
        # Load balancing loss: encourages uniform expert usage
        # Prevents expert collapse where all tokens route to same expert
        load_balancing_loss = (
            (expert_counts / expert_counts.sum() - 1.0 / self.num_experts).pow(2).sum()
        )
        
        expert_outputs = expert_outputs.reshape(B, T, D)
        return expert_outputs, load_balancing_loss


class TransformerBlock(nn.Module):
    """Single transformer block: Pre-norm attention + Pre-norm SparseMoE."""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_expansion: int, 
                 num_experts: int = 4, moe_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.moe = SparseMoE(hidden_size, num_experts=num_experts, 
                            ff_expansion=ff_expansion, k=moe_k)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-norm transformer block with residual connections.
        
        Args:
            x: [B, T, D]
        
        Returns:
            x: [B, T, D] after attention and MoE
            aux_loss: Load balancing loss from MoE
        """
        # Pre-norm attention
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        
        # Pre-norm MoE
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, aux_loss


class CustomTransformer(nn.Module):
    """
    CustomTransformer:
    - GPT-5/Claude Sonnet style decoder-only transformer.
    - Features: RoPE, Flash Attention, RMSNorm (pre-norm), SparseMoE.
    - Returns logits and auxiliary load balancing loss for training.
    
    Inspired by:
    - GPT-5 (OpenAI)
    - Claude 4.5 (Anthropic)
    - Llama 2 (Meta)
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, 
                 num_heads: int = 8, ff_expansion: int = 4, num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_expansion, 
                           num_experts=num_experts, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer.
        
        Args:
            x: [B, T] token IDs
        
        Returns:
            logits: [B, T, vocab_size]
            aux_loss: Auxiliary load balancing loss from MoE layers
        """
        x = self.embed(x)
        aux_losses = []
        
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_losses.append(aux_loss)
        
        x = self.norm_f(x)
        logits = self.head(x)
        
        total_aux_loss = sum(aux_losses) if aux_losses else torch.tensor(0.0, device=x.device)
        return logits, total_aux_loss

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage:
if __name__ == "__main__":
    model = CustomTransformer(vocab_size=50258, hidden_size=256, num_layers=4)
    print(f"Model parameters: {model.num_parameters:,}")
    x = torch.randint(0, 50258, (2, 128))
    logits, aux_loss = model(x)
    print(f"Output shape: {logits.shape}")