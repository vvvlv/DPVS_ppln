"""Transformer building blocks for vision models."""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for vision transformers.
    
    Applies self-attention over spatial dimensions after flattening feature maps.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W  # Number of spatial positions
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Project to Q, K, V
        qkv = self.qkv(x_flat)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # Final projection
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape back to spatial: (B, N, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network with GELU activation.
    
    Standard transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden layer dimension (typically 4*dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Apply FFN
        out = self.net(x_flat)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class TransformerBlock(nn.Module):
    """
    Vision Transformer Block.
    
    Structure:
        x -> LayerNorm -> Self-Attention -> Add -> LayerNorm -> FFN -> Add -> out
    
    Uses pre-normalization (LayerNorm before each sub-layer) for better training stability.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
        """
        super().__init__()
        
        # Layer normalization (operates on channel dimension)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Self-attention
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForward(dim, mlp_hidden_dim, dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Self-attention with residual
        B, C, H, W = x.shape
        x_norm = self.norm1(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.attn(x_norm)
        
        # FFN with residual
        x_norm = self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.ffn(x_norm)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Blocks.
    
    Can be used as a bottleneck or intermediate layer in CNNs.
    """
    
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Feature dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        for block in self.blocks:
            x = block(x)
        return x

