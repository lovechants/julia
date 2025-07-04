import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.layers import Layer, Linear
from typing import Optional, Tuple
import math
from julia.core.ops_attention import *

"""
Attentions: 
    MLA 
    MHA 
    GQA
    Flash 
"""


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need"

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self, d_model: int, num_heads: int = 8, dropout: float = 0.1, bias: bool = True
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.w_q = Linear(d_model, d_model, bias=bias)
        self.w_k = Linear(d_model, d_model, bias=bias)
        self.w_v = Linear(d_model, d_model, bias=bias)
        self.w_o = Linear(d_model, d_model, bias=bias)

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of multi-head attention

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        from julia.core.ops_attention import (
            ScaledDotProductAttention,
            MultiHeadReshape,
            MultiHeadMerge,
        )

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)  # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = MultiHeadReshape.apply(Q, self.num_heads)
        K = MultiHeadReshape.apply(K, self.num_heads)
        V = MultiHeadReshape.apply(V, self.num_heads)

        # Scaled dot-product attention
        output, attention_weights = ScaledDotProductAttention.apply(
            Q,
            K,
            V,
            mask,
            self.scale,
            self.dropout if self.training else 0.0,
            self.training,
        )

        # Merge heads: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        target_shape = (batch_size, seq_len, self.d_model)
        output = MultiHeadMerge.apply(output, target_shape)

        # Final linear projection
        output = self.w_o(output)

        return output, attention_weights


class SelfAttention(Layer):
    """
    Self-Attention layer (simplified version of MultiHeadAttention where Q=K=V)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of self-attention

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        return self.attention(x, x, x, mask)


class GroupedQueryAttention(Layer):
    """
    Grouped Query Attention (GQA) - more efficient than MHA for large models

    Uses fewer key-value heads than query heads, with queries grouped to share K,V projections.

    Args:
        d_model: Model dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_query_heads)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (
            d_model % num_query_heads == 0
        ), "d_model must be divisible by num_query_heads"
        assert (
            num_query_heads % num_kv_heads == 0
        ), "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.d_k = d_model // num_query_heads
        self.dropout = dropout

        # Linear projections
        self.w_q = Linear(d_model, num_query_heads * self.d_k)
        self.w_k = Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of grouped query attention
        """

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, num_query_heads * d_k)
        K = self.w_k(key)  # (batch_size, seq_len, num_kv_heads * d_k)
        V = self.w_v(value)  # (batch_size, seq_len, num_kv_heads * d_k)

        # Reshape for attention computation
        Q = MultiHeadReshape.apply(
            Q, self.num_query_heads
        )  # (batch_size, num_query_heads, seq_len, d_k)
        K = MultiHeadReshape.apply(
            K, self.num_kv_heads
        )  # (batch_size, num_kv_heads, seq_len, d_k)
        V = MultiHeadReshape.apply(
            V, self.num_kv_heads
        )  # (batch_size, num_kv_heads, seq_len, d_k)

        # Expand K and V to match Q's number of heads
        K_expanded = GroupedKVExpansion.apply(K, self.num_queries_per_kv)
        V_expanded = GroupedKVExpansion.apply(V, self.num_queries_per_kv)

        # Scaled dot-product attention
        output, attention_weights = ScaledDotProductAttention.apply(
            Q,
            K_expanded,
            V_expanded,
            mask,
            self.scale,
            self.dropout if self.training else 0.0,
            self.training,
        )

        # Merge heads
        target_shape = (batch_size, seq_len, self.d_model)
        output = MultiHeadMerge.apply(output, target_shape)

        # Final projection
        output = self.w_o(output)

        return output, attention_weights


class MultiHeadLatentAttention(Layer):
    """
    Multi-Head Latent Attention (MLA) - uses latent key-value compression

    Compresses key-value representations through a latent bottleneck for efficiency.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        latent_dim: Dimension of latent compression
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Query projection (full dimension)
        self.w_q = Linear(d_model, d_model)

        # Latent compression for keys and values
        self.w_kv_compress = Linear(d_model, latent_dim)
        self.w_k_decompress = Linear(latent_dim, d_model)
        self.w_v_decompress = Linear(latent_dim, d_model)

        # Output projection
        self.w_o = Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of multi-head latent attention
        """

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Query projection (standard)
        Q = self.w_q(query)
        Q = MultiHeadReshape.apply(Q, self.num_heads)

        # Latent compression for K,V
        kv_latent = self.w_kv_compress(key)  # Compress both K and V representations

        # Decompress to get K and V
        K = self.w_k_decompress(kv_latent)
        V = self.w_v_decompress(kv_latent)

        # Reshape K and V
        K = MultiHeadReshape.apply(K, self.num_heads)
        V = MultiHeadReshape.apply(V, self.num_heads)

        # Scaled dot-product attention
        output, attention_weights = ScaledDotProductAttention.apply(
            Q,
            K,
            V,
            mask,
            self.scale,
            self.dropout if self.training else 0.0,
            self.training,
        )

        # Merge heads
        target_shape = (batch_size, seq_len, self.d_model)
        output = MultiHeadMerge.apply(output, target_shape)

        # Final projection
        output = self.w_o(output)

        return output, attention_weights


class PositionalEncoding(Layer):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need"

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout

        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)

        # Create div_term for sinusoidal pattern
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cos to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        # Add batch dimension
        pe = pe[None, :, :]  # (1, max_len, d_model)

        # Register as buffer (non-trainable parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        seq_len = x.shape[1]

        # Get positional encodings for current sequence length
        pe_slice = self.pe[:, :seq_len, :]
        pe_tensor = Tensor(pe_slice, requires_grad=False)

        # Add positional encoding
        x = x + pe_tensor

        # Apply dropout if training
        if self.training and self.dropout > 0:
            x = x.dropout(self.dropout, self.training)

        return x


class FlashAttention(Layer):
    """
    Placeholder for Flash Attention - would require specialized CUDA kernels
    For now, falls back to standard attention
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        print("Warning: Flash Attention not implemented - using standard attention")
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass - currently delegates to standard attention
        """
        return self.attention(query, key, value, mask)


# Utility functions for creating attention masks


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create causal (lower triangular) mask for autoregressive models

    Args:
        seq_len: Sequence length

    Returns:
        Causal mask tensor of shape (seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return Tensor(mask, requires_grad=False)


def create_padding_mask(lengths: list, max_len: int) -> Tensor:
    """
    Create padding mask for variable length sequences

    Args:
        lengths: List of actual sequence lengths
        max_len: Maximum sequence length

    Returns:
        Padding mask tensor of shape (batch_size, max_len)
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, length in enumerate(lengths):
        if length < max_len:
            mask[i, length:] = True

    return Tensor(mask, requires_grad=False)


def create_attention_mask(
    causal: bool = False, seq_len: int = None, padding_lengths: list = None
) -> Tensor:
    """
    Create combined attention mask

    Args:
        causal: Whether to apply causal masking
        seq_len: Sequence length (required if causal=True)
        padding_lengths: List of sequence lengths for padding mask

    Returns:
        Combined attention mask
    """
    masks = []

    if causal and seq_len is not None:
        causal_mask = create_causal_mask(seq_len)
        masks.append(causal_mask)

    if padding_lengths is not None:
        max_len = seq_len or max(padding_lengths)
        padding_mask = create_padding_mask(padding_lengths, max_len)
        # Expand dimensions for broadcasting with attention scores
        padding_mask = Tensor(padding_mask.data[:, None, :], requires_grad=False)
        masks.append(padding_mask)

    if masks:
        # Combine masks with logical OR
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = Tensor(
                np.logical_or(combined_mask.data, mask.data), requires_grad=False
            )
        return combined_mask

    return None


# TODO -> BeamSearch, Position wise feed forward
# Flip a coin on encode + decode layers
