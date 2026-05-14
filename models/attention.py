# models/attention.py
"""
Self-attention mechanism used on top of BiLSTM hidden states.

Two variants are provided:
  - AdditiveAttention  : Bahdanau-style single-head additive attention
  - ScaledDotAttention : Scaled dot-product (single query vector against all keys)

The BiLSTM classifier uses AdditiveAttention by default.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive self-attention with a learnable query vector.

    Given a sequence of hidden states H ∈ R^{B × L × D}, computes a
    context vector c ∈ R^{B × D} as a weighted sum of H:

        e_t = v^T · tanh(W · h_t + b)
        α   = softmax(e)                  (masked where padding)
        c   = Σ_t α_t · h_t

    Args:
        hidden_dim: Dimensionality of each hidden state vector.
        attn_dim:   Internal projection dimension (default = hidden_dim).
    """

    def __init__(self, hidden_dim: int, attn_dim: int | None = None) -> None:
        super().__init__()
        attn_dim = attn_dim or hidden_dim

        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden : (B, L, D) — BiLSTM output hidden states.
            mask   : (B, L) BoolTensor — True where tokens are padding.
                     Padding positions are excluded from softmax.

        Returns:
            context : (B, D) weighted context vector.
            weights : (B, L) attention weight distribution.
        """
        # (B, L, attn_dim) → (B, L, 1) → (B, L)
        energy = self.v(torch.tanh(self.W(hidden))).squeeze(-1)

        if mask is not None:
            # Fill padding positions with a large negative value
            energy = energy.masked_fill(mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)          # (B, L)

        # Handle all-padding edge case (produces NaN → replace with uniform)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (B, D)
        return context, weights


class ScaledDotAttention(nn.Module):
    """
    Scaled dot-product attention with a single learned query vector.

    Useful as a drop-in replacement for AdditiveAttention when speed
    is preferred over expressiveness.

    Args:
        hidden_dim: Dimensionality of each hidden state vector.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        # Single learnable query vector
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden : (B, L, D)
            mask   : (B, L) BoolTensor for padding positions.

        Returns:
            context : (B, D)
            weights : (B, L)
        """
        # Expand query to (B, D, 1) and compute dot product
        q = self.query.unsqueeze(0).unsqueeze(-1).expand(hidden.size(0), -1, 1)
        energy = torch.bmm(hidden, q).squeeze(-1) / self.scale  # (B, L)

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


# ---------------------------------------------------------------------------
# Utility: build padding mask from lengths
# ---------------------------------------------------------------------------

def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths : (B,) tensor of actual (un-padded) lengths.
        max_len : Total sequence length L.

    Returns:
        mask : (B, L) BoolTensor — True at padding positions.
    """
    batch_size = lengths.size(0)
    # Range tensor (1, L) compared with lengths (B, 1)
    range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = range_tensor >= lengths.unsqueeze(1)     # True = padding
    return mask
