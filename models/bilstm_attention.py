# models/bilstm_attention.py
"""
BiLSTM + Self-Attention text classifier.

Architecture:
  Embedding → BiLSTM (stacked) → AdditiveAttention → Dropout → Linear

The attention layer collapses the sequence dimension into a single
context vector, which is then projected to class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import AdditiveAttention, make_padding_mask


class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM with additive self-attention for text classification.

    Args:
        vocab_size   : Size of the input vocabulary (including PAD/UNK).
        embed_dim    : Word embedding dimensionality.
        hidden_dim   : Hidden size per LSTM direction.
        num_classes  : Number of output classes.
        num_layers   : Number of stacked BiLSTM layers.
        dropout      : Dropout probability applied to LSTM output & attention.
        pad_idx      : Index of <PAD> token (embeddings kept as zero vectors).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM outputs hidden_dim * 2 features per token
        lstm_output_dim = hidden_dim * 2

        self.attention = AdditiveAttention(hidden_dim=lstm_output_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ids     : (B, L) LongTensor of token IDs.
            lengths : (B,)   LongTensor of actual (non-padded) lengths.

        Returns:
            dict with keys:
              "logits"   : (B, num_classes) raw class scores.
              "attn_weights" : (B, L) attention distribution for inspection.
        """
        # --- Embedding ---
        embedded = self.dropout(self.embedding(ids))          # (B, L, E)

        # --- Pack for efficiency (handles variable lengths) ---
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )                                                     # (B, L, 2H)

        hidden = self.dropout(hidden)

        # --- Attention ---
        pad_mask = make_padding_mask(lengths, max_len=hidden.size(1))
        context, attn_weights = self.attention(hidden, mask=pad_mask)
        # context: (B, 2H)

        # --- Classification ---
        logits = self.classifier(context)                     # (B, C)

        return {"logits": logits, "attn_weights": attn_weights}

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_bilstm_model(cfg: dict, vocab_size: int, pad_idx: int) -> BiLSTMAttentionClassifier:
    """
    Instantiate BiLSTMAttentionClassifier from a config dict.

    Args:
        cfg       : Full config dict (loaded from config.yaml).
        vocab_size: Number of tokens in the vocabulary.
        pad_idx   : Padding token ID.

    Returns:
        Initialised (but untrained) model.
    """
    bcfg = cfg["bilstm"]
    return BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embed_dim=bcfg["embed_dim"],
        hidden_dim=bcfg["hidden_dim"],
        num_classes=cfg["data"]["num_classes"],
        num_layers=bcfg["num_layers"],
        dropout=bcfg["dropout"],
        pad_idx=pad_idx,
    )
