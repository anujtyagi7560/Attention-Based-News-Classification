# models/bert_classifier.py
"""
BERT fine-tuning wrapper for sequence classification.

Uses HuggingFace `transformers.BertForSequenceClassification`, which
appends a linear classification head on top of the [CLS] token output.

This module provides:
  - BERTClassifier   : thin nn.Module wrapper with a clean forward interface
  - build_bert_model : factory using config dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from typing import Optional


class BERTClassifier(nn.Module):
    """
    Wraps `BertForSequenceClassification` to expose a consistent interface
    with the rest of the codebase.

    Args:
        model_name  : HuggingFace model identifier (e.g. 'bert-base-uncased').
        num_classes : Number of output classes.
        dropout     : Classifier dropout (overrides BERT default 0.1).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids      : (B, L) token IDs from BertTokenizer.
            attention_mask : (B, L) 1 for real tokens, 0 for padding.
            labels         : (B,) optional — if provided, loss is computed
                             internally by HuggingFace and returned.

        Returns:
            dict with:
              "logits" : (B, num_classes)
              "loss"   : scalar if labels were provided, else None.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    # ------------------------------------------------------------------
    # Parameter groups for differential learning rates
    # ------------------------------------------------------------------

    def get_optimizer_grouped_parameters(
        self,
        bert_lr: float,
        head_lr: float,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        Returns two parameter groups:
          1. BERT backbone params at `bert_lr`
          2. Classification head params at `head_lr`

        No weight decay is applied to bias / LayerNorm parameters.
        """
        no_decay = ["bias", "LayerNorm.weight"]

        backbone_params = [
            p for n, p in self.bert.bert.named_parameters()
        ]
        head_params = [
            p for n, p in self.bert.classifier.named_parameters()
        ]

        def split_by_decay(params, named_params):
            decay, no_d = [], []
            for name, param in named_params:
                if any(nd in name for nd in no_decay):
                    no_d.append(param)
                else:
                    decay.append(param)
            return decay, no_d

        backbone_named = list(self.bert.bert.named_parameters())
        head_named = list(self.bert.classifier.named_parameters())

        bb_decay, bb_no_decay = split_by_decay(backbone_params, backbone_named)
        hd_decay, hd_no_decay = split_by_decay(head_params, head_named)

        return [
            {"params": bb_decay,    "lr": bert_lr, "weight_decay": weight_decay},
            {"params": bb_no_decay, "lr": bert_lr, "weight_decay": 0.0},
            {"params": hd_decay,    "lr": head_lr, "weight_decay": weight_decay},
            {"params": hd_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_bert_model(cfg: dict) -> BERTClassifier:
    """
    Instantiate BERTClassifier from config.

    Args:
        cfg: Full config dict.

    Returns:
        Initialised BERTClassifier.
    """
    bcfg = cfg["bert"]
    return BERTClassifier(
        model_name=bcfg["model_name"],
        num_classes=cfg["data"]["num_classes"],
    )
