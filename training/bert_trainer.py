# training/bert_trainer.py
"""
BERT fine-tuning trainer using HuggingFace Trainer API.

Wraps HuggingFace `Trainer` with:
  - Custom compute_metrics (accuracy + macro F1)
  - Linear warmup schedule
  - Mixed precision (fp16) when available
  - Early stopping callback
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, Optional

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from datasets import Dataset as HFDataset

from models.bert_classifier import BERTClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics function
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute accuracy and macro F1.
    Called by HuggingFace Trainer after each evaluation step.
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------

class HFNewsDataset(HFDataset):
    """
    Thin wrapper so PyTorch Datasets work with HuggingFace Trainer.
    HF Trainer expects __getitem__ to return dicts with string keys.
    """
    pass


def pytorch_ds_to_hf(torch_dataset) -> HFDataset:
    """
    Convert a PyTorch AGNewsBERTDataset to a HuggingFace Dataset.
    Needed because HF Trainer expects a HF-compatible dataset.
    """
    import torch
    from torch.utils.data import DataLoader

    all_input_ids, all_masks, all_labels = [], [], []
    loader = DataLoader(torch_dataset, batch_size=256, num_workers=4)
    for batch in loader:
        all_input_ids.append(batch["input_ids"].numpy())
        all_masks.append(batch["attention_mask"].numpy())
        all_labels.append(batch["label"].numpy())

    return HFDataset.from_dict({
        "input_ids": np.concatenate(all_input_ids).tolist(),
        "attention_mask": np.concatenate(all_masks).tolist(),
        "labels": np.concatenate(all_labels).tolist(),
    })


# ---------------------------------------------------------------------------
# BERTTrainer
# ---------------------------------------------------------------------------

class BERTTrainer:
    """
    Wraps HuggingFace `Trainer` to fine-tune BERTClassifier.

    Args:
        model      : BERTClassifier instance.
        tokenizer  : Matching HuggingFace tokenizer.
        cfg        : Full config dict.
        output_dir : Where to write checkpoints and results.
    """

    def __init__(
        self,
        model: BERTClassifier,
        tokenizer: PreTrainedTokenizerBase,
        cfg: dict,
        output_dir: str = "checkpoints/bert",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.output_dir = output_dir

        bcfg = cfg["bert"]
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=bcfg["epochs"],
            per_device_train_batch_size=bcfg["batch_size"],
            per_device_eval_batch_size=bcfg["batch_size"] * 2,
            learning_rate=bcfg["learning_rate"],
            weight_decay=bcfg["weight_decay"],
            warmup_ratio=bcfg["warmup_ratio"],
            fp16=bcfg.get("fp16", False),
            gradient_accumulation_steps=bcfg.get("gradient_accumulation_steps", 1),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=cfg["training"].get("log_every_n_steps", 100),
            report_to="none",           # disable W&B / MLflow by default
            seed=cfg["data"]["random_seed"],
        )

    def train(
        self,
        train_dataset,
        val_dataset,
    ) -> Dict:
        """
        Fine-tune BERT using HuggingFace Trainer.

        Args:
            train_dataset : PyTorch AGNewsBERTDataset (train).
            val_dataset   : PyTorch AGNewsBERTDataset (validation).

        Returns:
            HuggingFace TrainOutput (contains metrics and global_step).
        """
        logger.info("Converting datasets to HuggingFace format …")
        hf_train = pytorch_ds_to_hf(train_dataset)
        hf_val = pytorch_ds_to_hf(val_dataset)

        hf_train.set_format("torch")
        hf_val.set_format("torch")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        patience = self.cfg["training"]["early_stopping_patience"]

        trainer = Trainer(
            model=self.model.bert,          # pass the underlying HF model
            args=self.training_args,
            train_dataset=hf_train,
            eval_dataset=hf_val,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        logger.info("Starting BERT fine-tuning …")
        train_result = trainer.train()
        logger.info(f"BERT training complete: {train_result.metrics}")

        # Save final model and tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

        return train_result.metrics
