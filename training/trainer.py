# training/trainer.py
"""
Generic training loop for PyTorch models (used by BiLSTM + Attention).

Features:
  - Train / validation loop with progress bars (tqdm)
  - Learning rate scheduling (cosine annealing or step)
  - Gradient clipping
  - Early stopping
  - Best-model checkpoint saving
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import get_logger
from utils.helpers import set_seed, get_device, save_checkpoint, load_checkpoint

logger = get_logger(__name__)


class Trainer:
    """
    Training loop for any PyTorch nn.Module that returns a dict with 'logits'.

    Args:
        model       : PyTorch model (output dict must contain 'logits').
        cfg         : Full config dict.
        device      : torch.device to train on.
        checkpoint_dir : Directory for saving best checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        bcfg = cfg["bilstm"]
        self.epochs = bcfg["epochs"]
        self.clip_grad = bcfg["clip_grad_norm"]
        self.patience = cfg["training"]["early_stopping_patience"]

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=bcfg["learning_rate"],
            weight_decay=bcfg["weight_decay"],
        )

        # Scheduler
        scheduler_name = bcfg.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif scheduler_name == "step":
            self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        self.best_val_acc = 0.0
        self._no_improve = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> Tuple[float, float]:
        """
        Run one epoch (train or eval).

        Returns:
            (avg_loss, accuracy)
        """
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        desc = "Train" if training else "Val  "
        with torch.set_grad_enabled(training):
            for batch in tqdm(loader, desc=desc, leave=False):
                ids = batch["ids"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                labels = batch["labels"].to(self.device)

                output = self.model(ids, lengths)
                logits = output["logits"]
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "bilstm",
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader : DataLoader for training set.
            val_loader   : DataLoader for validation set.
            model_name   : Prefix for checkpoint filenames.

        Returns:
            history dict with loss/accuracy curves.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs | "
            f"device={self.device} | "
            f"params={self.model.count_parameters():,}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # ---- Early stopping + checkpointing ----
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._no_improve = 0
                ckpt_path = self.checkpoint_dir / f"{model_name}_best.pt"
                save_checkpoint(self.model, self.optimizer, epoch, val_acc, str(ckpt_path))
                logger.info(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved.")
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    logger.info(
                        f"  Early stopping triggered after {self.patience} epochs without improvement."
                    )
                    break

        # Save training history
        hist_path = self.checkpoint_dir / f"{model_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training complete. Best val_acc={self.best_val_acc:.4f}")
        return self.history

    def load_best(self, model_name: str = "bilstm") -> None:
        """Load the best checkpoint back into self.model."""
        ckpt_path = self.checkpoint_dir / f"{model_name}_best.pt"
        load_checkpoint(self.model, str(ckpt_path), self.device)
        logger.info(f"Loaded best checkpoint from {ckpt_path}")
