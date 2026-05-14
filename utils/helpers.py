# utils/helpers.py
"""
General utility helpers:
  - set_seed         : reproducibility
  - get_device       : auto-select GPU / MPS / CPU
  - save_checkpoint  : save model + optimizer state
  - load_checkpoint  : restore model weights
  - load_config      : parse config.yaml into a dict
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Auto-select the best available device.

    Priority: CUDA → MPS (Apple Silicon) → CPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
) -> None:
    """
    Save model weights, optimizer state, epoch, and best metric.

    Args:
        model     : PyTorch model.
        optimizer : Optimizer (state saved for potential resumption).
        epoch     : Current epoch number.
        val_acc   : Validation accuracy at this checkpoint.
        path      : File path for the checkpoint (.pt).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load checkpoint weights into model (and optionally optimizer).

    Args:
        model     : Model instance (must match saved architecture).
        path      : Path to the checkpoint file.
        device    : Device to map tensors to.
        optimizer : Pass to also restore optimizer state.

    Returns:
        The full checkpoint dict (contains epoch, val_acc, etc.).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        f"Checkpoint loaded ← {path}  "
        f"(epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?'):.4f})"
    )
    return ckpt


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Parse a YAML config file into a Python dict.

    Args:
        path: Relative or absolute path to config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg
