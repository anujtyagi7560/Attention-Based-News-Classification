# evaluation/evaluator.py
"""
Unified evaluation for all three models.

Provides:
  - evaluate_bilstm   : runs inference on a DataLoader
  - evaluate_bert     : runs inference on a DataLoader (BERT)
  - evaluate_baseline : wraps TFIDFBaseline.evaluate
  - plot_confusion_matrix
  - plot_training_curves
  - compare_models    : prints a side-by-side comparison table
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions_bilstm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BiLSTM model inference.

    Returns:
        all_preds  : (N,) predicted class indices.
        all_labels : (N,) true class indices.
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        ids = batch["ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"]

        output = model(ids, lengths)
        preds = output["logits"].argmax(dim=-1).cpu()

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def get_predictions_bert(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BERT model inference.

    Returns:
        all_preds  : (N,) predicted class indices.
        all_labels : (N,) true class indices.
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        output = model(input_ids, attention_mask)
        preds = output["logits"].argmax(dim=-1).cpu()

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Print classification report and return summary metrics.

    Returns:
        dict with 'accuracy' and 'macro_f1'.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print(f"\n{'=' * 60}")
    print(f"  {model_name} Evaluation")
    print(f"{'=' * 60}")
    print(classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
    ))
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"{'=' * 60}")

    return {"accuracy": acc, "macro_f1": f1}


def evaluate_bilstm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    preds, labels = get_predictions_bilstm(model, loader, device)
    return compute_metrics(preds, labels, class_names, model_name="BiLSTM + Attention")


def evaluate_bert(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    preds, labels = get_predictions_bert(model, loader, device)
    return compute_metrics(preds, labels, class_names, model_name="BERT Fine-tuned")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a normalised confusion matrix."""
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and accuracy curves from a training history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Training curves saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison table.

    Args:
        results: {model_name: {"accuracy": ..., "macro_f1": ...}}
    """
    header = f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<30} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f}"
        )
    print("=" * len(header))

    # Compute misclassification rate reduction vs TF-IDF
    baseline_key = next((k for k in results if "TF-IDF" in k or "Baseline" in k), None)
    if baseline_key:
        base_err = 1.0 - results[baseline_key]["accuracy"]
        print(f"\nMisclassification rate vs {baseline_key}:")
        for name, m in results.items():
            if name == baseline_key:
                continue
            err = 1.0 - m["accuracy"]
            reduction = (base_err - err) / base_err * 100
            print(f"  {name:<28}: {reduction:+.1f}%")
