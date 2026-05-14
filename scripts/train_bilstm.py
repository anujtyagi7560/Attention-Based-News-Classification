#!/usr/bin/env python3
# scripts/train_bilstm.py
"""
End-to-end training script for the BiLSTM + Self-Attention classifier.

Usage:
    python scripts/train_bilstm.py
    python scripts/train_bilstm.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import load_config, set_seed, get_device
from utils.logger import get_logger, setup_logging
from data.dataset import load_ag_news, build_vocab_from_split, get_bilstm_dataloaders
from models.bilstm_attention import build_bilstm_model
from training.trainer import Trainer
from evaluation.evaluator import evaluate_bilstm, plot_confusion_matrix, plot_training_curves

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM + Attention classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(log_file=os.path.join(cfg["paths"]["logs_dir"], "bilstm_train.log"))
    set_seed(cfg["data"]["random_seed"])
    device = get_device()

    # -----------------------------------------------------------------------
    # 1. Load data & build vocabulary
    # -----------------------------------------------------------------------
    logger.info("Loading AG News dataset …")
    raw = load_ag_news(cache_dir=cfg["paths"]["data_dir"])

    # Build vocab from training split
    logger.info("Building vocabulary …")
    vocab = build_vocab_from_split(raw["train"], cfg["vocab"])
    logger.info(f"Vocabulary size: {len(vocab):,}")

    # Save vocab for later inference
    vocab_path = os.path.join(cfg["paths"]["checkpoint_dir"], "vocab.json")
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab.state_dict(), f)
    logger.info(f"Vocabulary saved to {vocab_path}")

    # -----------------------------------------------------------------------
    # 2. Create DataLoaders
    # -----------------------------------------------------------------------
    logger.info("Creating DataLoaders …")
    train_loader, val_loader, test_loader = get_bilstm_dataloaders(cfg, vocab)

    # -----------------------------------------------------------------------
    # 3. Build model
    # -----------------------------------------------------------------------
    model = build_bilstm_model(cfg, vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
    )
    history = trainer.train(train_loader, val_loader, model_name="bilstm")

    # -----------------------------------------------------------------------
    # 5. Evaluate on test set (using best checkpoint)
    # -----------------------------------------------------------------------
    trainer.load_best(model_name="bilstm")
    class_names = cfg["data"]["class_names"]

    metrics = evaluate_bilstm(model, test_loader, device, class_names=class_names)

    # -----------------------------------------------------------------------
    # 6. Save results & plots
    # -----------------------------------------------------------------------
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "bilstm_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(
        history,
        title="BiLSTM + Attention",
        save_path=os.path.join(results_dir, "bilstm_training_curves.png"),
    )

    logger.info("BiLSTM training complete.")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
