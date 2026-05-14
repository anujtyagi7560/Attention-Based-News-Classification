#!/usr/bin/env python3
# scripts/train_baseline.py
"""
Train the TF-IDF + Logistic Regression baseline on AG News.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset

from utils.helpers import load_config, set_seed
from utils.logger import get_logger, setup_logging
from models.tfidf_baseline import build_tfidf_baseline

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(log_file=os.path.join(cfg["paths"]["logs_dir"], "baseline_train.log"))
    set_seed(cfg["data"]["random_seed"])

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading AG News dataset …")
    raw = load_dataset("ag_news", cache_dir=cfg["paths"]["data_dir"])
    train_split = raw["train"]
    test_split = raw["test"]

    train_texts = train_split["text"]
    train_labels = train_split["label"]
    test_texts = test_split["text"]
    test_labels = test_split["label"]

    # -----------------------------------------------------------------------
    # 2. Build & train baseline
    # -----------------------------------------------------------------------
    logger.info("Fitting TF-IDF + Logistic Regression …")
    baseline = build_tfidf_baseline(cfg)
    baseline.fit(train_texts, train_labels)
    logger.info("Training complete.")

    # -----------------------------------------------------------------------
    # 3. Evaluate
    # -----------------------------------------------------------------------
    class_names = cfg["data"]["class_names"]
    metrics = baseline.evaluate(test_texts, test_labels, class_names=class_names)

    # -----------------------------------------------------------------------
    # 4. Save model + results
    # -----------------------------------------------------------------------
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "tfidf_baseline.joblib")
    baseline.save(ckpt_path)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "baseline_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Baseline — Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
