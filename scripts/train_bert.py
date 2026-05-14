#!/usr/bin/env python3
# scripts/train_bert.py
"""
Fine-tune BERT for AG News classification.

Usage:
    python scripts/train_bert.py
    python scripts/train_bert.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import BertTokenizerFast

from utils.helpers import load_config, set_seed, get_device
from utils.logger import get_logger, setup_logging
from data.dataset import load_ag_news, get_bert_dataloaders
from models.bert_classifier import build_bert_model
from training.bert_trainer import BERTTrainer
from evaluation.evaluator import evaluate_bert

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on AG News")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(log_file=os.path.join(cfg["paths"]["logs_dir"], "bert_train.log"))
    set_seed(cfg["data"]["random_seed"])
    device = get_device()

    # -----------------------------------------------------------------------
    # 1. Tokenizer & DataLoaders
    # -----------------------------------------------------------------------
    model_name = cfg["bert"]["model_name"]
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    logger.info("Creating DataLoaders …")
    train_loader, val_loader, test_loader = get_bert_dataloaders(cfg, tokenizer)

    # -----------------------------------------------------------------------
    # 2. Build model
    # -----------------------------------------------------------------------
    logger.info("Building BERT model …")
    model = build_bert_model(cfg)
    logger.info(f"BERT parameters: {model.count_parameters():,}")

    # -----------------------------------------------------------------------
    # 3. Fine-tune
    # -----------------------------------------------------------------------
    bert_output_dir = os.path.join(cfg["paths"]["checkpoint_dir"], "bert")
    trainer = BERTTrainer(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=bert_output_dir,
    )

    train_metrics = trainer.train(
        train_dataset=train_loader.dataset,
        val_dataset=val_loader.dataset,
    )

    # -----------------------------------------------------------------------
    # 4. Evaluate on test set
    # -----------------------------------------------------------------------
    logger.info("Evaluating on test set …")
    model = model.to(device)
    class_names = cfg["data"]["class_names"]
    metrics = evaluate_bert(model, test_loader, device, class_names=class_names)

    # -----------------------------------------------------------------------
    # 5. Save results
    # -----------------------------------------------------------------------
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "bert_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("BERT fine-tuning complete.")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
