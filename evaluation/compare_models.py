#!/usr/bin/env python3
# scripts/compare_models.py
"""
Load saved results from all three models and print a comparison table.

Assumes the following JSON result files exist:
  - results/baseline_test_metrics.json
  - results/bilstm_test_metrics.json
  - results/bert_test_metrics.json

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --results_dir results/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluator import compare_models
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_FILES = {
    "TF-IDF Baseline":        "baseline_test_metrics.json",
    "BiLSTM + Attention":     "bilstm_test_metrics.json",
    "BERT Fine-tuned":        "bert_test_metrics.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Result file not found: {path} — skipping {model_name}")
            continue
        with open(path) as f:
            results[model_name] = json.load(f)

    if not results:
        logger.error("No result files found. Run the training scripts first.")
        sys.exit(1)

    compare_models(results)


if __name__ == "__main__":
    main()
