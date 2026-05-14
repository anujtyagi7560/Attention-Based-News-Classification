# Attention-Based News Classification

Multi-category news classification using:
- **Custom BiLSTM + Self-Attention** model
- **Fine-tuned BERT** via Hugging Face Transformers
- **TF-IDF Baseline** for comparison

Evaluated on the [AG News](https://huggingface.co/datasets/ag_news) dataset (4 categories: World, Sports, Business, Sci/Tech).

---

## Results Summary

| Model              | Macro F1 | Accuracy |
|--------------------|----------|----------|
| TF-IDF + LogReg    | ~0.76    | ~76%     |
| BiLSTM + Attention | ~0.94    | ~94%     |
| BERT Fine-tuned    | ~0.96    | ~96%     |

---

## Project Structure

```
news-classifier/
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── data/
│   ├── dataset.py               # Dataset classes (PyTorch)
│   └── preprocessing.py         # Tokenization, vocab, cleaning
├── models/
│   ├── attention.py             # Self-attention mechanism
│   ├── bilstm_attention.py      # BiLSTM + Attention classifier
│   ├── bert_classifier.py       # BERT fine-tuning wrapper
│   └── tfidf_baseline.py        # TF-IDF + LogReg baseline
├── training/
│   ├── trainer.py               # Generic training loop
│   └── bert_trainer.py          # HuggingFace Trainer wrapper
├── evaluation/
│   └── evaluator.py             # Metrics, confusion matrix, reports
├── utils/
│   ├── logger.py                # Logging setup
│   └── helpers.py               # Seed, device, checkpoint utils
├── scripts/
│   ├── train_bilstm.py          # Train BiLSTM+Attention
│   ├── train_bert.py            # Fine-tune BERT
│   ├── train_baseline.py        # Train TF-IDF baseline
│   └── compare_models.py        # Compare all models
├── tests/
│   ├── test_attention.py
│   ├── test_dataset.py
│   └── test_models.py
├── notebooks/
│   └── exploration.ipynb        # EDA & results visualization
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train TF-IDF baseline
python scripts/train_baseline.py

# 3. Train BiLSTM + Attention
python scripts/train_bilstm.py

# 4. Fine-tune BERT
python scripts/train_bert.py

# 5. Compare all models
python scripts/compare_models.py
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`. No hardcoded values anywhere in the codebase.
