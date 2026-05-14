# data/dataset.py
"""
PyTorch Dataset classes for AG News.

- AGNewsDataset      : for BiLSTM + Attention (uses custom Vocabulary)
- AGNewsBERTDataset  : for BERT (uses HuggingFace tokenizer)

Both datasets are loaded from HuggingFace `datasets` library.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from data.preprocessing import Vocabulary, tokenize, pad_sequence


# ---------------------------------------------------------------------------
# AG News label mapping  (HuggingFace uses 0-indexed)
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# ---------------------------------------------------------------------------
# Helper: load raw HuggingFace splits
# ---------------------------------------------------------------------------

def load_ag_news(cache_dir: Optional[str] = None) -> DatasetDict:
    """Download / load AG News from HuggingFace Hub."""
    return load_dataset("ag_news", cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Dataset for BiLSTM + Attention
# ---------------------------------------------------------------------------

class AGNewsDataset(Dataset):
    """
    Returns tokenized + encoded integer sequences for the custom model.

    Args:
        hf_split  : A HuggingFace dataset split (train / test).
        vocab     : A built Vocabulary instance.
        max_len   : Maximum sequence length (longer sequences truncated).
    """

    def __init__(
        self,
        hf_split,
        vocab: Vocabulary,
        max_len: int = 128,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len

        self.texts: List[str] = hf_split["text"]
        self.labels: List[int] = hf_split["label"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = tokenize(self.texts[idx])
        ids = self.vocab.encode(tokens)[: self.max_len]
        length = len(ids)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Collate function (passed to DataLoader)
    # ------------------------------------------------------------------

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        ids = [item["ids"] for item in batch]
        lengths = [item["length"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        padded, _ = pad_sequence(
            [i.tolist() for i in ids],
            pad_idx=self.vocab.pad_idx,
            max_len=self.max_len,
        )
        lengths_tensor = torch.stack(lengths)

        return {"ids": padded, "lengths": lengths_tensor, "labels": labels}


# ---------------------------------------------------------------------------
# Dataset for BERT
# ---------------------------------------------------------------------------

class AGNewsBERTDataset(Dataset):
    """
    Tokenizes with a HuggingFace tokenizer and returns input_ids,
    attention_mask, and labels ready for BERT.

    Args:
        hf_split   : A HuggingFace dataset split.
        tokenizer  : Any PreTrainedTokenizer (e.g. BertTokenizerFast).
        max_len    : Maximum subword token length.
    """

    def __init__(
        self,
        hf_split,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts: List[str] = hf_split["text"]
        self.labels: List[int] = hf_split["label"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_vocab_from_split(hf_split, cfg: dict) -> Vocabulary:
    """
    Build a Vocabulary from the training split using config values.

    Args:
        hf_split : HuggingFace train split.
        cfg      : vocab section of config dict.

    Returns:
        Built Vocabulary instance.
    """
    corpus = [tokenize(text) for text in hf_split["text"]]
    vocab = Vocabulary(
        max_size=cfg["max_vocab_size"],
        min_freq=cfg["min_freq"],
    )
    vocab.build(corpus)
    return vocab


def get_bilstm_dataloaders(
    cfg: dict,
    vocab: Vocabulary,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for the BiLSTM model.
    A portion of the train split is used for validation.
    """
    raw = load_ag_news(cache_dir=cfg["paths"]["data_dir"])

    # Split train → train + val
    train_val = raw["train"].train_test_split(
        test_size=cfg["data"]["test_size"],
        seed=cfg["data"]["random_seed"],
    )
    train_split = train_val["train"]
    val_split = train_val["test"]
    test_split = raw["test"]

    max_len = cfg["data"]["max_seq_len"]
    batch_size = cfg["bilstm"]["batch_size"]

    train_ds = AGNewsDataset(train_split, vocab, max_len)
    val_ds = AGNewsDataset(val_split, vocab, max_len)
    test_ds = AGNewsDataset(test_split, vocab, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader


def get_bert_dataloaders(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for BERT fine-tuning."""
    raw = load_ag_news(cache_dir=cfg["paths"]["data_dir"])

    train_val = raw["train"].train_test_split(
        test_size=cfg["data"]["test_size"],
        seed=cfg["data"]["random_seed"],
    )

    max_len = cfg["bert"]["max_seq_len"]
    batch_size = cfg["bert"]["batch_size"]

    train_ds = AGNewsBERTDataset(train_val["train"], tokenizer, max_len)
    val_ds = AGNewsBERTDataset(train_val["test"], tokenizer, max_len)
    test_ds = AGNewsBERTDataset(raw["test"], tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
