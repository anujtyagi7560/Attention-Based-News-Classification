# data/preprocessing.py
"""
Text preprocessing, tokenization, and vocabulary building.
All logic is stateless (functions) or encapsulated in Vocabulary class.
"""

import re
import string
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs, strip punctuation and extra whitespace.
    Keeps alphanumeric characters and spaces.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"<.*?>", "", text)                     # strip HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Maps tokens to integer IDs and back.
    Supports <PAD> and <UNK> special tokens.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        max_size: int = 50_000,
        min_freq: int = 2,
    ) -> None:
        self.max_size = max_size
        self.min_freq = min_freq

        # Reserved IDs
        self.token2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}
        self._built = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, corpus: List[List[str]]) -> "Vocabulary":
        """
        Build vocabulary from a list of tokenized sentences.

        Args:
            corpus: List of token lists.

        Returns:
            self (for chaining)
        """
        counter: Counter = Counter()
        for tokens in corpus:
            counter.update(tokens)

        # Filter by min frequency, cap at max_size - 2 (reserved tokens)
        most_common = [
            token for token, freq in counter.most_common(self.max_size - 2)
            if freq >= self.min_freq
        ]

        for token in most_common:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        self._built = True
        return self

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert token list to ID list (unknown tokens → UNK)."""
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert ID list back to tokens."""
        return [self.idx2token.get(i, self.UNK_TOKEN) for i in ids]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token,
            "max_size": self.max_size,
            "min_freq": self.min_freq,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "Vocabulary":
        vocab = cls(max_size=state["max_size"], min_freq=state["min_freq"])
        vocab.token2idx = state["token2idx"]
        vocab.idx2token = {int(k): v for k, v in state["idx2token"].items()}
        vocab._built = True
        return vocab


# ---------------------------------------------------------------------------
# Padding / Collation helpers
# ---------------------------------------------------------------------------

def pad_sequence(
    sequences: List[List[int]],
    pad_idx: int,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of integer sequences to the same length.

    Args:
        sequences: List of variable-length ID lists.
        pad_idx:   Padding token index.
        max_len:   Force a fixed length; defaults to longest sequence.

    Returns:
        padded: (B, L) LongTensor
        lengths: (B,) LongTensor with original lengths (capped at max_len)
    """
    lengths = [len(s) for s in sequences]
    if max_len is None:
        max_len = max(lengths)

    padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        actual = min(length, max_len)
        padded[i, :actual] = torch.tensor(seq[:actual], dtype=torch.long)

    capped_lengths = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long)
    return padded, capped_lengths
