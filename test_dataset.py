# tests/test_dataset.py
"""Unit tests for data/preprocessing.py and data/dataset.py"""

import pytest
import torch
from data.preprocessing import clean_text, tokenize, Vocabulary, pad_sequence


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello World") == "hello world"

    def test_remove_url(self):
        result = clean_text("Visit https://example.com for more")
        assert "http" not in result
        assert "example" not in result

    def test_remove_punctuation(self):
        result = clean_text("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapse_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result


class TestVocabulary:
    @pytest.fixture
    def vocab(self):
        corpus = [["hello", "world"], ["hello", "foo"], ["bar", "baz", "foo"]]
        v = Vocabulary(max_size=100, min_freq=1)
        v.build(corpus)
        return v

    def test_special_tokens_at_fixed_ids(self, vocab):
        assert vocab.pad_idx == 0
        assert vocab.unk_idx == 1

    def test_known_tokens_encoded(self, vocab):
        ids = vocab.encode(["hello", "world"])
        assert all(i > 1 for i in ids)     # not PAD or UNK

    def test_unknown_token_maps_to_unk(self, vocab):
        ids = vocab.encode(["<unknown_token_xyz>"])
        assert ids[0] == vocab.unk_idx

    def test_min_freq_filtering(self):
        corpus = [["rare"], ["common", "common"]]
        v = Vocabulary(min_freq=2).build(corpus)
        assert "rare" not in v.token2idx
        assert "common" in v.token2idx

    def test_serialisation_roundtrip(self, vocab):
        state = vocab.state_dict()
        vocab2 = Vocabulary.from_state_dict(state)
        assert vocab.encode(["hello"]) == vocab2.encode(["hello"])
        assert len(vocab) == len(vocab2)


class TestPadSequence:
    def test_output_shape(self):
        seqs = [[1, 2, 3], [4, 5], [6]]
        padded, lengths = pad_sequence(seqs, pad_idx=0)
        assert padded.shape == (3, 3)
        assert lengths.tolist() == [3, 2, 1]

    def test_padding_values(self):
        seqs = [[1, 2], [3]]
        padded, _ = pad_sequence(seqs, pad_idx=99)
        assert padded[1, 1].item() == 99

    def test_max_len_truncation(self):
        seqs = [[1, 2, 3, 4, 5]]
        padded, lengths = pad_sequence(seqs, pad_idx=0, max_len=3)
        assert padded.shape == (1, 3)
        assert lengths[0].item() == 3
