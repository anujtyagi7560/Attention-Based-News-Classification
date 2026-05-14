# tests/test_models.py
"""
Forward-pass smoke tests for all three classifiers.
These tests confirm shapes and types without requiring the full dataset.
"""

import pytest
import torch
from models.bilstm_attention import BiLSTMAttentionClassifier
from models.tfidf_baseline import TFIDFBaseline


BATCH = 4
SEQ_LEN = 32
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 4


class TestBiLSTMAttentionClassifier:
    @pytest.fixture
    def model(self):
        return BiLSTMAttentionClassifier(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            num_layers=2,
            dropout=0.1,
            pad_idx=0,
        )

    def test_logits_shape(self, model):
        ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN)
        out = model(ids, lengths)
        assert out["logits"].shape == (BATCH, NUM_CLASSES)

    def test_attn_weights_shape(self, model):
        ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN)
        out = model(ids, lengths)
        assert out["attn_weights"].shape == (BATCH, SEQ_LEN)

    def test_variable_lengths(self, model):
        ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        lengths = torch.tensor([32, 20, 15, 10])
        out = model(ids, lengths)
        assert out["logits"].shape == (BATCH, NUM_CLASSES)

    def test_no_nan_in_logits(self, model):
        ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN)
        out = model(ids, lengths)
        assert not torch.isnan(out["logits"]).any()

    def test_parameter_count_nonzero(self, model):
        assert model.count_parameters() > 0

    def test_eval_mode_no_dropout(self, model):
        """In eval mode, two identical forward passes should give identical output."""
        model.eval()
        ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN)
        out1 = model(ids, lengths)
        out2 = model(ids, lengths)
        assert torch.allclose(out1["logits"], out2["logits"])


class TestTFIDFBaseline:
    @pytest.fixture
    def trained_baseline(self):
        texts = [
            "The president signed new legislation today",
            "Manchester United beat Arsenal in the cup final",
            "Tech stocks rallied after the Fed rate decision",
            "Scientists discover a new species in the Amazon",
        ] * 10   # repeat to give the model enough samples
        labels = [0, 1, 2, 3] * 10
        baseline = TFIDFBaseline(max_features=500, ngram_range=(1, 1))
        baseline.fit(texts, labels)
        return baseline

    def test_predict_shape(self, trained_baseline):
        texts = ["breaking news today", "football match result"]
        preds = trained_baseline.predict(texts)
        assert len(preds) == 2

    def test_predict_proba_shape(self, trained_baseline):
        texts = ["stock market news"]
        proba = trained_baseline.predict_proba(texts)
        assert proba.shape == (1, 4)
        import numpy as np
        assert abs(proba.sum() - 1.0) < 1e-5
