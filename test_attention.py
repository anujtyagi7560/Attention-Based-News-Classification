# tests/test_attention.py
"""Unit tests for models/attention.py"""

import pytest
import torch
from models.attention import AdditiveAttention, ScaledDotAttention, make_padding_mask


BATCH, SEQ_LEN, DIM = 4, 20, 64


class TestMakePaddingMask:
    def test_shape(self):
        lengths = torch.tensor([10, 15, 20, 5])
        mask = make_padding_mask(lengths, max_len=SEQ_LEN)
        assert mask.shape == (BATCH, SEQ_LEN)

    def test_padded_positions_are_true(self):
        lengths = torch.tensor([5, 10])
        mask = make_padding_mask(lengths, max_len=10)
        # First sample: positions 5..9 should be True (padding)
        assert mask[0, 4] == False   # last real token
        assert mask[0, 5] == True    # first padding
        # Second sample: no padding
        assert mask[1, 9] == False


class TestAdditiveAttention:
    @pytest.fixture
    def attn(self):
        return AdditiveAttention(hidden_dim=DIM)

    def test_output_shapes(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        context, weights = attn(hidden)
        assert context.shape == (BATCH, DIM)
        assert weights.shape == (BATCH, SEQ_LEN)

    def test_weights_sum_to_one_without_mask(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        _, weights = attn(hidden)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_masked_positions_have_zero_weight(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        lengths = torch.tensor([5, 10, 15, 20])
        mask = make_padding_mask(lengths, max_len=SEQ_LEN)
        _, weights = attn(hidden, mask=mask)
        # Padding positions should have ~0 weight
        for i, length in enumerate(lengths.tolist()):
            pad_weights = weights[i, length:]
            assert pad_weights.abs().max().item() < 1e-5

    def test_no_nan_in_output(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        context, weights = attn(hidden)
        assert not torch.isnan(context).any()
        assert not torch.isnan(weights).any()


class TestScaledDotAttention:
    @pytest.fixture
    def attn(self):
        return ScaledDotAttention(hidden_dim=DIM)

    def test_output_shapes(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        context, weights = attn(hidden)
        assert context.shape == (BATCH, DIM)
        assert weights.shape == (BATCH, SEQ_LEN)

    def test_weights_sum_to_one(self, attn):
        hidden = torch.randn(BATCH, SEQ_LEN, DIM)
        _, weights = attn(hidden)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)
