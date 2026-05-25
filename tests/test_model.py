# tests/test_model.py
"""
Tests for the CNN+LSTM Range-Doppler classifier.
Forward-pass contracts: shapes, output range, determinism. No training.
"""

import torch
import pytest

from src.data.generator import T, N_RANGE, N_DOPPLER
from src.model.cnn_lstm import CnnLstmClassifier, build_model

BATCH = 4


@pytest.fixture
def model() -> CnnLstmClassifier:
    m = build_model(lstm_hidden=64)
    m.eval()
    return m


def _dummy(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, T, 1, N_RANGE, N_DOPPLER)


def test_forward_output_shape(model):
    out = model(_dummy())
    assert out.shape == (BATCH, 1)


def test_forward_single_sample(model):
    out = model(_dummy(1))
    assert out.shape == (1, 1)


def test_output_is_probability(model):
    """Sigmoid output must be in [0, 1]."""
    out = model(_dummy())
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_eval_is_deterministic(model):
    x = _dummy()
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_build_model_returns_correct_type():
    m = build_model()
    assert isinstance(m, CnnLstmClassifier)


def test_input_shape_matches_spec(model):
    """Model must accept exactly (batch, T=8, 1, 64, 128)."""
    x = torch.randn(2, 8, 1, 64, 128)
    out = model(x)
    assert out.shape == (2, 1)


def test_different_lstm_hidden():
    m = build_model(lstm_hidden=32)
    m.eval()
    out = m(_dummy(2))
    assert out.shape == (2, 1)
