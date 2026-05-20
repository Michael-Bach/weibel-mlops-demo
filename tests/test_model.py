# tests/test_model.py
"""
Tests for the RadarClassifier architecture.
Verifies forward-pass contracts: input/output shapes and logit semantics.
No training — instantiate, eval, forward only.
"""

import pytest
import torch

from src.models.classifier import RadarClassifier, build_model

INPUT_DIM = 64
BATCH_SIZE = 8


@pytest.fixture
def model():
    m = RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32, 16], dropout=0.0)
    m.eval()
    return m


def test_forward_batch_output_shape(model):
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 2)


def test_forward_single_sample_shape(model):
    x = torch.randn(1, INPUT_DIM)
    out = model(x)
    assert out.shape == (1, 2)


def test_output_is_raw_logits(model):
    # Logits are not bounded to [0,1]; softmax probabilities must sum to 1
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    out = model(x)
    probs = torch.softmax(out, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


def test_build_model_matches_params():
    # build_model reads params.yaml — runs from repo root in CI and locally
    model = build_model("params.yaml")
    assert isinstance(model, RadarClassifier)
    # params.yaml sets signal_length=128; model input dim must match
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == (2, 2)


def test_eval_mode_is_deterministic(model):
    x = torch.randn(4, INPUT_DIM)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
