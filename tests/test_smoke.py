"""Smoke tests — verify core modules import and produce plausible outputs."""

import numpy as np
import pytest
import torch
import yaml


def _params():
    with open("params_ppi.yaml") as f:
        return yaml.safe_load(f)


# ── Data generation ───────────────────────────────────────────────────────────

def test_ppi_generator_target():
    from src.data.ppi_generator import make_sample

    rng = np.random.default_rng(0)
    features, label = make_sample(params=_params(), has_target=True, rng=rng, sample_seed=42)

    assert features.shape[0] == 3, "expected 3 temporal-feature channels"
    assert features.ndim == 3
    assert label.ndim == 2
    assert label.max() > 0, "label map should contain a positive response"


def test_ppi_generator_no_target():
    from src.data.ppi_generator import make_sample

    rng = np.random.default_rng(1)
    features, label = make_sample(params=_params(), has_target=False, rng=rng, sample_seed=43)

    assert label.max() == pytest.approx(0.0), "no-target label should be all zeros"


# ── Models — forward pass ─────────────────────────────────────────────────────

@pytest.fixture
def ppi_batch():
    """(B=2, 3, 180, 64) temporal-feature batch."""
    return torch.zeros(2, 3, 180, 64)


def test_cnn_forward(ppi_batch):
    from src.model.ppi_cnn import PPIDetectorCNN

    model = PPIDetectorCNN()
    model.eval()
    with torch.no_grad():
        out = model(ppi_batch)

    assert out.shape == (2, 180, 64)
    assert torch.isfinite(out).all()


def test_transformer_forward():
    from src.model.ppi_transformer import PPITransformerDetector

    model = PPITransformerDetector()
    model.eval()
    x = torch.zeros(2, 10, 180, 64)   # (B, N_sweeps, H, W)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 180, 64)
    assert torch.isfinite(out).all()


def test_conv_gru_forward():
    from src.model.conv_gru import ConvGRUDetector

    model = ConvGRUDetector()
    model.eval()
    sweep = torch.zeros(2, 1, 180, 64)
    h = torch.zeros(2, 1, 180, 64)
    with torch.no_grad():
        prob_map, h_out = model(sweep, h)

    assert prob_map.shape == h_out.shape == (2, 1, 180, 64)
    assert torch.isfinite(prob_map).all()
