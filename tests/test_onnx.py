# tests/test_onnx.py
"""
Tests for ONNX export and inference.
Exports a minimal CNN+LSTM to a temp file — no dependency on artifacts/model.onnx,
so this runs cleanly in CI before the training step.
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.data.generator import N_DOPPLER, N_RANGE, T
from src.model.cnn_lstm import CnnLstmClassifier, build_model

BATCH = 4


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory) -> str:
    """Export a minimal untrained model to a temp ONNX file."""
    m = build_model(lstm_hidden=32)
    m.eval()
    dummy = torch.randn(1, T, 1, N_RANGE, N_DOPPLER)
    path = str(tmp_path_factory.mktemp("onnx") / "test_model.onnx")
    torch.onnx.export(
        m, dummy, path,
        opset_version=17,
        input_names=["sequence"],
        output_names=["prob"],
        dynamic_axes={"sequence": {0: "batch_size"}, "prob": {0: "batch_size"}},
        dynamo=False,
    )
    return path


def test_onnx_spec_valid(onnx_path):
    onnx.checker.check_model(onnx.load(onnx_path))


def test_onnx_session_loads(onnx_path):
    sess = ort.InferenceSession(onnx_path)
    assert sess is not None


def test_onnx_output_shape(onnx_path):
    sess = ort.InferenceSession(onnx_path)
    x = np.random.randn(BATCH, T, 1, N_RANGE, N_DOPPLER).astype(np.float32)
    out = sess.run(None, {"sequence": x})[0]
    assert out.shape == (BATCH, 1)


def test_onnx_single_sample(onnx_path):
    sess = ort.InferenceSession(onnx_path)
    x = np.random.randn(1, T, 1, N_RANGE, N_DOPPLER).astype(np.float32)
    out = sess.run(None, {"sequence": x})[0]
    assert out.shape == (1, 1)


def test_onnx_output_is_probability(onnx_path):
    sess = ort.InferenceSession(onnx_path)
    x = np.random.randn(10, T, 1, N_RANGE, N_DOPPLER).astype(np.float32)
    out = sess.run(None, {"sequence": x})[0]
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_onnx_matches_pytorch(onnx_path):
    """ONNX runtime must produce the same output as PyTorch."""
    torch.manual_seed(0)
    m = build_model(lstm_hidden=32)
    m.eval()

    x_np = np.random.default_rng(42).standard_normal(
        (BATCH, T, 1, N_RANGE, N_DOPPLER)
    ).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        pt_out = m(x_pt).numpy()

    # Re-export this exact model
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = f.name
    try:
        torch.onnx.export(
            m, x_pt[:1], tmp, opset_version=17,
            input_names=["sequence"], output_names=["prob"],
            dynamic_axes={"sequence": {0: "batch_size"}, "prob": {0: "batch_size"}},
            dynamo=False,
        )
        sess = ort.InferenceSession(tmp)
        ort_out = sess.run(None, {"sequence": x_np})[0]
    finally:
        os.unlink(tmp)

    np.testing.assert_allclose(pt_out, ort_out, atol=1e-4,
                               err_msg="PyTorch and ONNX outputs diverge")


def test_missing_model_raises(tmp_path):
    with pytest.raises(Exception):
        ort.InferenceSession(str(tmp_path / "nonexistent.onnx"))
