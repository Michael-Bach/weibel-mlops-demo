# tests/test_onnx.py
"""
Tests for ONNX inference.
Exports minimal models to temp files — no dependency on artifacts/model.onnx,
so this runs cleanly in CI before the training step.
"""

import numpy as np
import pytest
import torch

from src.inference.validate_onnx import OnnxPredictor
from src.models.classifier import RadarClassifier

INPUT_DIM = 64
BATCH_SIZE = 4


def _export(model: RadarClassifier, tmp_path_factory, name: str) -> str:
    tmp = tmp_path_factory.mktemp(name)
    model.eval()
    onnx_path = str(tmp / "test_model.onnx")
    torch.onnx.export(
        model,
        torch.randn(1, INPUT_DIM),
        onnx_path,
        opset_version=17,
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={"signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return onnx_path


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory):
    """Minimal model without FFT — no-FFT path."""
    return _export(
        RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32], dropout=0.0),
        tmp_path_factory,
        "onnx_no_fft",
    )


@pytest.fixture(scope="module")
def onnx_fft_model_path(tmp_path_factory):
    """Minimal model with FFT — covers the production code path."""
    return _export(
        RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32], dropout=0.0, use_fft=True),
        tmp_path_factory,
        "onnx_fft",
    )


# ── Basic interface tests (no-FFT model) ──────────────────────────────────────

def test_predictor_loads(onnx_model_path):
    predictor = OnnxPredictor(onnx_model_path)
    assert predictor.session is not None


def test_predict_batch_output_shapes(onnx_model_path):
    predictor = OnnxPredictor(onnx_model_path)
    signals = np.random.randn(BATCH_SIZE, INPUT_DIM).astype(np.float32)
    labels, confidences = predictor.predict(signals)
    assert labels.shape == (BATCH_SIZE,)
    assert confidences.shape == (BATCH_SIZE,)


def test_predict_1d_input_broadcasts(onnx_model_path):
    """Single flat signal should be treated as batch of 1."""
    predictor = OnnxPredictor(onnx_model_path)
    signal = np.random.randn(INPUT_DIM).astype(np.float32)
    labels, confidences = predictor.predict(signal)
    assert labels.shape == (1,)
    assert confidences.shape == (1,)


def test_labels_are_binary(onnx_model_path):
    predictor = OnnxPredictor(onnx_model_path)
    signals = np.random.randn(20, INPUT_DIM).astype(np.float32)
    labels, _ = predictor.predict(signals)
    assert set(labels.tolist()).issubset({0, 1})


def test_confidences_in_unit_interval(onnx_model_path):
    predictor = OnnxPredictor(onnx_model_path)
    signals = np.random.randn(20, INPUT_DIM).astype(np.float32)
    _, confidences = predictor.predict(signals)
    assert (confidences >= 0.0).all()
    assert (confidences <= 1.0).all()


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxPredictor(str(tmp_path / "nonexistent.onnx"))


# ── FFT model export and numerical equivalence ────────────────────────────────

def test_fft_model_onnx_export_loads(onnx_fft_model_path):
    """FFT model exports without error and loads into onnxruntime."""
    predictor = OnnxPredictor(onnx_fft_model_path)
    assert predictor.session is not None


def test_fft_model_output_shapes(onnx_fft_model_path):
    predictor = OnnxPredictor(onnx_fft_model_path)
    signals = np.random.randn(BATCH_SIZE, INPUT_DIM).astype(np.float32)
    labels, confidences = predictor.predict(signals)
    assert labels.shape == (BATCH_SIZE,)
    assert confidences.shape == (BATCH_SIZE,)


def test_onnx_matches_pytorch_no_fft(onnx_model_path):
    """ONNX and PyTorch must produce identical logits for the no-FFT model."""
    torch.manual_seed(0)
    model = RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32], dropout=0.0)
    model.eval()

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        pt_logits = model(x_pt).numpy()

    # Re-export with the exact same weights used above
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = f.name
    try:
        torch.onnx.export(
            model, x_pt[:1], tmp,
            opset_version=17,
            input_names=["signal"], output_names=["logits"],
            dynamic_axes={"signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        import onnxruntime as ort
        sess = ort.InferenceSession(tmp)
        ort_logits = sess.run(None, {"signal": x_np})[0]
    finally:
        os.unlink(tmp)

    np.testing.assert_allclose(pt_logits, ort_logits, atol=1e-4,
                               err_msg="PyTorch and ONNX logits diverge (no-FFT model)")


def test_onnx_matches_pytorch_fft(onnx_fft_model_path):
    """ONNX and PyTorch must produce identical logits for the FFT model."""
    torch.manual_seed(1)
    model = RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32], dropout=0.0, use_fft=True)
    model.eval()

    rng = np.random.default_rng(43)
    x_np = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        pt_logits = model(x_pt).numpy()

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = f.name
    try:
        torch.onnx.export(
            model, x_pt[:1], tmp,
            opset_version=17,
            input_names=["signal"], output_names=["logits"],
            dynamic_axes={"signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        import onnxruntime as ort
        sess = ort.InferenceSession(tmp)
        ort_logits = sess.run(None, {"signal": x_np})[0]
    finally:
        os.unlink(tmp)

    np.testing.assert_allclose(pt_logits, ort_logits, atol=1e-4,
                               err_msg="PyTorch and ONNX logits diverge (FFT model)")
