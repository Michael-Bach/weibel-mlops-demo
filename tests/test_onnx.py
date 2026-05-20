# tests/test_onnx.py
"""
Tests for ONNX inference.
Exports a minimal model to a temp file — no dependency on artifacts/model.onnx,
so this runs cleanly in CI before the training step.
"""

import numpy as np
import pytest
import torch

from src.inference.predict_onnx import OnnxPredictor
from src.models.classifier import RadarClassifier

INPUT_DIM = 64
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory):
    """Export a minimal RadarClassifier to a temp ONNX file."""
    tmp = tmp_path_factory.mktemp("onnx")
    model = RadarClassifier(input_dim=INPUT_DIM, hidden_dims=[32], dropout=0.0)
    model.eval()

    onnx_path = str(tmp / "test_model.onnx")
    dummy = torch.randn(1, INPUT_DIM)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={"signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return onnx_path


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
