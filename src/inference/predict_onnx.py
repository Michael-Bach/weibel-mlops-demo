# src/inference/predict_onnx.py
"""
ONNX runtime inference — runs RadarClassifier without a PyTorch dependency.
Suitable for edge deployment or any environment where PyTorch is too heavy.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPredictor:
    def __init__(self, model_path: str = "artifacts/model.onnx"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            signals: float32 array of shape [batch_size, signal_length] or [signal_length]
        Returns:
            labels:      int array [batch_size]   — 0=clutter, 1=target
            confidences: float array [batch_size] — probability of predicted class
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        signals = signals.astype(np.float32)

        logits = self.session.run(None, {self.input_name: signals})[0]

        # Manual softmax — avoids scipy dependency
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        labels = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        return labels, confidences


def predict_single(
    signal: np.ndarray, model_path: str = "artifacts/model.onnx"
) -> tuple[int, float]:
    """Convenience wrapper — returns (label, confidence) for one signal."""
    predictor = OnnxPredictor(model_path)
    labels, confidences = predictor.predict(signal)
    return int(labels[0]), float(confidences[0])


if __name__ == "__main__":
    import yaml

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    signal_length = params["data"]["signal_length"]

    predictor = OnnxPredictor()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((4, signal_length)).astype(np.float32)
    labels, confidences = predictor.predict(batch)

    for i, (label, conf) in enumerate(zip(labels, confidences)):
        class_name = "target" if label == 1 else "clutter"
        print(f"Sample {i}: {class_name} ({conf:.3f})")
