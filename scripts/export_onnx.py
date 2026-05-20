# scripts/export_onnx.py
"""
Export trained RadarClassifier to ONNX format.

ONNX is the standard interchange format for edge deployment —
decouples the model from the PyTorch runtime entirely.
"""

from pathlib import Path

import torch
import yaml

from src.models.classifier import build_model


def export(params_path: str = "params.yaml") -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)

    artifacts = Path("artifacts")
    model_path = artifacts / "model_best.pt"
    onnx_path = artifacts / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {model_path} — run train.py first")

    model = build_model(params_path)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    signal_length = params["data"]["signal_length"]

    # Dummy input — ONNX tracing needs a concrete tensor to trace the graph
    dummy_input = torch.randn(1, signal_length)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=params["onnx"]["opset_version"],
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={
            "signal": {0: "batch_size"},   # variable batch size at inference
            "logits": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Input:  signal — shape [batch_size, {signal_length}]")
    print("Output: logits — shape [batch_size, 2]")


if __name__ == "__main__":
    export()