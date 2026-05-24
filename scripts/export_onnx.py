# scripts/export_onnx.py
"""
Export trained RadarClassifier to ONNX format.

ONNX is the standard interchange format for edge deployment —
decouples the model from the PyTorch runtime entirely.
"""

import hashlib
from pathlib import Path

import mlflow
import onnx
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

    # Validate the exported graph against the ONNX spec
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Log the ONNX artifact to the most recent MLflow run so there is a traceable
    # link between the training run and the deployed artifact.
    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
    try:
        runs = mlflow.search_runs(
            experiment_names=["weibel-mlops-demo"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(onnx_path))
                mlflow.set_tag("onnx_sha256", onnx_sha)
    except Exception:
        pass  # MLflow unavailable — export still succeeds

    print(f"Exported ONNX model to {onnx_path}")
    print("ONNX spec check: passed")
    print(f"SHA-256 (12): {onnx_sha}")
    print(f"Input:  signal — shape [batch_size, {signal_length}]")
    print("Output: logits — shape [batch_size, 2]")


if __name__ == "__main__":
    export()
