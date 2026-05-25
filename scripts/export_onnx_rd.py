"""
Export the CNN+LSTM Range-Doppler model to ONNX.

Uses legacy TorchScript exporter (dynamo=False) so the Python for-loop over T
scans is fully unrolled into T replicated ONNX op-graphs — no ONNX LSTM operator,
no GPU required at runtime.

Input  shape: (batch_size, T=8, 1, 64, 128)
Output shape: (batch_size, 1)  — sigmoid P(target present)
"""

import hashlib
from pathlib import Path

import mlflow
import onnx
import torch
import yaml

from src.data.generator import N_DOPPLER, N_RANGE, T
from src.model.cnn_lstm import build_model


def export(params_path: str = "params.yaml") -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)

    mp = params.get("rd_model", {})
    lstm_hidden = mp.get("lstm_hidden", 64)
    opset = params.get("onnx", {}).get("opset_version", 17)

    artifacts = Path("artifacts")
    model_path = artifacts / "model_best.pt"
    onnx_path = artifacts / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint at {model_path} — run src/train.py first")

    model = build_model(lstm_hidden=lstm_hidden)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Fixed T=8: Python for-loop unrolls to T replicated op-graphs in ONNX
    dummy = torch.randn(1, T, 1, N_RANGE, N_DOPPLER)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=opset,
        input_names=["sequence"],
        output_names=["prob"],
        dynamic_axes={
            "sequence": {0: "batch_size"},
            "prob": {0: "batch_size"},
        },
        dynamo=False,  # force TorchScript path — avoids dynamo Split/opset issue
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

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
        pass

    print(f"Exported ONNX model → {onnx_path}")
    print(f"ONNX spec check: passed | opset {opset} | SHA-256: {onnx_sha}")
    print(f"Input:  sequence — shape [batch_size, {T}, 1, {N_RANGE}, {N_DOPPLER}]")
    print("Output: prob     — shape [batch_size, 1]")


if __name__ == "__main__":
    export()
