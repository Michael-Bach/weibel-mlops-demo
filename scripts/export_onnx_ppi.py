"""
Export trained PPIDetectorCNN to a self-contained ONNX file.

Usage:
    PYTHONPATH=. python scripts/export_onnx_ppi.py
"""

from pathlib import Path

import torch
import yaml
import onnx
import onnxruntime as ort

from src.model.ppi_cnn import build_model


def export(params_path: str = "params_ppi.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)

    ckpt_path = Path("artifacts/ppi_model_best.pt")
    onnx_path = Path(p["onnx"]["model_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run src/train_ppi.py first")

    n_sw    = p["radar"]["n_sweeps"]
    n_az    = p["radar"]["n_azimuths"]
    n_range = p["radar"]["n_ranges"]

    base = build_model(params_path)
    base.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    base.eval()

    # Wrap with sigmoid so the ONNX model outputs probabilities directly
    import torch.nn as nn
    class _WithSigmoid(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return torch.sigmoid(self.m(x))
    model = _WithSigmoid(base)

    dummy = torch.randn(1, 3, n_az, n_range)   # 3 temporal channels (max/mean/std)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        dynamo=False,
        opset_version=p["onnx"]["opset_version"],
        input_names=["ppi_sequence"],
        output_names=["prob_map"],
        dynamic_axes={
            "ppi_sequence": {0: "batch"},
            "prob_map":     {0: "batch"},
        },
    )

    # Spec check
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Numerical smoke-test
    sess = ort.InferenceSession(str(onnx_path))
    x_np = dummy.numpy()
    out  = sess.run(None, {"ppi_sequence": x_np})[0]
    assert out.shape == (1, n_az, n_range), f"Unexpected shape {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0

    print(f"Exported  : {onnx_path}  ({onnx_path.stat().st_size // 1024} KB)")
    print(f"Input     : ppi_sequence  ({n_sw}, {n_az}, {n_range})")
    print(f"Output    : prob_map      ({n_az}, {n_range})")
    print("ONNX spec : passed")


if __name__ == "__main__":
    export()
