"""
Export trained ConvGRUDetector to a single-step streaming ONNX model.

The hidden state h IS the detection confidence heatmap — single channel,
values in [0, 1], directly renderable as a probability map after every sweep.

    Inputs:
      sweep_norm : (1, 1, n_az, n_range)  EMA-normalised amplitude
      h_in       : (1, 1, n_az, n_range)  confidence heatmap from previous sweep

    Outputs:
      prob_map   : (1, 1, n_az, n_range)  updated confidence heatmap ∈ [0, 1]
      h_out      : (1, 1, n_az, n_range)  same tensor — pass back in next sweep

prob_map and h_out are identical: the hidden state IS the output heatmap.
Reset h_in to zeros at the start of each new scene.

Usage:
    PYTHONPATH=. python scripts/export_onnx_recurrent.py [params_recurrent.yaml]
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.conv_gru import build_recurrent_model


def export(params_path: str = "params_recurrent.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)

    ckpt_path = Path("artifacts/recurrent_model_best.pt")
    onnx_path = Path(p["onnx"]["model_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — run src/train_recurrent.py first"
        )

    n_az    = p["radar"]["n_azimuths"]
    n_range = p["radar"]["n_ranges"]

    model = build_recurrent_model(p)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    dummy_sweep = torch.zeros(1, 1, n_az, n_range)
    dummy_h     = torch.zeros(1, 1, n_az, n_range)  # single-channel confidence map

    torch.onnx.export(
        model,
        (dummy_sweep, dummy_h),
        str(onnx_path),
        dynamo=False,
        opset_version=p["onnx"]["opset_version"],
        input_names=["sweep_norm", "h_in"],
        output_names=["prob_map", "h_out"],
        dynamic_axes={
            "sweep_norm": {0: "batch"},
            "h_in":       {0: "batch"},
            "prob_map":   {0: "batch"},
            "h_out":      {0: "batch"},
        },
    )

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Smoke-test: run 5 steps, verify h updates and stays in [0, 1]
    import time
    sess  = ort.InferenceSession(str(onnx_path))
    h_np  = np.zeros((1, 1, n_az, n_range), dtype=np.float32)

    latencies = []
    for step in range(5):
        sweep_np = np.random.randn(1, 1, n_az, n_range).astype(np.float32)
        t0 = time.perf_counter()
        prob_map, h_np = sess.run(None, {"sweep_norm": sweep_np, "h_in": h_np})
        latencies.append((time.perf_counter() - t0) * 1000)

    assert prob_map.shape == (1, 1, n_az, n_range), f"Bad shape: {prob_map.shape}"
    assert prob_map.min() >= 0.0 and prob_map.max() <= 1.0, "prob_map out of [0,1]"
    assert h_np.shape == (1, 1, n_az, n_range)
    assert h_np.std() > 0, "Hidden state did not update"

    size_kb = onnx_path.stat().st_size // 1024
    lat_med = float(np.median(latencies))

    print(f"Exported  : {onnx_path}  ({size_kb} KB)")
    print("Interface : (sweep_norm, h_in) → (prob_map, h_out)")
    print(f"h shape   : (1, 1, {n_az}, {n_range})  — h IS the confidence heatmap")
    print(f"Latency   : {lat_med:.2f} ms median over 5 steps")
    print("ONNX spec : passed")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "params_recurrent.yaml"
    export(params_path)
