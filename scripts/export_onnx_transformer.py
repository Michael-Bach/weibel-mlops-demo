"""
Export PPITransformerDetector to ONNX.

Input  name : "ppi_stack"      shape (1, N_SW, n_az, n_range)  float32
Output name : "probability_map" shape (1, n_az, n_range)        float32
"""
import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.ppi_transformer import build_transformer

with open(ROOT / "params_ppi.yaml") as f:
    p = yaml.safe_load(f)

n_sw    = p["radar"]["n_sweeps"]
n_az    = p["radar"]["n_azimuths"]
n_range = p["radar"]["n_ranges"]

ckpt = ROOT / "artifacts" / "transformer_model_best.pt"
if not ckpt.exists():
    print(f"ERROR: checkpoint not found at {ckpt}")
    sys.exit(1)

_base = build_transformer(n_sweeps=n_sw)
_base.load_state_dict(torch.load(ckpt, map_location="cpu"))
_base.eval()

n_params = sum(q.numel() for q in _base.parameters())
print(f"Parameters: {n_params:,}")

# Wrap with sigmoid so ONNX model outputs probabilities directly
class _Wrapped(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.sigmoid(self.m(x))

model = _Wrapped(_base)
model.eval()

dummy = torch.zeros(1, n_sw, n_az, n_range)
out_path = ROOT / "artifacts" / "transformer_model.onnx"

torch.onnx.export(
    model,
    dummy,
    str(out_path),
    input_names=["ppi_stack"],
    output_names=["probability_map"],
    dynamic_axes={
        "ppi_stack":       {0: "batch"},
        "probability_map": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,   # legacy exporter: produces a single self-contained file
)
print(f"Exported: {out_path}")

# Quick smoke test
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(str(out_path))
test_in = np.random.rand(1, n_sw, n_az, n_range).astype(np.float32)
out = sess.run(None, {"ppi_stack": test_in})[0]
print(f"Output shape: {out.shape}  min={out.min():.3f}  max={out.max():.3f}")
print("OK")
