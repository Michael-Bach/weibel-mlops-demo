# Inference-only image — no PyTorch, no W&B, no DVC.
# Training runs in CI; this image is the deployment artifact.
FROM python:3.11-slim

WORKDIR /app

# Install only inference dependencies (~200 MB vs ~4 GB with PyTorch)
RUN pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "onnxruntime>=1.18.0" \
    "pyyaml>=6.0"

# Copy inference code and model artifact
COPY src/__init__.py               src/
COPY src/inference/__init__.py     src/inference/
COPY src/inference/predict_onnx.py src/inference/
COPY artifacts/model.onnx          artifacts/
COPY params.yaml                   .

ENV PYTHONPATH=/app

# Verify the model loads and runs a sample batch
CMD ["python", "-c", "\
import numpy as np; \
from src.inference.predict_onnx import OnnxPredictor; \
import yaml; \
params = yaml.safe_load(open('params.yaml')); \
n = params['data']['signal_length']; \
p = OnnxPredictor('artifacts/model.onnx'); \
labels, confs = p.predict(np.random.randn(4, n).astype('float32')); \
[print(f'sample {i}: {\"target\" if l == 1 else \"clutter\"} ({c:.3f})') for i, (l, c) in enumerate(zip(labels, confs))] \
"]
