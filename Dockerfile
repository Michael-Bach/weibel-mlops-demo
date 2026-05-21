# Inference serving image — no PyTorch, no MLflow, no DVC.
FROM python:3.11-slim

WORKDIR /app

# Inference + serving dependencies only (~250 MB vs ~4 GB with PyTorch)
RUN pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "onnxruntime>=1.18.0" \
    "pyyaml>=6.0" \
    "fastapi>=0.111.0" \
    "uvicorn>=0.30.0"

# Copy inference code, serving layer, and model artifact
COPY src/__init__.py                src/
COPY src/inference/__init__.py      src/inference/
COPY src/inference/predict_onnx.py  src/inference/
COPY src/serving/__init__.py        src/serving/
COPY src/serving/serve.py           src/serving/
COPY artifacts/model.onnx           artifacts/
COPY params.yaml                    .

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "src.serving.serve:app", "--host", "0.0.0.0", "--port", "8080"]
