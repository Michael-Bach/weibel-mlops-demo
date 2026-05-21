# src/serving/serve.py
"""
FastAPI inference server — wraps OnnxPredictor as an HTTP endpoint.
Designed to run inside the Docker container, served by uvicorn.
"""

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.inference.predict_onnx import OnnxPredictor

app = FastAPI(title="Radar Signal Classifier", version="1.0")

with open("params.yaml") as f:
    _params = yaml.safe_load(f)
SIGNAL_LENGTH = _params["data"]["signal_length"]
MODEL_PATH = _params["onnx"]["model_path"]

_predictor = OnnxPredictor(MODEL_PATH)


class SignalRequest(BaseModel):
    # Flat list of float32 values — one signal or a batch
    signals: list[list[float]]

    @field_validator("signals")
    @classmethod
    def check_signal_length(cls, v):
        for i, s in enumerate(v):
            if len(s) != SIGNAL_LENGTH:
                raise ValueError(f"Signal {i} has length {len(s)}, expected {SIGNAL_LENGTH}")
        return v


class PredictionResponse(BaseModel):
    predictions: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "signal_length": SIGNAL_LENGTH}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SignalRequest):
    try:
        signals = np.array(request.signals, dtype=np.float32)
        labels, confidences = _predictor.predict(signals)
        return PredictionResponse(
            predictions=[
                {
                    "label": int(label),
                    "class": "target" if label == 1 else "clutter",
                    "confidence": round(float(conf), 4),
                }
                for label, conf in zip(labels, confidences)
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8080, reload=False)
