# src/models/classifier.py
"""
MLP classifier for radar signal classification (target vs. clutter).

Deliberately simple — the pipeline is the point, not the architecture.
In production: swap for a 1D-CNN or transformer operating on range-Doppler maps.
"""

import torch
import torch.nn as nn
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


class RadarClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int], dropout: float, use_fft: bool = False
    ):
        super().__init__()
        self.use_fft = use_fft

        # rfft of a length-N signal produces N//2+1 unique frequency bins
        feature_dim = input_dim // 2 + 1 if use_fft else input_dim
        if use_fft:
            self.register_buffer("hanning_window", torch.hann_window(input_dim))

        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # binary: target vs. clutter
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fft:
            x = x * self.hanning_window
            spec = torch.fft.rfft(x, dim=1)
            # magnitude via real ops — avoids ReduceL2 opset-18 compat issue
            x = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2))
        return self.net(x)


def build_model(params_path: str = "params.yaml") -> RadarClassifier:
    params = load_params(params_path)
    return RadarClassifier(
        input_dim=params["data"]["signal_length"],
        hidden_dims=params["model"]["hidden_dims"],
        dropout=params["model"]["dropout"],
        use_fft=params["model"].get("use_fft", False),
    )


if __name__ == "__main__":
    model = build_model()
    print(model)
    x = torch.randn(8, 128)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")