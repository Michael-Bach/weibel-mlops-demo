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
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()

        layers = []
        prev_dim = input_dim
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
        return self.net(x)


def build_model(params_path: str = "params.yaml") -> RadarClassifier:
    params = load_params(params_path)
    return RadarClassifier(
        input_dim=params["data"]["signal_length"],
        hidden_dims=params["model"]["hidden_dims"],
        dropout=params["model"]["dropout"],
    )


if __name__ == "__main__":
    model = build_model()
    print(model)
    x = torch.randn(8, 128)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")