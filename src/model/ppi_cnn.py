"""
Lightweight FCN for PPI radar target detection.

Input  : (batch, 3, n_az, n_range)
         3 channels = [temporal_max, temporal_mean, temporal_std]
         computed from n_sweeps raw amplitude maps.
Output : (batch, n_az, n_range)  — per-cell probability in [0, 1]

All spatial dimensions are preserved (padding=1 throughout).
"""

import torch
import torch.nn as nn
import yaml


class PPIDetectorCNN(nn.Module):
    def __init__(self, filters: list = None):
        super().__init__()
        if filters is None:
            filters = [16, 32, 32, 16]

        layers = []
        in_ch = 3           # temporal feature channels: max / mean / std
        for out_ch in filters:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))   # logit map
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits — apply sigmoid for probabilities."""
        return self.net(x).squeeze(1)               # (B, n_az, n_range)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probability map for inference."""
        return torch.sigmoid(self.forward(x))


def build_model(params_path: str = "params_ppi.yaml") -> PPIDetectorCNN:
    with open(params_path) as f:
        p = yaml.safe_load(f)
    return PPIDetectorCNN(filters=p["model"]["filters"])
