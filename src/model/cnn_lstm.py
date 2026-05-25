"""
CNN+LSTM classifier for multi-scan Range-Doppler sequences.

Architecture:
  Shared 2D CNN encodes each scan (1, 64, 128) → feature vector.
  LSTMCell unrolls over T=8 scans (Python loop → ONNX-unrolled at export).
  Final hidden state → Linear + Sigmoid → P(target present).

ONNX export input: (batch, T, 1, N_RANGE, N_DOPPLER) with T fixed at 8.
"""

import torch
import torch.nn as nn

from src.data.generator import N_DOPPLER, N_RANGE, T

_CNN_CH1 = 16
_CNN_CH2 = 32
_POOL_OUT = (4, 8)   # AdaptiveAvgPool2d target — (32, 4, 8) = 1024 features
_FEATURE_DIM = _CNN_CH2 * _POOL_OUT[0] * _POOL_OUT[1]  # 1024


class _ScanEncoder(nn.Module):
    """Shared CNN: (B, 1, N_RANGE, N_DOPPLER) → (B, _FEATURE_DIM)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, _CNN_CH1, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                          # → (16, 32, 64)
            nn.Conv2d(_CNN_CH1, _CNN_CH2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                          # → (32, 16, 32)
            nn.AdaptiveAvgPool2d(_POOL_OUT),                          # → (32, 4, 8)
            nn.Flatten(),                                             # → 1024
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CnnLstmClassifier(nn.Module):
    """Sequence classifier over T Range-Doppler maps.

    Input:  (batch, T, 1, N_RANGE, N_DOPPLER)
    Output: (batch, 1)  — sigmoid probability that a target is present
    """

    def __init__(self, lstm_hidden: int = 64) -> None:
        super().__init__()
        self.encoder = _ScanEncoder()
        self.lstm_cell = nn.LSTMCell(_FEATURE_DIM, lstm_hidden)
        self.lstm_hidden = lstm_hidden
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_len, C, H, W = x.shape
        h = torch.zeros(B, self.lstm_hidden, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.lstm_hidden, device=x.device, dtype=x.dtype)

        # Python for-loop → ONNX unrolls this into T replicated op-graphs
        for t in range(T_len):
            feat = self.encoder(x[:, t])        # (B, _FEATURE_DIM)
            h, c = self.lstm_cell(feat, (h, c))

        return torch.sigmoid(self.head(h))      # (B, 1)


def build_model(lstm_hidden: int = 64) -> CnnLstmClassifier:
    return CnnLstmClassifier(lstm_hidden=lstm_hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    m.eval()
    x = torch.randn(2, T, 1, N_RANGE, N_DOPPLER)
    out = m(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Parameters: {count_parameters(m):,}")
