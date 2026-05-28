"""
Convolutional GRU detector — confidence-map hidden state.

The hidden state h is a single-channel probability map over the PPI grid:
    h  : (B, 1, n_az, n_range) ∈ [0, 1]

h IS the detection confidence heatmap — directly inspectable after every sweep.
No post-processing needed: sigmoid(h) = h (already a probability).

Each sweep advances h via learned GRU gating:
    feat = encoder(sweep_norm)              # (B, C, H, W) multi-channel features
    r    = σ( Conv_r([feat ‖ h]) )         # reset:  how much of old h to expose?
    z    = σ( Conv_z([feat ‖ h]) )         # update: how much new evidence to absorb?
    ñ    = σ( Conv_n([feat ‖ r·h]) )       # candidate: new probability estimate ∈ [0,1]
    h′   = (1−z)·h + z·ñ                  # weighted blend — stays in [0,1]

ONNX interface (single-step streaming):
    Inputs:  sweep_norm (B,1,H,W)  h_in (B,1,H,W)
    Outputs: prob_map   (B,1,H,W)  h_out (B,1,H,W)   where prob_map ≡ h_out
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUDetector(nn.Module):
    """
    Encoder + single-state ConvGRU where h IS the confidence heatmap.

    h ∈ [0,1]^{1×n_az×n_range} — the hidden state equals the detection
    probability map and can be rendered as a heatmap after any sweep.

    Typical use:
        h = torch.zeros(1, 1, n_az, n_range)
        for sweep in stream:
            prob_map, h = model(sweep, h)
            display(h[0, 0])        # confidence heatmap — updates each sweep
    """

    def __init__(self, enc_ch: tuple[int, int] = (16, 32)):
        super().__init__()

        # Sweep encoder: 1 → enc_ch[0] → enc_ch[1] feature channels
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in enc_ch:
            enc_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        C = enc_ch[-1]  # encoder output channels

        # GRU gates — each maps cat([feat(C), h(1)]) → 1 confidence channel
        self.r_gate = nn.Conv2d(C + 1, 1, 3, padding=1)   # reset gate
        self.z_gate = nn.Conv2d(C + 1, 1, 3, padding=1)   # update gate
        self.cand   = nn.Conv2d(C + 1, 1, 3, padding=1)   # candidate (uses r·h)

        self.enc_ch = enc_ch
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sweep_norm: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sweep_norm : (B, 1, n_az, n_range)  EMA-normalised amplitude
            h          : (B, 1, n_az, n_range)  confidence heatmap ∈ [0, 1]
        Returns:
            prob_map   : (B, 1, n_az, n_range)  updated confidence heatmap ∈ [0, 1]
            h_next     : same tensor — pass back in on the next sweep
        """
        feat = self.encoder(sweep_norm)                         # (B, C, H, W)
        xh   = torch.cat([feat, h], dim=1)                     # (B, C+1, H, W)

        r = torch.sigmoid(self.r_gate(xh))
        z = torch.sigmoid(self.z_gate(xh))
        n = torch.sigmoid(self.cand(torch.cat([feat, r * h], dim=1)))

        h_next = (1.0 - z) * h + z * n                         # ∈ [0, 1]
        return h_next, h_next

    def zero_state(
        self,
        batch: int,
        n_az: int,
        n_range: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Zero-initialised confidence map for the start of a new scene."""
        return torch.zeros(batch, 1, n_az, n_range, device=device)


def build_recurrent_model(params: dict | None = None) -> ConvGRUDetector:
    if params is None:
        return ConvGRUDetector()
    enc_ch = tuple(params.get("model", {}).get("enc_ch", [16, 32]))
    return ConvGRUDetector(enc_ch=enc_ch)
