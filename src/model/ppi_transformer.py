"""
Patch Temporal Transformer for PPI radar target detection.

Key difference from the CNN:
  CNN   — computes fixed temporal statistics (max, mean, std) then learns
          a spatial filter on top of those hand-engineered features.
  PPI-TF — receives the raw 10-sweep amplitude stack and learns WHICH sweeps
            to attend to at each spatial location.  The attention weights are
            directly interpretable: high attention on the 1-2 sweeps where the
            target passed through that cell.

Architecture
------------
Input  : (B, N_SW, H, W)  noise-normalised amplitude stack (not pre-aggregated)
  1. Patch embed   — divide each sweep into (PH × PW) patches, project to d_model
  2. Sweep pos enc — add learnable positional encoding for sweeps 0..N_SW-1
  3. Temporal attn — TransformerEncoder operates across N_SW sweep tokens per patch
  4. Aggregate     — mean-pool over sweeps, decode back to patch amplitude map
  5. Reshape       — assemble patches back into (B, H, W) probability logits

Output : (B, H, W)  logits — apply sigmoid for probabilities

Patch grid: 30 az × 16 range  (6-bin × 4-bin patches on 180×64 PPI)
Parameters: ~19 k  (comparable to the CNN)
"""

import math
import torch
import torch.nn as nn


class PPITransformerDetector(nn.Module):
    PATCH_H = 6    # azimuth bins per patch   → 180 / 6  = 30 patches
    PATCH_W = 4    # range   bins per patch   → 64  / 4  = 16 patches

    def __init__(
        self,
        n_sweeps:   int = 10,
        d_model:    int = 32,
        nhead:      int = 4,
        n_layers:   int = 2,
        dim_ff:     int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_sweeps = n_sweeps
        self.d_model  = d_model
        patch_dim     = self.PATCH_H * self.PATCH_W   # 24

        # 1. Linear projection of each patch amplitude vector → d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # 2. Learnable sweep positional encoding
        self.sweep_pos = nn.Embedding(n_sweeps, d_model)

        # 3. Transformer encoder — attends across N_SW sweep tokens per patch
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,          # (batch, seq, dim)
            norm_first=True,           # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 4. Decode: mean-pooled token → patch amplitude logits
        self.decode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, patch_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.sweep_pos.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N_SW, H, W)  noise-normalised amplitude stack
        returns : (B, H, W)  logits
        """
        B, N, H, W = x.shape
        PH, PW     = self.PATCH_H, self.PATCH_W
        nph        = H // PH   # 30
        npw        = W // PW   # 16

        # ── 1. Extract patches: (B, N, nph, npw, PH, PW) ───────────────────
        xp = x.reshape(B, N, nph, PH, npw, PW)
        xp = xp.permute(0, 2, 4, 1, 3, 5)       # (B, nph, npw, N, PH, PW)
        xp = xp.reshape(B * nph * npw, N, PH * PW)

        # ── 2. Embed + positional encoding ──────────────────────────────────
        tok = self.patch_embed(xp)               # (B*nph*npw, N, d)
        pos = self.sweep_pos(
            torch.arange(N, device=x.device)
        )                                        # (N, d)
        tok = tok + pos.unsqueeze(0)             # broadcast over batch dim

        # ── 3. Temporal self-attention across N sweeps ───────────────────────
        out = self.transformer(tok)              # (B*nph*npw, N, d)

        # ── 4. Aggregate sweeps → decode ────────────────────────────────────
        out = out.mean(dim=1)                    # (B*nph*npw, d)
        out = self.decode(out)                   # (B*nph*npw, PH*PW)

        # ── 5. Reassemble spatial map ────────────────────────────────────────
        out = out.reshape(B, nph, npw, PH, PW)
        out = out.permute(0, 1, 3, 2, 4)         # (B, nph, PH, npw, PW)
        out = out.reshape(B, H, W)               # (B, H, W)  logits

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return average attention weight matrix (N_SW × N_SW) across all patches
        and heads — useful for visualising which sweeps the model attends to.
        Only works with a single TransformerEncoderLayer (n_layers=1) variant;
        for n_layers>1 returns the first layer's weights.
        """
        B, N, H, W = x.shape
        PH, PW     = self.PATCH_H, self.PATCH_W
        nph, npw   = H // PH, W // PW

        xp  = x.reshape(B, N, nph, PH, npw, PW)
        xp  = xp.permute(0, 2, 4, 1, 3, 5).reshape(B * nph * npw, N, PH * PW)
        tok = self.patch_embed(xp)
        tok = tok + self.sweep_pos(torch.arange(N, device=x.device)).unsqueeze(0)

        layer = self.transformer.layers[0]
        with torch.no_grad():
            _, attn = layer.self_attn(tok, tok, tok, need_weights=True,
                                      average_attn_weights=True)
        return attn.mean(dim=0)   # (N_SW, N_SW)  averaged over all patches


def build_transformer(n_sweeps: int = 10) -> PPITransformerDetector:
    return PPITransformerDetector(n_sweeps=n_sweeps)
