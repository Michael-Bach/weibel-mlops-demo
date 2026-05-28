"""
PPISequenceDataset — generates variable-length PPI sweep sequences for
training the ConvGRU recurrent detector.

Key differences from make_sample() / PPIDataset:
  - Returns T raw amplitude sweeps, not pre-aggregated temporal features.
  - Label is per-sweep (target position at each step t), not the union
    across all sweeps — so the model learns to fire at the current cell,
    not at the trail.
  - Sequence length T is a parameter, supporting curriculum training
    (short sequences early, longer ones later).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.ppi_generator import (
    generate_ppi_sequence,
    generate_clutter_only,
)


class PPISequenceDataset(Dataset):
    """
    Each sample is (sweeps, labels):
        sweeps : float32  (T, 1, n_az, n_range)  raw amplitude, normalised
        labels : float32  (T, n_az, n_range)      per-sweep soft Gaussian

    Normalisation: each sweep divided by the EMA noise floor estimate
    computed up to that sweep (mirrors what the app does at inference).
    This keeps the dataset consistent with the streaming deployment path.

    For clutter-only samples labels is all zeros.
    """

    def __init__(
        self,
        params: dict,
        n_samples: int,
        seq_len: int = 20,
        seed_offset: int = 0,
    ):
        self.params      = params
        self.n_samples   = n_samples
        self.seq_len     = seq_len
        self.seed_offset = seed_offset

        r = params["radar"]
        self.n_az    = int(r["n_azimuths"])
        self.n_range = int(r["n_ranges"])
        self.snr_lo, self.snr_hi = params["data"]["snr_range"]

        rng = np.random.default_rng(params["data"]["seed"] + seed_offset)
        self._seeds    = rng.integers(1, 10_000_000, size=n_samples).tolist()
        self._has_tgt  = [i % 2 == 0 for i in range(n_samples)]  # balanced
        self._t_params = []
        for i in range(n_samples):
            if self._has_tgt[i]:
                margin = int(params["data"].get("range_margin", 10))
                self._t_params.append({
                    "radar": r,
                    "target": {
                        "snr_db":              float(rng.uniform(self.snr_lo, self.snr_hi)),
                        "range_bin":           float(rng.integers(margin, self.n_range - margin)),
                        "azimuth_deg":         float(rng.uniform(0, 360)),
                        "radial_velocity":     float(rng.uniform(-3.0, 3.0)),
                        "tangential_velocity": float(rng.uniform(-4.0, 4.0)),
                    },
                })
            else:
                self._t_params.append(None)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        seed    = self._seeds[idx]
        has_tgt = self._has_tgt[idx]

        if has_tgt:
            ppi_full, label_full, _ = generate_ppi_sequence(
                self._t_params[idx], seed=seed
            )
            # ppi_full  : (n_sw_cfg, n_az, n_range)
            # label_full: (n_sw_cfg, n_az, n_range) soft Gaussian per sweep
        else:
            ppi_full   = generate_clutter_only(self.params, seed=seed)
            label_full = np.zeros_like(ppi_full)

        n_sw_cfg = ppi_full.shape[0]

        # If seq_len > generated sweeps, tile the sequence
        if self.seq_len > n_sw_cfg:
            reps = (self.seq_len + n_sw_cfg - 1) // n_sw_cfg
            ppi_full   = np.tile(ppi_full,   (reps, 1, 1))[:self.seq_len]
            label_full = np.tile(label_full, (reps, 1, 1))[:self.seq_len]
        else:
            ppi_full   = ppi_full[:self.seq_len]
            label_full = label_full[:self.seq_len]

        # EMA noise floor normalisation — matches inference-time preprocessing
        sweeps_norm = _ema_normalise(ppi_full)  # (T, n_az, n_range)

        sweeps_t = torch.from_numpy(sweeps_norm[:, np.newaxis])  # (T, 1, n_az, n_range)
        labels_t = torch.from_numpy(label_full.astype(np.float32))  # (T, n_az, n_range)

        return sweeps_t, labels_t


def _ema_normalise(ppi: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """
    Apply per-range EMA noise floor normalisation sweep by sweep.

    The same procedure runs in the app at inference time so that training
    and deployment see identical input distributions.

    Args:
        ppi   : (T, n_az, n_range) raw amplitude
        alpha : EMA decay (0.9 → 10-sweep timescale)
    Returns:
        (T, n_az, n_range) float32 normalised amplitude
    """
    T = ppi.shape[0]
    out = np.empty_like(ppi, dtype=np.float32)

    # Initialise from first sweep
    noise_floor = np.percentile(ppi[0], 10, axis=0)  # (n_range,)
    noise_floor = noise_floor.clip(1e-3)

    for t in range(T):
        sweep = ppi[t]  # (n_az, n_range)
        out[t] = (sweep / noise_floor).astype(np.float32)
        # Update EMA after using current estimate
        noise_floor = alpha * noise_floor + (1.0 - alpha) * np.percentile(sweep, 10, axis=0)
        noise_floor = noise_floor.clip(1e-3)

    return out
