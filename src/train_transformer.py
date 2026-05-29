"""
Train PPITransformerDetector on synthetically generated PPI sweep sequences.

Input to the model: noise-normalised raw amplitude stack (10, n_az, n_range).
Unlike the CNN which hand-engineers max/mean/std features, the transformer
receives the full temporal sequence and learns which sweeps to attend to.

Usage:
    PYTHONPATH=. python src/train_transformer.py [params_ppi.yaml]

Outputs:
    artifacts/transformer_model_best.pt   — best checkpoint
    artifacts/transformer_metrics.json    — {"val_f1": ..., "n_params": ...}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ppi_generator import generate_ppi_sequence, generate_clutter_only
from src.model.ppi_transformer import build_transformer


# ── Raw-stack sample builder ──────────────────────────────────────────────────

def _make_raw_sample(params: dict, has_target: bool,
                     rng: np.random.Generator, sample_seed: int):
    """
    Returns (X, y) where X is the noise-normalised raw amplitude stack.

    X : float32  (N_SW, n_az, n_range)  — 10 normalised sweeps
    y : float32  (n_az, n_range)        — soft Gaussian label (0 if no target)

    Normalisation: divide each sweep by the per-range 10th-percentile noise
    floor estimated across the full sequence (same floor used by temporal_features,
    so CNN and Transformer see comparable signal magnitudes).
    Clip to [0, 30] to suppress rare extreme values without destroying dynamics.
    """
    r       = params["radar"]
    d       = params["data"]
    n_range = int(r["n_ranges"])
    n_az    = int(r["n_azimuths"])
    margin  = int(d.get("range_margin", 10))
    snr_lo, snr_hi = d["snr_range"]

    if has_target:
        t_params = {
            "radar": r,
            "target": {
                "snr_db":              float(rng.uniform(snr_lo, snr_hi)),
                "range_bin":           float(rng.integers(margin, n_range - margin)),
                "azimuth_deg":         float(rng.uniform(0, 360)),
                "radial_velocity":     float(rng.uniform(-3.0, 3.0)),
                "tangential_velocity": float(rng.uniform(-4.0, 4.0)),
            },
        }
        ppi, label_seq, _ = generate_ppi_sequence(t_params, seed=int(sample_seed))
        label = label_seq.max(axis=0)   # (n_az, n_range) — target ever visible here
    else:
        ppi   = generate_clutter_only(params, seed=int(sample_seed))
        label = np.zeros((n_az, n_range), dtype=np.float32)

    # Noise-floor normalisation: same as temporal_features but keep all sweeps
    noise_fl = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-3)
    X = (ppi / noise_fl).clip(0, 30).astype(np.float32)   # (N_SW, n_az, n_range)

    return X, label.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawSweepDataset(Dataset):
    def __init__(self, params: dict, n_samples: int, seed_offset: int = 0):
        self.params      = params
        self.n_samples   = n_samples
        self.seed_offset = seed_offset
        rng = np.random.default_rng(seed_offset)
        self._has_target = np.arange(n_samples) % 2 == 0
        rng.shuffle(self._has_target)
        self._seeds = rng.integers(0, 2**31, size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(self._seeds[idx]))
        X, y = _make_raw_sample(
            self.params,
            has_target=bool(self._has_target[idx]),
            rng=rng,
            sample_seed=self.seed_offset + idx,
        )
        return torch.tensor(X), torch.tensor(y)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _f1(prob: np.ndarray, gt: np.ndarray, thr: float = 0.3) -> float:
    pred = prob > thr
    gt_b = gt   > thr
    tp = float((pred & gt_b).sum())
    fp = float((pred & ~gt_b).sum())
    fn = float((~pred & gt_b).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


# ── Training ──────────────────────────────────────────────────────────────────

def train(params_path: str = "params_ppi.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)

    Path("artifacts").mkdir(exist_ok=True)

    # Use same data/training hyperparams as CNN for a fair comparison
    epochs     = p["training"]["epochs"]
    lr         = p["training"]["learning_rate"]
    batch_size = p["training"]["batch_size"]
    seed       = p["data"]["seed"]
    n_train    = p["data"]["n_train"]
    n_val      = p["data"]["n_val"]
    n_sweeps   = p["radar"]["n_sweeps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    tr_ds  = RawSweepDataset(p, n_train, seed_offset=seed)
    val_ds = RawSweepDataset(p, n_val,   seed_offset=seed + 100_000)
    tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model    = build_transformer(n_sweeps=n_sweeps).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"Params  : {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1   = 0.0
    best_ckpt = Path("artifacts/transformer_model_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            prob = torch.sigmoid(model(Xb))
            loss = criterion(prob, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
        epoch_loss /= n_train

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                preds.append(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
                gts.append(yb.numpy())
        val_f1 = _f1(np.concatenate(preds), np.concatenate(gts))
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)

        print(f"Epoch {epoch:02d}/{epochs} | loss {epoch_loss:.4f} | val_F1 {val_f1:.4f}")

    print(f"\nBest val F1 : {best_f1:.4f}")
    Path("artifacts/transformer_metrics.json").write_text(
        json.dumps({"val_f1": round(best_f1, 4), "n_params": n_params}, indent=2)
    )


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "params_ppi.yaml"
    train(params_path)
