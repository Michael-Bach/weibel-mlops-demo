"""
Train PPIDetectorCNN on synthetically generated PPI sweep sequences.

Input to the model: 3-channel temporal feature maps (max/mean/std after
clutter-floor normalisation), NOT the raw 10-channel amplitude cube.
This suppresses range-dependent clutter and makes the moving target
visible as a bright streak in the max and std channels.

Dataset is 50 % target / 50 % clutter-only, generated on-the-fly so
the full dataset never occupies more than batch_size samples in RAM.

Usage:
    PYTHONPATH=. python src/train_ppi.py [params_ppi.yaml]

Outputs:
    artifacts/ppi_model_best.pt   — best checkpoint (val F1 @ thr 0.3)
    artifacts/ppi_metrics.json    — {"val_f1": ..., "n_params": ...}
    MLflow run logged under experiment "ppi-radar-detector"
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ppi_generator import make_sample
from src.model.ppi_cnn import build_model


# ── Dataset ───────────────────────────────────────────────────────────────────

class PPIDataset(Dataset):
    def __init__(self, params: dict, n_samples: int, seed_offset: int = 0):
        self.params      = params
        self.n_samples   = n_samples
        self.seed_offset = seed_offset
        rng = np.random.default_rng(seed_offset)
        # Pre-decide which samples have a target (balanced 50/50)
        self._has_target = np.arange(n_samples) % 2 == 0
        rng.shuffle(self._has_target)
        # Store per-sample RNG state (for reproducible target parameters)
        self._seeds = rng.integers(0, 2**31, size=n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self._seeds[idx]))
        X, y = make_sample(
            self.params,
            has_target=bool(self._has_target[idx]),
            rng=rng,
            sample_seed=self.seed_offset + idx,
        )
        return torch.tensor(X), torch.tensor(y)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _f1(prob: np.ndarray, gt: np.ndarray, thr: float = 0.3) -> float:
    """F1: binarise both at thr=0.3 (Gaussian label > e^-0.5 at σ≈5 cells)."""
    pred = prob > thr
    gt_b = gt  > thr
    tp = float((pred & gt_b).sum())
    fp = float((pred & ~gt_b).sum())
    fn = float((~pred & gt_b).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


# ── Training ──────────────────────────────────────────────────────────────────

def train(params_path: str = "params_ppi.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)

    Path("artifacts").mkdir(exist_ok=True)

    epochs      = p["training"]["epochs"]
    lr          = p["training"]["learning_rate"]
    batch_size  = p["training"]["batch_size"]
    baseline_f1 = p["training"]["baseline_f1"]
    seed        = p["data"]["seed"]
    n_train     = p["data"]["n_train"]
    n_val       = p["data"]["n_val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    tr_ds  = PPIDataset(p, n_train, seed_offset=seed)
    val_ds = PPIDataset(p, n_val,   seed_offset=seed + 100_000)
    tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model     = build_model(params_path).to(device)
    n_params  = sum(q.numel() for q in model.parameters())
    print(f"Params  : {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_f1   = 0.0
    best_ckpt = Path("artifacts/ppi_model_best.pt")

    mlflow.set_experiment("ppi-radar-detector")
    with mlflow.start_run():
        # Log all hyperparameters from params file
        mlflow.log_params({
            "epochs":       epochs,
            "lr":           lr,
            "batch_size":   batch_size,
            "n_train":      n_train,
            "n_val":        n_val,
            "seed":         seed,
            "n_azimuths":   p["radar"]["n_azimuths"],
            "n_ranges":     p["radar"]["n_ranges"],
            "n_sweeps":     p["radar"]["n_sweeps"],
            "snr_range":    str(p["data"]["snr_range"]),
            "filters":      str(p["model"]["filters"]),
            "n_params":     n_params,
            "device":       str(device),
        })
        mlflow.log_artifact(params_path, artifact_path="config")

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for Xb, yb in tr_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                prob = torch.sigmoid(model(Xb))
                loss = criterion(prob, yb)
                loss.backward()
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

            mlflow.log_metrics({"train_loss": epoch_loss, "val_f1": val_f1}, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_ckpt)

            scheduler.step()
            print(f"Epoch {epoch:02d}/{epochs} | loss {epoch_loss:.4f} | val_F1 {val_f1:.4f}")

        print(f"\nBest val F1 : {best_f1:.4f}")
        mlflow.log_metric("best_val_f1", best_f1)

        if best_f1 >= baseline_f1:
            mlflow.log_artifact(str(best_ckpt), artifact_path="model")
            print(f"PASSED : {best_f1:.4f} >= gate {baseline_f1}")
        else:
            print(f"FAILED : {best_f1:.4f} < gate {baseline_f1}")
            mlflow.set_tag("ci_gate", "FAILED")
            sys.exit(1)

    metrics_path = Path("artifacts") / "ppi_metrics.json"
    metrics_path.write_text(
        json.dumps({"val_f1": round(best_f1, 4), "n_params": n_params}, indent=2)
    )


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "params_ppi.yaml"
    train(params_path)
