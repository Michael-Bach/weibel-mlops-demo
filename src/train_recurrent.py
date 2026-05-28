"""
Train ConvGRUDetector on synthetically generated PPI sweep sequences.

Key differences from train_ppi.py:
  - Input: raw amplitude sweeps (EMA-normalised), one per timestep.
    No hand-crafted temporal statistics — the GRU learns integration.
  - Loss: weighted BCE (not MSE) — more appropriate for sparse binary maps.
  - Curriculum: seq_len grows in stages so the model first learns single-
    sweep detection before being asked to integrate over 20 sweeps.
  - Hidden state zeroed at the start of each batch sequence.

Usage:
    PYTHONPATH=. python src/train_recurrent.py [params_recurrent.yaml]

Outputs:
    artifacts/recurrent_model_best.pt   — best checkpoint (val F1)
    artifacts/recurrent_metrics.json    — {"val_f1": ..., "n_params": ...}
    MLflow run under experiment "recurrent-radar-detector"
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sequence_dataset import PPISequenceDataset
from src.model.conv_gru import build_recurrent_model


# ── Metrics ───────────────────────────────────────────────────────────────────

def _f1(prob: np.ndarray, gt: np.ndarray, thr: float = 0.3) -> float:
    pred = prob > thr
    gt_b = gt   > thr
    tp = float((pred & gt_b).sum())
    fp = float((pred & ~gt_b).sum())
    fn = float((~pred & gt_b).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


# ── Curriculum helper ─────────────────────────────────────────────────────────

def _seq_len_for_epoch(curriculum: list[dict], epoch: int) -> int:
    """Return the seq_len to use for a given epoch (1-indexed)."""
    for stage in curriculum:
        if epoch <= stage["epoch_end"]:
            return stage["seq_len"]
    return curriculum[-1]["seq_len"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(params_path: str = "params_recurrent.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)

    Path("artifacts").mkdir(exist_ok=True)

    tr_cfg   = p["training"]
    epochs      = tr_cfg["epochs"]
    lr          = tr_cfg["learning_rate"]
    wd          = tr_cfg["weight_decay"]
    batch_size  = tr_cfg["batch_size"]
    pos_weight  = tr_cfg["pos_weight"]
    baseline_f1 = tr_cfg["baseline_f1"]
    curriculum  = tr_cfg["curriculum"]
    n_train     = p["data"]["n_train"]
    n_val       = p["data"]["n_val"]
    seed        = p["data"]["seed"]

    n_az    = p["radar"]["n_azimuths"]
    n_range = p["radar"]["n_ranges"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    model    = build_recurrent_model(p).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"Params  : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1   = 0.0
    best_ckpt = Path("artifacts/recurrent_model_best.pt")

    # Validation dataset uses the longest seq_len for a stable metric
    val_seq_len = curriculum[-1]["seq_len"]
    val_ds = PPISequenceDataset(p, n_val, seq_len=val_seq_len, seed_offset=seed + 100_000)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    mlflow.set_experiment("recurrent-radar-detector")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":       epochs,
            "lr":           lr,
            "weight_decay": wd,
            "batch_size":   batch_size,
            "pos_weight":   pos_weight,
            "n_train":      n_train,
            "n_val":        n_val,
            "seed":         seed,
            "n_azimuths":   n_az,
            "n_ranges":     n_range,
            "snr_range":    str(p["data"]["snr_range"]),
            "enc_ch":       str(p["model"]["enc_ch"]),
            "n_params":     n_params,
            "device":       str(device),
        })
        mlflow.log_artifact(params_path, artifact_path="config")

        for epoch in range(1, epochs + 1):
            seq_len = _seq_len_for_epoch(curriculum, epoch)

            # Rebuild training loader when seq_len changes (avoids DataLoader restart)
            tr_ds = PPISequenceDataset(p, n_train, seq_len=seq_len, seed_offset=seed)
            tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)

            # ── Train ──────────────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0

            for sweeps, labels in tr_dl:
                # sweeps : (B, T, 1, n_az, n_range)
                # labels : (B, T, n_az, n_range)
                sweeps = sweeps.to(device)
                labels = labels.to(device)
                B, T = sweeps.shape[:2]

                optimizer.zero_grad()

                h = model.zero_state(B, n_az, n_range, device=device)
                batch_loss = torch.tensor(0.0, device=device)

                pw = torch.tensor(pos_weight, device=device)
                for t in range(T):
                    prob_map, h = model(sweeps[:, t], h)
                    tgt = labels[:, t]
                    # Weighted MSELoss: upweight target cells so their gradient isn't
                    # drowned out by the ~31:1 background-to-foreground cell ratio.
                    w = 1.0 + (tgt > 0.01).float() * (pw - 1.0)
                    batch_loss = batch_loss + (w * (prob_map.squeeze(1) - tgt).pow(2)).mean()

                batch_loss = batch_loss / T
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += batch_loss.item() * B

            epoch_loss /= n_train

            # ── Validate (last sweep output only) ──────────────────────────
            model.eval()
            preds_last, gts_last = [], []

            with torch.no_grad():
                for sweeps, labels in val_dl:
                    sweeps = sweeps.to(device)
                    B, T   = sweeps.shape[:2]
                    h = model.zero_state(B, n_az, n_range, device=device)

                    for t in range(T):
                        prob_map, h = model(sweeps[:, t], h)

                    # Evaluate F1 on last sweep output
                    preds_last.append(prob_map.squeeze(1).cpu().numpy())
                    gts_last.append(labels[:, -1].numpy())

            val_f1 = _f1(np.concatenate(preds_last), np.concatenate(gts_last))
            mlflow.log_metrics(
                {"train_loss": epoch_loss, "val_f1": val_f1, "seq_len": seq_len},
                step=epoch,
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_ckpt)

            scheduler.step()
            print(
                f"Epoch {epoch:02d}/{epochs} | T={seq_len:2d} | "
                f"loss {epoch_loss:.4f} | val_F1 {val_f1:.4f}"
            )

        print(f"\nBest val F1 : {best_f1:.4f}")
        mlflow.log_metric("best_val_f1", best_f1)

        if best_f1 >= baseline_f1:
            mlflow.log_artifact(str(best_ckpt), artifact_path="model")
            print(f"PASSED : {best_f1:.4f} >= gate {baseline_f1}")
        else:
            print(f"FAILED : {best_f1:.4f} < gate {baseline_f1}")
            mlflow.set_tag("ci_gate", "FAILED")
            sys.exit(1)

    metrics = {"val_f1": round(best_f1, 4), "n_params": n_params}
    Path("artifacts/recurrent_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "params_recurrent.yaml"
    train(params_path)
