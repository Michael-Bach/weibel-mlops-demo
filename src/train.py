"""
Training pipeline for the CNN+LSTM Range-Doppler sequence classifier.

Generates data on-the-fly (no pre-saved .npy files), trains with BCE loss,
logs T and CNN architecture as MLflow params, saves best checkpoint.
CI gate: sys.exit(1) if val_accuracy < rd_training.baseline_accuracy.
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.generator import T, generate_dataset
from src.model.cnn_lstm import build_model


def _load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_loaders(params: dict) -> tuple[DataLoader, DataLoader]:
    dp = params["rd_data"]
    X, y = generate_dataset(dp["n_samples"], dp["snr_db"], seed=dp["noise_seed"])

    n = len(X)
    n_val = int(n * dp["val_split"])
    rng = np.random.default_rng(dp["noise_seed"])
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # (n, T, 64, 128) → (n, T, 1, 64, 128)  — add channel dim for CNN
    X_t = torch.tensor(X[train_idx]).unsqueeze(2)
    X_v = torch.tensor(X[val_idx]).unsqueeze(2)
    y_t = torch.tensor(y[train_idx], dtype=torch.float32)
    y_v = torch.tensor(y[val_idx], dtype=torch.float32)

    bs = params["rd_training"]["batch_size"]
    return (
        DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True),
        DataLoader(TensorDataset(X_v, y_v), batch_size=bs),
    )


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            prob = model(X).squeeze(1)
            pred = (prob >= 0.5).long()
            correct += (pred == y.long()).sum().item()
            total += len(y)
    return correct / total


def train(params_path: str = "params.yaml") -> float:
    params = _load_params(params_path)
    tp = params["rd_training"]
    mp = params.get("rd_model", {})
    dp = params["rd_data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow.set_experiment("weibel-mlops-demo")

    with mlflow.start_run():
        mlflow.log_params({
            # Sequence / CNN architecture params — required by spec
            "T": dp.get("T", T),
            "n_range": dp.get("n_range", 64),
            "n_doppler": dp.get("n_doppler", 128),
            "cnn_channels": "16,32",
            "cnn_kernel": "3x3",
            "lstm_hidden": mp.get("lstm_hidden", 64),
            # Data / training params
            "n_samples": dp["n_samples"],
            "snr_db": dp["snr_db"],
            "val_split": dp["val_split"],
            "epochs": tp["epochs"],
            "batch_size": tp["batch_size"],
            "learning_rate": tp["learning_rate"],
        })

        train_loader, val_loader = _make_loaders(params)
        model = build_model(lstm_hidden=mp.get("lstm_hidden", 64)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"])

        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)

        best_acc = 0.0
        best_epoch = 0

        for epoch in range(tp["epochs"]):
            model.train()
            total_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                prob = model(X).squeeze(1)
                loss = F.binary_cross_entropy(prob, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_acc = _evaluate(model, val_loader, device)
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_accuracy": val_acc},
                step=epoch + 1,
            )
            print(f"Epoch {epoch+1:02d}/{tp['epochs']} | loss {avg_loss:.4f} | acc {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), artifacts / "model_best.pt")

        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_artifact(str(artifacts / "model_best.pt"))

    metrics = {"val_accuracy": best_acc}
    with open(artifacts / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest val accuracy: {best_acc:.4f}  (epoch {best_epoch})")

    baseline = tp["baseline_accuracy"]
    if best_acc < baseline:
        print(f"FAILED: {best_acc:.4f} < {baseline}")
        sys.exit(1)

    print(f"PASSED: {best_acc:.4f} >= {baseline}")
    return best_acc


if __name__ == "__main__":
    train()
