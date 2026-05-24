import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import build_model


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_data(params: dict) -> tuple[DataLoader, DataLoader]:
    processed = Path("data/processed")
    X_train = torch.tensor(np.load(processed / "X_train.npy"))
    X_val = torch.tensor(np.load(processed / "X_val.npy"))
    y_train = torch.tensor(np.load(processed / "y_train.npy"))
    y_val = torch.tensor(np.load(processed / "y_val.npy"))

    batch_size = params["training"]["batch_size"]
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def train(params_path: str = "params.yaml") -> float:
    params = load_params(params_path)
    tp = params["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow.set_experiment("weibel-mlops-demo")

    with mlflow.start_run():
        mlflow.log_params({
            **params["data"],
            **params["model"],
            **params["training"],
        })

        train_loader, val_loader = load_data(params)
        model = build_model(params_path).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"])
        criterion = nn.CrossEntropyLoss()

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
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_acc = evaluate(model, val_loader, device)
            avg_loss = total_loss / len(train_loader)

            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_accuracy": val_acc},
                step=epoch + 1,
            )

            print(f"Epoch {epoch+1:02d}/{tp['epochs']} | loss: {avg_loss:.4f} | acc: {val_acc:.4f}")

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
    print(f"\nBest val accuracy: {best_acc:.4f}")

    baseline = tp["baseline_accuracy"]
    if best_acc < baseline:
        print(f"FAILED: {best_acc:.4f} < baseline {baseline}")
        sys.exit(1)

    print(f"PASSED: {best_acc:.4f} >= baseline {baseline}")
    return best_acc


if __name__ == "__main__":
    train()
