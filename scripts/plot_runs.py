"""
Read all runs from the local MLflow experiment and plot training curves.
Saves artifacts/run_comparison.png — open it after running.

Usage:
    python scripts/plot_runs.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


def run_label(run) -> str:
    p = run.data.params
    fft = p.get("use_fft", "?")
    lr = p.get("learning_rate", "?")
    return f"fft={fft}  lr={lr}"


def get_metric_history(client: MlflowClient, run_id: str, metric: str):
    history = client.get_metric_history(run_id, metric)
    history.sort(key=lambda m: m.step)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    return steps, values


def main():
    mlflow.set_experiment("weibel-mlops-demo")
    client = MlflowClient()

    experiment = client.get_experiment_by_name("weibel-mlops-demo")
    if experiment is None:
        print("No experiment found. Run train.py first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )

    if not runs:
        print("No runs found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MLflow run comparison — weibel-mlops-demo", fontsize=13)

    for run in runs:
        label = run_label(run)
        steps, loss = get_metric_history(client, run.info.run_id, "train_loss")
        if steps:
            ax1.plot(steps, loss, marker="o", markersize=3, label=label)

        steps, acc = get_metric_history(client, run.info.run_id, "val_accuracy")
        if steps:
            ax2.plot(steps, acc, marker="o", markersize=3, label=label)

    ax1.set_title("Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.95, 1.001)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("artifacts/run_comparison.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
