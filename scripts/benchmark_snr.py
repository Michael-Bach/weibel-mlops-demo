# scripts/benchmark_snr.py
"""
Sweeps SNR from 0 to 25 dB and measures ONNX model accuracy at each level.
Generates equal numbers of target and clutter samples per SNR step.
Saves plot to artifacts/snr_benchmark.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import yaml

from src.data.generate import generate_clutter, generate_target


def run_benchmark(
    model_path: str = "artifacts/model.onnx",
    snr_min: float = -10.0,
    snr_max: float = 25.0,
    n_steps: int = 36,
    n_samples_per_class: int = 200,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    signal_length = params["data"]["signal_length"]

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(seed)
    snr_values = np.linspace(snr_min, snr_max, n_steps)
    accuracies = np.empty(n_steps)

    for i, snr_db in enumerate(snr_values):
        targets = np.stack(
            [generate_target(signal_length, snr_db, rng) for _ in range(n_samples_per_class)]
        )
        clutters = np.stack(
            [generate_clutter(signal_length, snr_db, rng) for _ in range(n_samples_per_class)]
        )
        X = np.concatenate([targets, clutters], axis=0).astype(np.float32)
        y = np.array([1] * n_samples_per_class + [0] * n_samples_per_class)

        logits = session.run(None, {input_name: X})[0]
        preds = logits.argmax(axis=1)
        accuracies[i] = (preds == y).mean()

    return snr_values, accuracies


def plot(
    snr_values: np.ndarray, accuracies: np.ndarray, out_path: str = "artifacts/snr_benchmark.png"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_values, accuracies * 100, marker="o", markersize=4, linewidth=2, color="#1f77b4")
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80% gate")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ONNX Model Accuracy vs SNR — Target vs. Clutter")
    ax.set_ylim(0, 105)
    ax.set_xlim(snr_values[0], snr_values[-1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    snr_values, accuracies = run_benchmark()
    for snr, acc in zip(snr_values, accuracies):
        print(f"SNR {snr:5.1f} dB → {acc * 100:.1f}%")
    plot(snr_values, accuracies)
