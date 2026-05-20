# src/data/generate.py
"""
Synthetic radar signal generator — target vs. clutter classification.

Signal model:
  Target:  sinusoidal return at a random frequency, amplitude set by SNR
  Clutter: bandlimited noise (low-frequency dominated, mimicking ground/sea return)

All parameters driven from params.yaml for full reproducibility.
"""

from pathlib import Path

import numpy as np
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)["data"]


def generate_target(signal_length: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal target return with additive white Gaussian noise."""
    t = np.linspace(0, 1, signal_length)
    freq = rng.uniform(5, 30)          # target doppler frequency (normalized)
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = 10 ** (snr_db / 20)   # convert dB to linear amplitude
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
    noise = rng.normal(0, 1, signal_length)
    return (signal + noise).astype(np.float32)


def generate_clutter(signal_length: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Bandlimited noise clutter — low-frequency dominated."""
    noise = rng.normal(0, 1, signal_length)
    # Low-pass filter via cumulative sum (cheap approximation of 1/f structure)
    clutter = np.cumsum(noise)
    clutter -= clutter.mean()
    # Scale to same power regime as targets for a fair classification challenge
    amplitude = 10 ** (snr_db / 20) * 0.4
    clutter = amplitude * clutter / (clutter.std() + 1e-8)
    return clutter.astype(np.float32)


def generate_dataset(params: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params["noise_seed"])
    n = params["n_samples"]
    length = params["signal_length"]
    snr_low, snr_high = params["snr_range"]
    n_clutter = int(n * params["clutter_ratio"])
    n_target = n - n_clutter

    samples, labels = [], []

    for _ in range(n_target):
        snr = rng.uniform(snr_low, snr_high)
        samples.append(generate_target(length, snr, rng))
        labels.append(1)  # target

    for _ in range(n_clutter):
        snr = rng.uniform(snr_low, snr_high)
        samples.append(generate_clutter(length, snr, rng))
        labels.append(0)  # clutter

    X = np.stack(samples)
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_split: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_val = int(n * val_split)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


if __name__ == "__main__":
    params = load_params()
    rng = np.random.default_rng(params["noise_seed"])

    print("Generating synthetic radar dataset...")
    X, y = generate_dataset(params)
    X_train, X_val, y_train, y_val = train_val_split(X, y, params["val_split"], rng)

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_val.npy", X_val)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_val.npy", y_val)

    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    np.save(raw / "X.npy", X)
    np.save(raw / "y.npy", y)

    print(f"Dataset: {len(X)} samples, {X.shape[1]}-point signals")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Target: {y.sum()} | Clutter: {(y == 0).sum()}")