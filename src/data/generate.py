# src/data/generate.py
"""
Synthetic radar signal generator — target vs. clutter classification.

Signal model:
  Target:  sinusoidal return at a random frequency, amplitude set by SNR
  Clutter: bandlimited noise (low-frequency dominated, mimicking ground/sea return)

All parameters driven from params.yaml for full reproducibility.
"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
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
    """Low-frequency dominated clutter at specified SNR above the AWGN noise floor.

    The clutter waveform is generated from a low-pass filtered noise process (cumulative
    sum as a cheap 1/f approximation), scaled to the requested SNR, then AWGN is added
    on top — identical treatment to the target model. This makes the SNR parameter
    physically meaningful: at low SNR both classes are noise-dominated and harder to
    separate; at high SNR the spectral structure (diffuse low-frequency vs sharp peak)
    is clearly visible.
    """
    raw = rng.normal(0, 1, signal_length)
    clutter = np.cumsum(raw)
    clutter -= clutter.mean()
    clutter /= clutter.std() + 1e-8          # unit-std low-frequency waveform
    amplitude = 10 ** (snr_db / 20)          # same SNR convention as generate_target
    clutter = amplitude * clutter
    noise = rng.normal(0, 1, signal_length)  # receiver AWGN — same noise floor as target
    return (clutter + noise).astype(np.float32)


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


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _data_hash(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(X.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


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

    # Lineage record — written alongside the data so every dataset version
    # carries a complete description of how it was produced.
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "params": params,
        "outputs": {
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "signal_length": X.shape[1],
            "class_distribution": {
                "target": int(y.sum()),
                "clutter": int((y == 0).sum()),
            },
        },
        "data_hash_md5": _data_hash(X, y),
    }
    meta_path = out / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset: {len(X)} samples, {X.shape[1]}-point signals")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Target: {y.sum()} | Clutter: {(y == 0).sum()}")
    print(f"Lineage: {meta_path}  hash={metadata['data_hash_md5'][:12]}…")