"""
Synthetic Range-Doppler sequence generator.

Generates sequences of T=8 consecutive Range-Doppler maps (64 range × 128 Doppler bins).
Target-present: 2D Gaussian blob drifting linearly in range (constant-velocity model),
Doppler bin >= 5 (non-stationary). Clutter-only: spatially-correlated noise concentrated
near range=0, Doppler=0, consistent across scans.
"""

import numpy as np

T = 8
N_RANGE = 64
N_DOPPLER = 128
_MIN_DOPPLER = 5  # MTI: non-stationary target criterion


def _clutter_envelope() -> np.ndarray:
    """Power envelope for clutter — decays exponentially from (range=0, doppler=0)."""
    r = np.arange(N_RANGE, dtype=np.float32)
    d = np.arange(N_DOPPLER, dtype=np.float32)
    return (1.0 + 2.0 * np.exp(-r[:, None] / 10.0)) * (1.0 + 3.0 * np.exp(-d[None, :] / 12.0))


_CLUTTER_ENV = _clutter_envelope()


def _target_sequence(rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """T Range-Doppler maps with a linearly-drifting 2D Gaussian target blob."""
    d_center = float(rng.integers(_MIN_DOPPLER, N_DOPPLER - 10))
    r_start = float(rng.integers(4, N_RANGE - int(T * 2.5) - 4))
    r_vel = rng.uniform(0.5, 2.0)  # range bins per scan (constant velocity)
    amplitude = 10.0 ** (snr_db / 20.0)

    r_idx = np.arange(N_RANGE, dtype=np.float32)[:, None]
    d_idx = np.arange(N_DOPPLER, dtype=np.float32)[None, :]

    scans = []
    for t in range(T):
        r_center = r_start + t * r_vel
        blob = amplitude * np.exp(
            -0.5 * ((r_idx - r_center) / 2.0) ** 2
            - 0.5 * ((d_idx - d_center) / 2.0) ** 2
        ).astype(np.float32)
        clutter = (_CLUTTER_ENV * rng.standard_normal((N_RANGE, N_DOPPLER))).astype(np.float32)
        noise = rng.standard_normal((N_RANGE, N_DOPPLER)).astype(np.float32)
        scans.append((blob + clutter + noise).astype(np.float32))

    return np.stack(scans)  # (T, N_RANGE, N_DOPPLER)


def _clutter_sequence(rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """T Range-Doppler maps with no target — spatially-correlated clutter only."""
    amplitude = 10.0 ** (snr_db / 20.0)
    base = amplitude * _CLUTTER_ENV

    scans = []
    for _ in range(T):
        variation = 1.0 + 0.25 * rng.standard_normal((N_RANGE, N_DOPPLER)).astype(np.float32)
        noise = rng.standard_normal((N_RANGE, N_DOPPLER)).astype(np.float32)
        scans.append((base * variation + noise).astype(np.float32))

    return np.stack(scans)  # (T, N_RANGE, N_DOPPLER)


def generate_dataset(
    n_samples: int, snr_db: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_samples sequences of T=8 Range-Doppler maps.

    Args:
        n_samples: total sequences (half target, half clutter)
        snr_db:    signal-to-noise ratio in dB (target amplitude = 10^(snr_db/20))
        seed:      RNG seed for reproducibility

    Returns:
        X: float32 array of shape (n_samples, T, N_RANGE, N_DOPPLER) = (n, 8, 64, 128)
        y: int64 array of shape (n_samples,)  — 1=target present, 0=clutter only
    """
    rng = np.random.default_rng(seed)
    n_target = n_samples // 2
    n_clutter = n_samples - n_target

    sequences, labels = [], []
    for _ in range(n_target):
        sequences.append(_target_sequence(rng, snr_db))
        labels.append(1)
    for _ in range(n_clutter):
        sequences.append(_clutter_sequence(rng, snr_db))
        labels.append(0)

    X = np.stack(sequences)
    y = np.array(labels, dtype=np.int64)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
