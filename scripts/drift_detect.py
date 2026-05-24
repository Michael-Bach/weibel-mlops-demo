"""
Feature drift detection for deployed radar classifier.

Computes Population Stability Index (PSI) and KL divergence between a
reference distribution (validation set at training time) and an incoming
batch of live inference inputs.

Operational scenario:
    The classifier is deployed to a radar system. Every N hours a batch of
    recent inference inputs is extracted and compared against the training
    distribution. If PSI exceeds the alert threshold the on-call team is
    notified and a retraining review is initiated.

PSI interpretation (standard industry thresholds):
    < 0.10    — stable, no action required
    0.10–0.20 — moderate shift, monitor closely
    > 0.20    — major shift, retraining review recommended

Usage:
    # Compare live batch against training reference:
    python scripts/drift_detect.py \\
        --reference data/processed/X_val.npy \\
        --current   data/incoming/X_live.npy

    # Exit codes: 0 = ok, 1 = warn (PSI 0.10–0.20), 2 = alert (PSI > 0.20)
    # Suitable for use in CI gates or cron-triggered monitoring jobs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PSI_WARN = 0.10
PSI_ALERT = 0.20
N_BINS = 10


def _signal_energy(X: np.ndarray) -> np.ndarray:
    """Mean squared amplitude — a scalar summary of signal power per sample."""
    return np.mean(X**2, axis=1)


def _spectral_centroid(X: np.ndarray) -> np.ndarray:
    """Frequency centroid of magnitude spectrum — sensitive to Doppler shifts."""
    spectrum = np.abs(np.fft.rfft(X, axis=1))
    freqs = np.arange(spectrum.shape[1], dtype=float)
    centroid = (freqs * spectrum).sum(axis=1) / (spectrum.sum(axis=1) + 1e-8)
    return centroid


def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index on signal energy distribution."""
    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = ref_counts / ref_counts.sum() + 1e-8
    cur_pct = cur_counts / cur_counts.sum() + 1e-8

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_kl(ref: np.ndarray, cur: np.ndarray, n_bins: int = N_BINS) -> float:
    """KL divergence D(current || reference)."""
    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    ref_hist, _ = np.histogram(ref, bins=bins, density=True)
    cur_hist, _ = np.histogram(cur, bins=bins, density=True)

    ref_p = ref_hist / (ref_hist.sum() + 1e-8) + 1e-8
    cur_p = cur_hist / (cur_hist.sum() + 1e-8) + 1e-8

    return float(np.sum(cur_p * np.log(cur_p / ref_p)))


def run_drift_check(reference: np.ndarray, current: np.ndarray) -> dict:
    ref_energy = _signal_energy(reference)
    cur_energy = _signal_energy(current)
    ref_centroid = _spectral_centroid(reference)
    cur_centroid = _spectral_centroid(current)

    psi_energy = compute_psi(ref_energy, cur_energy)
    psi_centroid = compute_psi(ref_centroid, cur_centroid)
    kl_energy = compute_kl(ref_energy, cur_energy)

    # Overall PSI: max over feature dimensions (conservative — any feature drifting triggers)
    psi_overall = max(psi_energy, psi_centroid)

    status = "ok"
    if psi_overall >= PSI_ALERT:
        status = "alert"
    elif psi_overall >= PSI_WARN:
        status = "warn"

    return {
        "psi_energy": round(psi_energy, 4),
        "psi_spectral_centroid": round(psi_centroid, 4),
        "psi_overall": round(psi_overall, 4),
        "kl_energy": round(kl_energy, 4),
        "status": status,
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "sample_counts": {"reference": len(reference), "current": len(current)},
        "distribution_stats": {
            "ref_energy_mean": round(float(ref_energy.mean()), 4),
            "cur_energy_mean": round(float(cur_energy.mean()), 4),
            "ref_centroid_mean": round(float(ref_centroid.mean()), 4),
            "cur_centroid_mean": round(float(cur_centroid.mean()), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Radar classifier drift detection")
    parser.add_argument("--reference", default="data/processed/X_val.npy",
                        help="Reference distribution (validation set at training time)")
    parser.add_argument("--current", required=True,
                        help="Current batch of live inference inputs (.npy)")
    parser.add_argument("--out", default="artifacts/drift_report.json",
                        help="Output path for drift report JSON")
    args = parser.parse_args()

    reference = np.load(args.reference)
    current = np.load(args.current)

    report = run_drift_check(reference, current)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    status = report["status"]
    psi = report["psi_overall"]
    kl = report["kl_energy"]

    print(f"PSI (overall): {psi:.4f}  KL (energy): {kl:.4f}  Status: {status.upper()}")
    print(f"Report written to {args.out}")

    if status == "alert":
        print(
            "\nDRIFT ALERT: PSI > 0.20 — significant distribution shift detected.\n"
            "Recommended actions:\n"
            "  1. Review recent radar operating conditions (new environment, hardware change?)\n"
            "  2. Collect labelled samples from the new distribution\n"
            "  3. Re-run training pipeline and compare val_accuracy\n"
            "  4. Promote new model version via scripts/register_model.py"
        )
        sys.exit(2)
    elif status == "warn":
        print("\nDRIFT WARNING: PSI 0.10–0.20 — moderate shift, monitor closely.")
        sys.exit(1)


if __name__ == "__main__":
    main()
