"""
1D signal comparison: ML (ONNX MLP) vs classical MTI detector.

Computes ROC curves and SNR-sweep accuracy for both detectors on the same
synthetic 1D radar signals, giving the missing symmetric 1D baseline.

The 1D ML model (artifacts/model.onnx) has FFT baked in and takes raw
time-domain signals.  The MTI detector (src/baseline/cfar.py) operates on
magnitude FFT spectra — so both receive the same raw signal and the FFT
is applied appropriately for each.

Outputs (written to artifacts/):
  roc_comparison_1d.png     — ROC curves (ML vs MTI) at one or more SNR values
  snr_comparison_1d.png     — Pd / accuracy vs SNR for both detectors
  roc_comparison_1d.json    — AUC values and threshold curves

Usage:
    python scripts/roc_comparison.py
    python scripts/roc_comparison.py --roc-snr 5 10 15 --n-samples 500
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.baseline.cfar import MTIThresholdDetector
from src.data.generate import generate_clutter, generate_target


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_session(path: str = "artifacts/model.onnx"):
    p = Path(path)
    if not p.exists():
        print(f"WARNING: 1D ONNX model not found at {path}. "
              "Run 'python src/training/train.py' then 'python scripts/export_onnx.py' first.")
        return None
    return ort.InferenceSession(str(p))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _ml_scores(session, signals: np.ndarray) -> np.ndarray:
    """Return P(target) for a batch of raw time-domain signals (n, signal_length)."""
    logits = session.run(None, {session.get_inputs()[0].name: signals.astype(np.float32)})[0]
    return _softmax(logits)[:, 1]


def _mti_scores(signals: np.ndarray) -> np.ndarray:
    """Return MTI peak-to-mean ratio for a batch of raw signals."""
    detector = MTIThresholdDetector()
    spectra  = np.abs(np.fft.rfft(signals, axis=1))
    return detector.score_batch(spectra)


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    lo, hi = scores.min(), scores.max()
    thrs = np.linspace(hi + 1e-6, lo - 1e-6, 400)
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    fprs, tprs = [], []
    for t in thrs:
        pred = scores >= t
        tprs.append((pred & (labels == 1)).sum() / max(pos, 1))
        fprs.append((pred & (labels == 0)).sum() / max(neg, 1))
    fpr = np.array(fprs, dtype=float)
    tpr = np.array(tprs, dtype=float)
    o   = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[o], fpr[o]))
    return fpr, tpr, auc


# ── Core evaluation functions ─────────────────────────────────────────────────

def eval_roc_at_snr(
    session,
    signal_length: int,
    snr_db: float,
    n_samples: int = 500,
    seed: int = 77,
) -> dict:
    """
    Generate n_samples balanced target/clutter signals at a fixed SNR,
    then compute ROC for both ML and MTI detectors.
    """
    rng = np.random.default_rng(seed)
    n_each = n_samples // 2

    targets  = np.stack([generate_target(signal_length, snr_db, rng)  for _ in range(n_each)])
    clutters = np.stack([generate_clutter(signal_length, snr_db, rng) for _ in range(n_each)])
    X = np.concatenate([targets, clutters], axis=0).astype(np.float32)
    y = np.array([1] * n_each + [0] * n_each, dtype=int)

    mti_sc = _mti_scores(X)

    if session is not None:
        ml_sc = _ml_scores(session, X)
        ml_fpr, ml_tpr, ml_auc = _roc_curve(ml_sc, y)
    else:
        ml_fpr, ml_tpr, ml_auc = np.array([0., 1.]), np.array([0., 1.]), 0.5

    mti_fpr, mti_tpr, mti_auc = _roc_curve(mti_sc, y)

    return {
        "snr_db":  snr_db,
        "ml_auc":  ml_auc,
        "mti_auc": mti_auc,
        "ml_fpr":  ml_fpr.tolist(),
        "ml_tpr":  ml_tpr.tolist(),
        "mti_fpr": mti_fpr.tolist(),
        "mti_tpr": mti_tpr.tolist(),
        "n_samples": len(X),
    }


def eval_snr_sweep(
    session,
    signal_length: int,
    snr_min: float = -10.0,
    snr_max: float = 25.0,
    n_steps: int   = 36,
    n_samples: int = 200,
    seed: int      = 99,
) -> dict:
    """
    Sweep SNR and measure detection accuracy (at each detector's own fixed threshold)
    plus detection probability at a fixed low Pfa operating point.

    For ML: threshold = 0.5 on P(target).
    For MTI: threshold = DEFAULT_THRESHOLD (tuned on validation set at 10 dB).

    Also reports Pd at a matched Pfa (~5%) for a fair comparison.
    """
    rng      = np.random.default_rng(seed)
    snr_vals = np.linspace(snr_min, snr_max, n_steps)
    mti_det  = MTIThresholdDetector()

    ml_accs, mti_accs         = [], []
    ml_pd_5pfa, mti_pd_5pfa   = [], []

    print(f"  SNR sweep: {n_steps} points × {n_samples} samples/class...")
    for i, snr in enumerate(snr_vals):
        if i % 9 == 0:
            print(f"    SNR {snr:+.1f} dB  [{i+1}/{n_steps}]")
        n_each = n_samples
        targets  = np.stack([generate_target(signal_length, snr, rng)  for _ in range(n_each)])
        clutters = np.stack([generate_clutter(signal_length, snr, rng) for _ in range(n_each)])
        X = np.concatenate([targets, clutters], axis=0).astype(np.float32)
        y = np.array([1] * n_each + [0] * n_each, dtype=int)

        mti_sc = _mti_scores(X)
        mti_pred = (mti_sc >= mti_det.threshold).astype(int)
        mti_accs.append(float((mti_pred == y).mean()))

        if session is not None:
            ml_sc   = _ml_scores(session, X)
            ml_pred = (ml_sc >= 0.5).astype(int)
            ml_accs.append(float((ml_pred == y).mean()))
        else:
            ml_accs.append(float("nan"))

        # Pd at 5% Pfa: find threshold on clutter scores that gives Pfa=0.05,
        # then measure Pd on target scores at that threshold.
        clutter_mti = mti_sc[n_each:]
        target_mti  = mti_sc[:n_each]
        thr_mti_5pfa = float(np.percentile(clutter_mti, 95))
        mti_pd_5pfa.append(float((target_mti >= thr_mti_5pfa).mean()))

        if session is not None:
            clutter_ml = ml_sc[n_each:]
            target_ml  = ml_sc[:n_each]
            thr_ml_5pfa = float(np.percentile(clutter_ml, 95))
            ml_pd_5pfa.append(float((target_ml >= thr_ml_5pfa).mean()))
        else:
            ml_pd_5pfa.append(float("nan"))

    return {
        "snr":          snr_vals.tolist(),
        "ml_accuracy":  ml_accs,
        "mti_accuracy": mti_accs,
        "ml_pd_5pfa":   ml_pd_5pfa,
        "mti_pd_5pfa":  mti_pd_5pfa,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_roc(roc_results: list, out_path: Path) -> None:
    n = len(roc_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, roc_results):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.500)")
        ax.plot(r["ml_fpr"],  r["ml_tpr"],  color="steelblue", lw=2.5,
                label=f"ML (1D MLP)   AUC {r['ml_auc']:.3f}")
        ax.plot(r["mti_fpr"], r["mti_tpr"], color="tab:orange", lw=2.5, ls="--",
                label=f"MTI detector  AUC {r['mti_auc']:.3f}")
        ax.set_xlabel("False Alarm Rate")
        ax.set_title(f"SNR = {r['snr_db']:+.0f} dB")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    axes[0].set_ylabel("Detection Rate")
    fig.suptitle("ROC curves — 1D ML (MLP) vs MTI spectral detector", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_snr(snr_data: dict, out_path: Path) -> None:
    snr = snr_data["snr"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(snr, [v * 100 for v in snr_data["ml_accuracy"]],
            "o-", color="steelblue", lw=2.5, ms=5, label="ML (1D MLP)")
    ax.plot(snr, [v * 100 for v in snr_data["mti_accuracy"]],
            "s--", color="tab:orange", lw=2.5, ms=5, label="MTI detector")
    ax.axhline(80, ls="--", color="red",  lw=1, alpha=0.7, label="80% gate")
    ax.axhline(50, ls=":",  color="#888", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Balanced accuracy vs SNR\n(each detector's own threshold)")
    ax.set_ylim(40, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(snr, [v * 100 for v in snr_data["ml_pd_5pfa"]],
             "o-", color="steelblue", lw=2.5, ms=5, label="ML (1D MLP)")
    ax2.plot(snr, [v * 100 for v in snr_data["mti_pd_5pfa"]],
             "s--", color="tab:orange", lw=2.5, ms=5, label="MTI detector")
    ax2.axhline(50, ls=":", color="#888", lw=1)
    ax2.axhline(90, ls=":", color="#666", lw=1)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Pd at 5% Pfa (%)")
    ax2.set_title("Detection probability vs SNR\n(matched 5% Pfa operating point)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("1D signal: ML vs MTI detector comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roc-snr",    type=float, nargs="+", default=[5.0, 10.0, 15.0],
                    help="SNR values for ROC evaluation (default: 5 10 15)")
    ap.add_argument("--n-samples",  type=int,   default=500,
                    help="Samples per class for ROC (default 500)")
    ap.add_argument("--snr-min",    type=float, default=-10.0)
    ap.add_argument("--snr-max",    type=float, default=25.0)
    ap.add_argument("--snr-steps",  type=int,   default=36)
    ap.add_argument("--sweep-samples", type=int, default=200,
                    help="Samples per class for SNR sweep accuracy (default 200)")
    ap.add_argument("--model-path", type=str, default="artifacts/model.onnx")
    ap.add_argument("--params",     type=str, default="params.yaml")
    ap.add_argument("--out-dir",    type=str, default="artifacts")
    return ap.parse_args()


def main():
    args = _parse_args()
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading 1D ONNX model...")
    session = _load_session(args.model_path)

    with open(args.params) as f:
        params = yaml.safe_load(f)
    signal_length = params["data"]["signal_length"]
    print(f"  Signal length: {signal_length} samples")

    results = {}

    # ── ROC curves at multiple SNR values ─────────────────────────────────────
    print(f"\n[1/2] ROC curves at SNR = {args.roc_snr}")
    roc_results = []
    for snr_val in args.roc_snr:
        print(f"  SNR = {snr_val:+.0f} dB ({args.n_samples} samples)...")
        r = eval_roc_at_snr(session, signal_length,
                             snr_db=snr_val, n_samples=args.n_samples)
        roc_results.append(r)
        print(f"    ML AUC: {r['ml_auc']:.3f}   MTI AUC: {r['mti_auc']:.3f}")
    results["roc"] = roc_results
    _plot_roc(roc_results, out / "roc_comparison_1d.png")

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    print(f"\n[2/2] SNR sweep ({args.snr_min}..{args.snr_max} dB, {args.snr_steps} steps)")
    snr_data = eval_snr_sweep(
        session, signal_length,
        snr_min=args.snr_min, snr_max=args.snr_max,
        n_steps=args.snr_steps, n_samples=args.sweep_samples,
    )
    results["snr_sweep"] = snr_data
    _plot_snr(snr_data, out / "snr_comparison_1d.png")

    # Print summary table
    snr_arr = np.array(snr_data["snr"])
    print("\n  Summary at key SNR points:")
    print(f"  {'SNR':>6}  {'ML acc':>8}  {'MTI acc':>9}  {'ML Pd@5%':>10}  {'MTI Pd@5%':>11}")
    for target_snr in [0.0, 5.0, 10.0, 15.0]:
        if target_snr < snr_arr.min() or target_snr > snr_arr.max():
            continue
        idx = int(np.argmin(np.abs(snr_arr - target_snr)))
        print(f"  {target_snr:>+5.0f}  "
              f"{snr_data['ml_accuracy'][idx]*100:>7.1f}%  "
              f"{snr_data['mti_accuracy'][idx]*100:>8.1f}%  "
              f"{snr_data['ml_pd_5pfa'][idx]*100:>9.1f}%  "
              f"{snr_data['mti_pd_5pfa'][idx]*100:>10.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = out / "roc_comparison_1d.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
