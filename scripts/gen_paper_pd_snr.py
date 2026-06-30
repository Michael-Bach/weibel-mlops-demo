"""
Add LRT, DP-TBD, and Transformer Pd vs SNR columns to comparison_snr.csv
and regenerate comparison_snr.png for the paper.

Evaluation protocol (cell-level, oracle-window):
  - Threshold calibrated on 200 clutter-only scenes at the same Pfa
    as CFAR (~4.66 %) — using the max score in a ±3-bin reference window.
  - 30 trials per SNR point (same count as the existing KF sweep).
  - Oracle window: ±3 bins around the target's mean position across all sweeps.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import onnxruntime as ort
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ppi_generator import generate_ppi_sequence, generate_clutter_only
from src.baseline.ppi_cfar_kf import PPICFARDetector

_tf_path   = ROOT / "artifacts/transformer_model.onnx"
tf_session = ort.InferenceSession(str(_tf_path)) if _tf_path.exists() else None
print(f"Transformer session: {'loaded' if tf_session else 'NOT FOUND'}")

with open(ROOT / "params_ppi.yaml") as f:
    params = yaml.safe_load(f)

N_SW    = params["radar"]["n_sweeps"]
N_AZ    = params["radar"]["n_azimuths"]
N_RANGE = params["radar"]["n_ranges"]
W       = 3        # oracle window half-width
N_CAL   = 200      # clutter scenes for threshold calibration
N_TRIALS = 30      # trials per SNR point (matches existing sweep)


def _tbd(ppi, max_vr=3, max_vaz=2):
    # axis=(0,1): per-range-bin noise floor (range-invariant, consistent with detection.py)
    nf     = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-6)
    normed = (ppi / nf).astype(np.float32)
    S      = normed[0].copy()
    size   = (2 * max_vaz + 1, 2 * max_vr + 1)
    for k in range(1, len(normed)):
        S = normed[k] + maximum_filter(S, size=size, mode=("wrap", "nearest"))
    return S


def _window_max(score_map, az_b, r_b):
    az_lo = (az_b - W) % N_AZ
    az_hi = (az_b + W + 1) % N_AZ
    r_lo  = max(0, r_b - W)
    r_hi  = min(N_RANGE, r_b + W + 1)
    if az_lo < az_hi:
        win = score_map[az_lo:az_hi, r_lo:r_hi]
    else:
        win = np.concatenate([score_map[az_lo:, r_lo:r_hi],
                              score_map[:az_hi, r_lo:r_hi]], axis=0)
    return float(win.max()) if win.size else 0.0


def _lrt_path(ppi, az_path, r_path):
    """Path-integrated LRT: sum per-sweep window-max of squared normed amplitudes."""
    nf     = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-6)
    normed = ppi / nf  # (N_SW, N_AZ, N_RANGE)
    score  = 0.0
    for sw, (az_b, r_b) in enumerate(zip(az_path, r_path)):
        az_lo = (az_b - W) % N_AZ
        az_hi = (az_b + W + 1) % N_AZ
        r_lo  = max(0, r_b - W)
        r_hi  = min(N_RANGE, r_b + W + 1)
        if az_lo < az_hi:
            patch = normed[sw, az_lo:az_hi, r_lo:r_hi]
        else:
            patch = np.concatenate([normed[sw, az_lo:, r_lo:r_hi],
                                    normed[sw, :az_hi, r_lo:r_hi]])
        score += float((patch ** 2).max()) if patch.size else 0.0
    return score


# ── Step 1: measure CFAR Pfa on clutter-only scenes ─────────────────────────
print("Measuring CFAR Pfa …")
cfar = PPICFARDetector(threshold_factor=2.5)
rng_cal = np.random.default_rng(42)
cfar_fa_cells = 0
for _ in range(N_CAL):
    seed  = int(rng_cal.integers(1, 1_000_000))
    ppi_c = generate_clutter_only(params, seed=seed)
    cfar_fa_cells += int(cfar.detect_sequence(ppi_c).sum())
cfar_pfa = cfar_fa_cells / (N_CAL * N_SW * N_AZ * N_RANGE)
print(f"  CFAR Pfa = {cfar_pfa:.4f}")

# ── Step 2: calibrate LRT and DP-TBD thresholds at matched Pfa ──────────────
print("Calibrating LRT / DP-TBD thresholds …")
# LRT calibration: path-integral score over N_SW independent random positions,
# matching the per-sweep integration used in detection.
# TBD calibration: score at a single random endpoint, matching detection readout.
rng_cal2 = np.random.default_rng(42)
lrt_ref_vals, tbd_ref_vals = [], []

for _ in range(N_CAL):
    seed    = int(rng_cal2.integers(1, 1_000_000))
    ppi_c   = generate_clutter_only(params, seed=seed)
    tm      = _tbd(ppi_c)
    az_path = [int(rng_cal2.integers(0, N_AZ))     for _ in range(N_SW)]
    r_path  = [int(rng_cal2.integers(W, N_RANGE-W)) for _ in range(N_SW)]
    lrt_ref_vals.append(_lrt_path(ppi_c, az_path, r_path))
    az_end  = int(rng_cal2.integers(0, N_AZ))
    r_end   = int(rng_cal2.integers(W, N_RANGE - W))
    tbd_ref_vals.append(_window_max(tm, az_end, r_end))

lrt_thr = float(np.percentile(lrt_ref_vals, 100 * (1.0 - cfar_pfa)))
tbd_thr = float(np.percentile(tbd_ref_vals, 100 * (1.0 - cfar_pfa)))

# Transformer threshold: calibrated the same way
if tf_session is not None:
    tf_ref_vals = []
    rng_cal3 = np.random.default_rng(42)
    for _ in range(N_CAL):
        seed  = int(rng_cal3.integers(1, 1_000_000))
        ppi_c = generate_clutter_only(params, seed=seed)
        nf    = np.percentile(ppi_c, 10, axis=(0, 1), keepdims=True).clip(1e-3)
        stack = (ppi_c / nf).clip(0, 30).astype(np.float32)[np.newaxis]
        tf_map = tf_session.run(None, {"ppi_stack": stack})[0][0]
        tf_path_max = 0.0
        for _ in range(N_SW):
            az_r = int(rng_cal3.integers(0, N_AZ))
            r_r  = int(rng_cal3.integers(W, N_RANGE - W))
            tf_path_max = max(tf_path_max, _window_max(tf_map, az_r, r_r))
        tf_ref_vals.append(tf_path_max)
    tf_thr = float(np.percentile(tf_ref_vals, 100 * (1.0 - cfar_pfa)))
else:
    tf_thr = 0.5
print(f"  LRT thr={lrt_thr:.3f}  TBD thr={tbd_thr:.3f}  TF thr={tf_thr:.3f}")

# ── Step 3: Pd vs SNR sweep ──────────────────────────────────────────────────
snr_vals = np.linspace(-20, 40, 16)
lrt_pd, tbd_pd, tf_pd = [], [], []

for snr in snr_vals:
    rng = np.random.default_rng(abs(int(snr * 100)) + 2007)
    lrt_hits = tbd_hits = tf_hits = 0

    for _ in range(N_TRIALS):
        seed = int(rng.integers(1, 1_000_000))
        p = {
            "radar": params["radar"],
            "target": dict(
                snr_db=float(snr),
                range_bin=float(rng.integers(10, N_RANGE - 10)),
                azimuth_deg=float(rng.uniform(0, 360)),
                radial_velocity=float(rng.uniform(-3, 3)),
                tangential_velocity=float(rng.uniform(-4, 4)),
            ),
        }
        ppi_t, _, _ = generate_ppi_sequence(p, seed=seed)
        tgt = p["target"]

        if tf_session is not None:
            nf     = np.percentile(ppi_t, 10, axis=(0, 1), keepdims=True).clip(1e-3)
            stack  = (ppi_t / nf).clip(0, 30).astype(np.float32)[np.newaxis]
            tf_map = tf_session.run(None, {"ppi_stack": stack})[0][0]
        else:
            tf_map = None

        # Precompute oracle path bins for all three algorithms
        az_path, r_path = [], []
        for sw in range(N_SW):
            r_sw  = float(tgt["range_bin"]) + sw * float(tgt["radial_velocity"])
            az_sw = (float(tgt["azimuth_deg"]) + sw * float(tgt["tangential_velocity"])) % 360.0
            az_path.append(int(round(az_sw / 360 * N_AZ)) % N_AZ)
            r_path.append(int(np.clip(round(r_sw), 0, N_RANGE - 1)))

        # LRT: path-integrated score (sum per-sweep window-max along oracle path)
        lrt_best = _lrt_path(ppi_t, az_path, r_path)

        # TBD: score at final oracle position in DP map
        tm       = _tbd(ppi_t)
        tbd_best = _window_max(tm, az_path[-1], r_path[-1])

        tf_best = 0.0
        if tf_map is not None:
            for sw in range(N_SW):
                tf_best = max(tf_best, _window_max(tf_map, az_path[sw], r_path[sw]))

        if lrt_best > lrt_thr:
            lrt_hits += 1
        if tbd_best > tbd_thr:
            tbd_hits += 1
        if tf_best > tf_thr:
            tf_hits += 1

    lrt_pd.append(lrt_hits / N_TRIALS)
    tbd_pd.append(tbd_hits / N_TRIALS)
    tf_pd.append(tf_hits   / N_TRIALS)
    print(f"  SNR {snr:+5.1f} dB  LRT={lrt_hits/N_TRIALS:.2f}  TBD={tbd_hits/N_TRIALS:.2f}  TF={tf_hits/N_TRIALS:.2f}")

# ── Step 4: merge into CSV ───────────────────────────────────────────────────
csv_path = ROOT / "artifacts" / "comparison_snr.csv"
df = pd.read_csv(csv_path)
df["lrt_pd"]         = lrt_pd
df["dptbd_pd"]       = tbd_pd
df["transformer_pd"] = tf_pd
df.to_csv(csv_path, index=False)
print(f"Saved updated CSV: {csv_path}")

# ── Step 5: regenerate comparison_snr.png ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
GRID_C = "rgba(128,128,128,0.2)"

ax.plot(snr_vals, df["cfar_kf_pd"]  * 100, color="white",   lw=2.0, ls="--",
        marker="o", markersize=5, label="CA-CFAR + KF")
ax.plot(snr_vals, df["cnn_kf_pd"]   * 100, color="#ffe66d",  lw=2.5,
        marker="s", markersize=5, label="CNN + KF")
ax.plot(snr_vals, df["gru_kf_pd"]   * 100, color="#c084fc",  lw=3.0,
        marker="D", markersize=6, label="ConvGRU + KF")
ax.plot(snr_vals, df["lrt_pd"]         * 100, color="#2ecc71",  lw=2.0, ls="-.",
        marker="^", markersize=5, label="LRT (non-coh.)  [cell]")
ax.plot(snr_vals, df["dptbd_pd"]       * 100, color="#e74c3c",  lw=2.0, ls="-.",
        marker="v", markersize=5, label="DP-TBD  [cell]")
ax.plot(snr_vals, df["transformer_pd"] * 100, color="#f39c12",  lw=2.0, ls="-.",
        marker="*", markersize=7, label="Transformer  [cell]")

ax.set_xlabel("SNR (dB)", color="white")
ax.set_ylabel("Probability of Detection (%)", color="white")
ax.tick_params(colors="white")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.25, color="#666")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#444", loc="upper left")
ax.set_title("Pd vs SNR — all six detectors (30 trials/point;\n"
             "solid = confirmed track, dash-dot = cell-level)",
             color="white", fontsize=11)

fig.tight_layout()
out = ROOT / "artifacts" / "comparison_snr.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")

# ── Step 6: print Table I numbers ───────────────────────────────────────────
print("\nTable I — Pd at representative SNR points:")
print(f"{'SNR':>6}  {'CFAR+KF':>8}  {'CNN+KF':>8}  {'GRU+KF':>8}  {'LRT':>8}  {'DP-TBD':>8}  {'TF':>8}")
for _, row in df[df["snr_db"].isin([-4, 0, 4, 8, 12])].iterrows():
    print(f"{row['snr_db']:>+6.0f}  "
          f"{row['cfar_kf_pd']*100:>7.1f}%  "
          f"{row['cnn_kf_pd']*100:>7.1f}%  "
          f"{row['gru_kf_pd']*100:>7.1f}%  "
          f"{row['lrt_pd']*100:>7.1f}%  "
          f"{row['dptbd_pd']*100:>7.1f}%  "
          f"{row['transformer_pd']*100:>7.1f}%")
