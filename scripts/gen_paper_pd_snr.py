"""
Add LRT and DP-TBD Pd vs SNR columns to comparison_snr.csv
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
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ppi_generator import generate_ppi_sequence, generate_clutter_only
from src.baseline.ppi_cfar_kf import PPICFARDetector

with open(ROOT / "params_ppi.yaml") as f:
    params = yaml.safe_load(f)

N_SW    = params["radar"]["n_sweeps"]
N_AZ    = params["radar"]["n_azimuths"]
N_RANGE = params["radar"]["n_ranges"]
W       = 3        # oracle window half-width
N_CAL   = 200      # clutter scenes for threshold calibration
N_TRIALS = 30      # trials per SNR point (matches existing sweep)


def _lrt(ppi):
    nf = np.percentile(ppi, 10, axis=(1, 2), keepdims=True).clip(1e-6)
    return ((ppi / nf) ** 2).sum(0)


def _tbd(ppi, max_vr=3, max_vaz=2):
    nf     = np.percentile(ppi, 10, axis=(1, 2), keepdims=True).clip(1e-6)
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
# Calibrate at the same evaluation level used for Pd:
# max score over N_SW randomly-placed oracle windows per clutter scene,
# so the false-positive rate matches how we evaluate target trials.
rng_cal2 = np.random.default_rng(42)
lrt_ref_vals, tbd_ref_vals = [], []

for _ in range(N_CAL):
    seed  = int(rng_cal2.integers(1, 1_000_000))
    ppi_c = generate_clutter_only(params, seed=seed)
    lm    = _lrt(ppi_c)
    tm    = _tbd(ppi_c)
    # Simulate a "fake target path" of N_SW random positions
    lrt_path_max = tbd_path_max = 0.0
    for _ in range(N_SW):
        az_r = int(rng_cal2.integers(0, N_AZ))
        r_r  = int(rng_cal2.integers(W, N_RANGE - W))
        lrt_path_max = max(lrt_path_max, _window_max(lm, az_r, r_r))
        tbd_path_max = max(tbd_path_max, _window_max(tm, az_r, r_r))
    lrt_ref_vals.append(lrt_path_max)
    tbd_ref_vals.append(tbd_path_max)

lrt_thr = float(np.percentile(lrt_ref_vals, 100 * (1.0 - cfar_pfa)))
tbd_thr = float(np.percentile(tbd_ref_vals, 100 * (1.0 - cfar_pfa)))
print(f"  LRT threshold = {lrt_thr:.4f}   DP-TBD threshold = {tbd_thr:.4f}")

# ── Step 3: Pd vs SNR sweep ──────────────────────────────────────────────────
snr_vals = np.linspace(-20, 40, 16)
lrt_pd   = []
tbd_pd   = []

for snr in snr_vals:
    rng = np.random.default_rng(abs(int(snr * 100)) + 2007)
    lrt_hits = tbd_hits = 0

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

        # Oracle: mean target position across all sweeps
        r_positions  = [tgt["range_bin"]    + sw * tgt["radial_velocity"]     for sw in range(N_SW)]
        az_positions = [(tgt["azimuth_deg"] + sw * tgt["tangential_velocity"]) % 360 for sw in range(N_SW)]
        r_mean  = int(np.clip(round(np.mean(r_positions)), 0, N_RANGE - 1))
        az_mean = int(round(np.mean([a / 360 * N_AZ for a in az_positions]))) % N_AZ

        lm = _lrt(ppi_t)
        tm = _tbd(ppi_t)

        # Oracle: max window score across all 10 sweep positions
        lrt_best = tbd_best = 0.0
        for sw in range(N_SW):
            r_sw  = float(tgt["range_bin"])  + sw * float(tgt["radial_velocity"])
            az_sw = (float(tgt["azimuth_deg"]) + sw * float(tgt["tangential_velocity"])) % 360.0
            az_b  = int(round(az_sw / 360 * N_AZ)) % N_AZ
            r_b   = int(np.clip(round(r_sw), 0, N_RANGE - 1))
            lrt_best = max(lrt_best, _window_max(lm, az_b, r_b))
            tbd_best = max(tbd_best, _window_max(tm, az_b, r_b))

        if lrt_best > lrt_thr:
            lrt_hits += 1
        if tbd_best > tbd_thr:
            tbd_hits += 1

    lrt_pd.append(lrt_hits / N_TRIALS)
    tbd_pd.append(tbd_hits / N_TRIALS)
    print(f"  SNR {snr:+5.1f} dB  LRT Pd={lrt_hits/N_TRIALS:.2f}  TBD Pd={tbd_hits/N_TRIALS:.2f}")

# ── Step 4: merge into CSV ───────────────────────────────────────────────────
csv_path = ROOT / "artifacts" / "comparison_snr.csv"
df = pd.read_csv(csv_path)
df["lrt_pd"]    = lrt_pd
df["dptbd_pd"]  = tbd_pd
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
ax.plot(snr_vals, df["lrt_pd"]      * 100, color="#2ecc71",  lw=2.0, ls="-.",
        marker="^", markersize=5, label="LRT (non-coh.)")
ax.plot(snr_vals, df["dptbd_pd"]    * 100, color="#e74c3c",  lw=2.0, ls="-.",
        marker="v", markersize=5, label="DP-TBD")

ax.set_xlabel("SNR (dB)", color="white")
ax.set_ylabel("Probability of Detection (%)", color="white")
ax.tick_params(colors="white")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.25, color="#666")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#444", loc="upper left")
ax.set_title("Pd vs SNR — all five detectors (30 trials/point)",
             color="white", fontsize=12)

fig.tight_layout()
out = ROOT / "artifacts" / "comparison_snr.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")

# ── Step 6: print Table I numbers ───────────────────────────────────────────
print("\nTable I — Pd at representative SNR points:")
print(f"{'SNR':>6}  {'CFAR+KF':>8}  {'CNN+KF':>8}  {'GRU+KF':>8}  {'LRT':>8}  {'DP-TBD':>8}")
for _, row in df[df["snr_db"].isin([-4, 0, 4, 8, 12])].iterrows():
    print(f"{row['snr_db']:>+6.0f}  "
          f"{row['cfar_kf_pd']*100:>7.1f}%  "
          f"{row['cnn_kf_pd']*100:>7.1f}%  "
          f"{row['gru_kf_pd']*100:>7.1f}%  "
          f"{row['lrt_pd']*100:>7.1f}%  "
          f"{row['dptbd_pd']*100:>7.1f}%")
