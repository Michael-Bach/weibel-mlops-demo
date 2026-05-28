"""Generate ROC curves for the paper: CFAR vs CNN vs GRU at SNR = 0, 3, 6 dB."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.data.ppi_generator import (
    generate_ppi_sequence, generate_clutter_only, temporal_features
)
from src.baseline.ppi_cfar_kf import PPICFARDetector

N_TRIALS = 400  # half target, half clutter
SNR_LIST  = [0.0, 3.0, 6.0]
SEED_BASE = 42

with open(ROOT / "params_ppi.yaml") as f:
    params = yaml.safe_load(f)

N_SW    = params["radar"]["n_sweeps"]
N_AZ    = params["radar"]["n_azimuths"]
N_RANGE = params["radar"]["n_ranges"]

cnn_session = ort.InferenceSession(str(ROOT / "artifacts/ppi_model.onnx"))
gru_session = ort.InferenceSession(str(ROOT / "artifacts/recurrent_model.onnx"))


def _roc(scores, labels):
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for _, lab in paired:
        if lab:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    tprs.append(1.0); fprs.append(1.0)
    auc = sum(
        (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )
    return fprs, tprs, auc


def collect_scores(snr_db: float):
    rng    = np.random.default_rng(SEED_BASE + int(snr_db * 10))
    cfar_d = PPICFARDetector()
    cnn_sc, gru_sc, cfar_sc, labs = [], [], [], []

    for i in range(N_TRIALS):
        has_t = (i % 2 == 0)
        seed  = int(rng.integers(1, 1_000_000))

        if has_t:
            p = {
                "radar": params["radar"],
                "target": dict(
                    snr_db=snr_db,
                    range_bin=float(rng.integers(10, N_RANGE - 10)),
                    azimuth_deg=float(rng.uniform(0, 360)),
                    radial_velocity=float(rng.uniform(-2, 2)),
                    tangential_velocity=float(rng.uniform(-3, 3)),
                ),
            }
            ppi, _, _ = generate_ppi_sequence(p, seed=seed)
            tgt_az = (p["target"]["azimuth_deg"] / 360 * N_AZ) % N_AZ
            tgt_r  = p["target"]["range_bin"]
        else:
            ppi = generate_clutter_only(params, seed=seed)
            p   = None

        # CNN score — max probability in 5-bin neighbourhood of true position
        feat     = temporal_features(ppi)[np.newaxis].astype(np.float32)
        cnn_map  = cnn_session.run(None, {"ppi_sequence": feat})[0][0]

        # GRU score — accumulate hidden state over all sweeps, take final map
        h = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
        for sw in range(N_SW):
            sweep_norm = ppi[sw][np.newaxis, np.newaxis].astype(np.float32)
            gru_out    = gru_session.run(None, {"sweep_norm": sweep_norm, "h_in": h})
            h = gru_out[1]
        gru_map = h[0, 0]

        # CFAR score — fraction of sweeps with a detection anywhere
        detect_seq  = cfar_d.detect_sequence(ppi)
        cfar_any    = float(np.mean([d.any() for d in detect_seq]))

        # Sample a test window: target position for target trials, random for clutter
        W = 5  # half-width in bins
        if has_t:
            az_i = int(round(tgt_az)) % N_AZ
            r_i  = int(np.clip(round(tgt_r), W, N_RANGE - W - 1))
        else:
            az_i = int(rng.integers(0, N_AZ))
            r_i  = int(rng.integers(W, N_RANGE - W))

        r_lo  = max(0, r_i - W); r_hi = min(N_RANGE, r_i + W + 1)
        az_idx = np.arange(az_i - W, az_i + W + 1) % N_AZ

        cnn_sc.append(float(cnn_map[az_idx, r_lo:r_hi].max()))
        gru_sc.append(float(gru_map[az_idx, r_lo:r_hi].max()))

        cfar_near = sum(
            1 for sw in range(N_SW) if detect_seq[sw][az_idx, r_lo:r_hi].any()
        )
        cfar_sc.append(cfar_near / N_SW)

        labs.append(int(has_t))

    return cnn_sc, gru_sc, cfar_sc, labs


fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
fig.suptitle(
    "ROC curves — CNN+KF, ConvGRU+KF, CA-CFAR  (PPI detector scores)",
    fontsize=12, y=1.01
)

for ax, snr in zip(axes, SNR_LIST):
    print(f"SNR = {snr:+.0f} dB ...")
    cnn_sc, gru_sc, cfar_sc, labs = collect_scores(snr)

    cf_fpr,  cf_tpr,  cf_auc  = _roc(cfar_sc, labs)
    cnn_fpr, cnn_tpr, cnn_auc = _roc(cnn_sc,  labs)
    gru_fpr, gru_tpr, gru_auc = _roc(gru_sc,  labs)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.500)")
    ax.plot(cf_fpr,  cf_tpr,  color="white",   lw=2.0, ls="--",
            label=f"CA-CFAR  AUC {cf_auc:.3f}")
    ax.plot(cnn_fpr, cnn_tpr, color="gold",    lw=2.0,
            label=f"CNN      AUC {cnn_auc:.3f}")
    ax.plot(gru_fpr, gru_tpr, color="magenta", lw=2.0,
            label=f"ConvGRU  AUC {gru_auc:.3f}")

    ax.set_title(f"SNR = {snr:+.0f} dB")
    ax.set_xlabel("False Alarm Rate")
    if ax is axes[0]:
        ax.set_ylabel("Detection Rate")
    ax.set_xlim(0, 0.4); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#0e1117")

fig.patch.set_facecolor("#0e1117")
for ax in axes:
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
fig.suptitle(
    "ROC curves — CNN+KF, ConvGRU+KF, CA-CFAR  (PPI detector scores)",
    fontsize=12, y=1.01, color="white"
)

fig.tight_layout()
out = ROOT / "artifacts" / "paper_roc.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")
