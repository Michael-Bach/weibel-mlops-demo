"""
Headless PPI pipeline comparison: CFAR+KF vs CNN+KF.

Ports all evaluation logic from app.py so it can run without a Streamlit server,
save results as files, and be wired into CI or called from scripts.

Outputs (written to artifacts/):
  comparison_results.json   — all scalar metrics
  comparison_snr.csv        — Pd × SNR table (4 pipelines)
  comparison_snr.png        — Pd vs SNR figure (4 pipelines)
  comparison_sweep.png      — cumulative Pd vs sweep at each --sweep-snr value
  comparison_roc.png        — ROC curves (ML CNN vs CA-CFAR)
  comparison_runtime.json   — per-operation timing (ms)

Usage:
    python scripts/compare_pipelines.py
    python scripts/compare_pipelines.py --n-trials 10 --n-scenes 50
    python scripts/compare_pipelines.py --sweep-snr -5 10 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.baseline.ppi_cfar_kf import PPICFARDetector, PPIKalmanTracker
from src.data.ppi_generator import (
    generate_clutter_only,
    generate_ppi_sequence,
    temporal_features,
)

# ── Constants derived from params ────────────────────────────────────────────

def _load_params(path: str = "params_ppi.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_session(path: str = "artifacts/ppi_model.onnx"):
    p = Path(path)
    if not p.exists():
        print(f"WARNING: ONNX model not found at {path} — CNN pipelines will be skipped.")
        return None
    return ort.InferenceSession(str(p))


def _load_gru_session(path: str = "artifacts/recurrent_model.onnx"):
    p = Path(path)
    if not p.exists():
        print(f"NOTE: GRU model not found at {path} — GRU pipeline will be skipped.")
        return None
    return ort.InferenceSession(str(p))


def _gru_step(
    sweep: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    noise_floor: np.ndarray,
    gru_session,
) -> tuple:
    """
    Advance GRU by one sweep.  Returns (prob_map, h1_next, h2_next, noise_floor_next).
    prob_map shape: (n_az, n_range).
    """
    sweep_norm = (sweep / noise_floor.clip(1e-3)).astype(np.float32)
    noise_floor_next = (0.9 * noise_floor + 0.1 * np.percentile(sweep, 10, axis=0)).clip(1e-3)
    sweep_in = sweep_norm[np.newaxis, np.newaxis]  # (1, 1, n_az, n_range)
    prob_map, h1_next, h2_next = gru_session.run(
        None, {"sweep_norm": sweep_in, "h1_in": h1, "h2_in": h2}
    )
    return prob_map[0, 0], h1_next, h2_next, noise_floor_next


def _gru_zero_state(ch1: int, ch2: int, n_az: int, n_range: int):
    h1 = np.zeros((1, ch1, n_az, n_range), dtype=np.float32)
    h2 = np.zeros((1, ch2, n_az, n_range), dtype=np.float32)
    return h1, h2


# ── Helper: CNN peak extraction ───────────────────────────────────────────────

def _cnn_peaks(ml_out: np.ndarray, threshold: float, max_peaks: int = 10) -> np.ndarray:
    """Threshold CNN probability map → top-N peaks as (r, az) array."""
    az_idxs, r_idxs = np.where(ml_out > threshold)
    valid = r_idxs >= 5
    az_idxs, r_idxs = az_idxs[valid], r_idxs[valid]
    if len(az_idxs) == 0:
        return np.empty((0, 2))
    scores = ml_out[az_idxs, r_idxs]
    order  = np.argsort(-scores)[:max_peaks]
    return np.column_stack([r_idxs[order], az_idxs[order]]).astype(float)


# ── Step 1: Operating-point calibration ──────────────────────────────────────

def calibrate_pfa(
    params: dict,
    session,
    gru_session=None,
    n_scenes: int = 200,
) -> dict:
    """
    Measure CFAR Pfa on clutter-only scenes, then find the CNN and GRU thresholds
    that produce the same cell-level false-alarm rate.

    Returns dict with:
      cfar_pfa          — CFAR Pfa per cell per sweep
      cnn_thr           — CNN threshold giving matched Pfa
      gru_thr           — GRU threshold giving matched Pfa
      cfar_kf_ft_rate   — mean confirmed false tracks per scene (CFAR+KF)
      cnn_kf_ft_rate    — mean confirmed false tracks per scene (CNN+KF)
      gru_kf_ft_rate    — mean confirmed false tracks per scene (GRU+KF)
    """
    N_SW    = params["radar"]["n_sweeps"]
    N_AZ    = params["radar"]["n_azimuths"]
    N_RANGE = params["radar"]["n_ranges"]
    GRU_CH1 = gru_session.get_inputs()[1].shape[1] if gru_session is not None else 16
    GRU_CH2 = gru_session.get_inputs()[2].shape[1] if gru_session is not None else 8

    rng  = np.random.default_rng(13)
    cfar = PPICFARDetector(threshold_factor=2.5)

    cfar_fa_cells     = 0
    cfar_kf_false_trk = 0
    cnn_vals: list    = []
    gru_vals: list    = []

    print(f"  Calibrating Pfa over {n_scenes} clutter-only scenes...")
    for i in range(n_scenes):
        if i % 50 == 0:
            print(f"    scene {i}/{n_scenes}")
        seed  = int(rng.integers(1, 1_000_000))
        ppi_c = generate_clutter_only(params, seed=seed)

        det_seq = cfar.detect_sequence(ppi_c)
        cfar_fa_cells += int(det_seq.sum())

        kf_c = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
        kf_c.process_sequence(ppi_c)
        cfar_kf_false_trk += len(kf_c.track_positions())

        if session is not None:
            for k in range(N_SW):
                feat_k = temporal_features(ppi_c[:k + 1])[np.newaxis].astype(np.float32)
                out_k  = session.run(None, {"ppi_sequence": feat_k})[0][0]
                cnn_vals.extend(out_k.ravel().tolist())

        if gru_session is not None:
            h1, h2 = _gru_zero_state(GRU_CH1, GRU_CH2, N_AZ, N_RANGE)
            nf = np.percentile(ppi_c[0], 10, axis=0).clip(1e-3)
            for k in range(N_SW):
                pm, h1, h2, nf = _gru_step(ppi_c[k], h1, h2, nf, gru_session)
                gru_vals.extend(pm.ravel().tolist())

    cfar_pfa = cfar_fa_cells / (n_scenes * N_SW * N_AZ * N_RANGE)

    if cnn_vals and session is not None:
        arr     = np.array(cnn_vals, dtype=np.float32)
        cnn_thr = float(np.clip(np.percentile(arr, 100.0 * (1.0 - cfar_pfa)), 0.01, 0.99))
    else:
        cnn_thr = 0.30

    if gru_vals and gru_session is not None:
        arr     = np.array(gru_vals, dtype=np.float32)
        gru_thr = float(np.clip(np.percentile(arr, 100.0 * (1.0 - cfar_pfa)), 0.01, 0.99))
    else:
        gru_thr = 0.30

    # False track rates (second pass, same seeds)
    rng2 = np.random.default_rng(13)
    cnn_kf_false_trk = gru_kf_false_trk = 0
    for _ in range(n_scenes):
        seed  = int(rng2.integers(1, 1_000_000))
        ppi_c = generate_clutter_only(params, seed=seed)

        if session is not None:
            kf_ml = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            for k in range(N_SW):
                feat   = temporal_features(ppi_c[:k + 1])[np.newaxis].astype(np.float32)
                ml_out = session.run(None, {"ppi_sequence": feat})[0][0]
                peaks  = _cnn_peaks(ml_out, cnn_thr)
                kf_ml.update_from_peaks(peaks, N_AZ, N_RANGE)
            cnn_kf_false_trk += len(kf_ml.track_positions())

        if gru_session is not None:
            kf_gru = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            h1, h2 = _gru_zero_state(GRU_CH1, GRU_CH2, N_AZ, N_RANGE)
            nf = np.percentile(ppi_c[0], 10, axis=0).clip(1e-3)
            for k in range(N_SW):
                pm, h1, h2, nf = _gru_step(ppi_c[k], h1, h2, nf, gru_session)
                peaks = _cnn_peaks(pm, gru_thr)
                kf_gru.update_from_peaks(peaks, N_AZ, N_RANGE)
            gru_kf_false_trk += len(kf_gru.track_positions())

    return {
        "cfar_pfa":        cfar_pfa,
        "cnn_thr":         cnn_thr,
        "gru_thr":         gru_thr,
        "cfar_kf_ft_rate": cfar_kf_false_trk / n_scenes,
        "cnn_kf_ft_rate":  cnn_kf_false_trk  / n_scenes,
        "gru_kf_ft_rate":  gru_kf_false_trk  / n_scenes,
    }


# ── Step 2: SNR sweep ─────────────────────────────────────────────────────────

def snr_sweep(
    params: dict,
    session,
    gru_session=None,
    cnn_thr: float = 0.30,
    gru_thr: float = 0.30,
    n_trials: int  = 30,
    snr_min: float = -20.0,
    snr_max: float = 40.0,
    n_points: int  = 16,
) -> dict:
    """
    Compare pipelines across n_points SNR values (n_trials each):
      1. CFAR standalone   — single-look Pd per (trial, sweep)
      2. CFAR + KF         — confirmed track within 6 bins
      3. CNN batch         — CNN on all sweeps, threshold at matched Pfa
      4. CNN  + KF         — incremental CNN feeding KF
      5. GRU  + KF         — streaming ConvGRU feeding KF (if gru_session provided)

    Returns dict with arrays: snr, cfar_sa, cfar_kf, cnn_batch, cnn_kf, gru_kf,
    cfar_kf_rmse, cnn_kf_rmse, gru_kf_rmse.
    """
    N_SW    = params["radar"]["n_sweeps"]
    N_AZ    = params["radar"]["n_azimuths"]
    N_RANGE = params["radar"]["n_ranges"]
    GRU_CH1 = gru_session.get_inputs()[1].shape[1] if gru_session is not None else 16
    GRU_CH2 = gru_session.get_inputs()[2].shape[1] if gru_session is not None else 8

    rng      = np.random.default_rng(0)
    snr_vals = np.linspace(snr_min, snr_max, n_points)
    cfar     = PPICFARDetector(threshold_factor=2.5)

    out_cfar_sa, out_cfar_kf, out_cnn_b, out_cnn_kf, out_gru_kf = [], [], [], [], []
    out_cfar_kf_rmse, out_cnn_kf_rmse, out_gru_kf_rmse = [], [], []

    print(f"  SNR sweep: {n_points} points × {n_trials} trials...")
    for si, snr in enumerate(snr_vals):
        print(f"    SNR {snr:+.1f} dB  [{si+1}/{n_points}]")
        cfar_sa_hits = 0
        cfar_kf_h = cnn_b_h = cnn_kf_h = gru_kf_h = 0
        cfar_kf_err = cnn_kf_err = gru_kf_err = 0.0
        cfar_kf_det = cnn_kf_det = gru_kf_det = 0

        for _ in range(n_trials):
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
            ppi_t, _, pos_t = generate_ppi_sequence(p, seed=seed)
            path_az = [int(az_d / 360 * N_AZ) % N_AZ for az_d, _ in pos_t]
            path_r  = [int(np.clip(r, 0, N_RANGE - 1)) for _, r in pos_t]

            # 1. CFAR standalone
            cfar_seq = cfar.detect_sequence(ppi_t)
            for t, (az_b, r_b) in enumerate(zip(path_az, path_r)):
                if cfar_seq[t, az_b, r_b]:
                    cfar_sa_hits += 1

            # 2. CFAR + KF
            kf_c = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            cfar_kf_found = False
            for k in range(N_SW):
                kf_c.update(ppi_t[k])
                if cfar_kf_found:
                    continue
                az_b_k, r_b_k = path_az[k], path_r[k]
                for tr in kf_c._tracks:
                    if tr.hits >= kf_c.min_hits:
                        az_w = float(tr.x[2]) % N_AZ
                        daz  = min(abs(az_w - az_b_k), N_AZ - abs(az_w - az_b_k))
                        dr   = abs(float(tr.x[0]) - r_b_k)
                        err  = np.sqrt(dr**2 + daz**2)
                        if err < 6:
                            cfar_kf_h   += 1
                            cfar_kf_err += err
                            cfar_kf_det += 1
                            cfar_kf_found = True
                            break

            if session is not None:
                feat   = temporal_features(ppi_t)[np.newaxis].astype(np.float32)
                ml_out = session.run(None, {"ppi_sequence": feat})[0][0]

                # 3. CNN batch
                for az_b, r_b in zip(path_az, path_r):
                    win = ml_out[max(0, az_b - 1):az_b + 2,
                                 max(0, r_b  - 1):r_b  + 2]
                    if float(win.max()) > cnn_thr:
                        cnn_b_h += 1
                        break

                # 4. CNN + KF
                kf_ml = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
                cnn_kf_found = False
                for k in range(N_SW):
                    feat_k   = temporal_features(ppi_t[:k + 1])[np.newaxis].astype(np.float32)
                    ml_out_k = session.run(None, {"ppi_sequence": feat_k})[0][0]
                    peaks    = _cnn_peaks(ml_out_k, cnn_thr)
                    kf_ml.update_from_peaks(peaks, N_AZ, N_RANGE)
                    if cnn_kf_found:
                        continue
                    az_b_k, r_b_k = path_az[k], path_r[k]
                    for tr in kf_ml._tracks:
                        if tr.hits >= kf_ml.min_hits:
                            az_w = float(tr.x[2]) % N_AZ
                            daz  = min(abs(az_w - az_b_k), N_AZ - abs(az_w - az_b_k))
                            dr   = abs(float(tr.x[0]) - r_b_k)
                            err  = np.sqrt(dr**2 + daz**2)
                            if err < 6:
                                cnn_kf_h   += 1
                                cnn_kf_err += err
                                cnn_kf_det += 1
                                cnn_kf_found = True
                                break

            # 5. GRU + KF (streaming)
            if gru_session is not None:
                kf_gru = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
                h1, h2 = _gru_zero_state(GRU_CH1, GRU_CH2, N_AZ, N_RANGE)
                nf = np.percentile(ppi_t[0], 10, axis=0).clip(1e-3)
                gru_kf_found = False
                for k in range(N_SW):
                    pm, h1, h2, nf = _gru_step(ppi_t[k], h1, h2, nf, gru_session)
                    peaks = _cnn_peaks(pm, gru_thr)
                    kf_gru.update_from_peaks(peaks, N_AZ, N_RANGE)
                    if gru_kf_found:
                        continue
                    az_b_k, r_b_k = path_az[k], path_r[k]
                    for tr in kf_gru._tracks:
                        if tr.hits >= kf_gru.min_hits:
                            az_w = float(tr.x[2]) % N_AZ
                            daz  = min(abs(az_w - az_b_k), N_AZ - abs(az_w - az_b_k))
                            dr   = abs(float(tr.x[0]) - r_b_k)
                            err  = np.sqrt(dr**2 + daz**2)
                            if err < 6:
                                gru_kf_h   += 1
                                gru_kf_err += err
                                gru_kf_det += 1
                                gru_kf_found = True
                                break

        out_cfar_sa.append(cfar_sa_hits / (n_trials * N_SW))
        out_cfar_kf.append(cfar_kf_h / n_trials)
        out_cnn_b.append(cnn_b_h      / n_trials)
        out_cnn_kf.append(cnn_kf_h    / n_trials)
        out_gru_kf.append(gru_kf_h    / n_trials)
        out_cfar_kf_rmse.append(cfar_kf_err / max(cfar_kf_det, 1))
        out_cnn_kf_rmse.append(cnn_kf_err   / max(cnn_kf_det,  1))
        out_gru_kf_rmse.append(gru_kf_err   / max(gru_kf_det,  1))

    return {
        "snr":           snr_vals.tolist(),
        "cfar_sa":       out_cfar_sa,
        "cfar_kf":       out_cfar_kf,
        "cnn_batch":     out_cnn_b,
        "cnn_kf":        out_cnn_kf,
        "gru_kf":        out_gru_kf,
        "cfar_kf_rmse":  out_cfar_kf_rmse,
        "cnn_kf_rmse":   out_cnn_kf_rmse,
        "gru_kf_rmse":   out_gru_kf_rmse,
    }


# ── Step 3: Per-sweep cumulative Pd ──────────────────────────────────────────

def sweep_profile(
    params: dict,
    session,
    gru_session=None,
    snr_db: float  = 10.0,
    cnn_thr: float = 0.30,
    gru_thr: float = 0.30,
    n_trials: int  = 30,
) -> dict:
    """
    Cumulative Pd vs sweep index for three pipelines at a fixed SNR.
    Also returns mean track-initiation latency for both KF pipelines.
    """
    N_SW    = params["radar"]["n_sweeps"]
    N_AZ    = params["radar"]["n_azimuths"]
    N_RANGE = params["radar"]["n_ranges"]
    GRU_CH1 = gru_session.get_inputs()[1].shape[1] if gru_session is not None else 16
    GRU_CH2 = gru_session.get_inputs()[2].shape[1] if gru_session is not None else 8

    rng  = np.random.default_rng(99)
    cfar = PPICFARDetector(threshold_factor=2.5)

    cfar_sa_cum = np.zeros(N_SW)
    cfar_kf_cum = np.zeros(N_SW)
    cnn_kf_cum  = np.zeros(N_SW)
    gru_kf_cum  = np.zeros(N_SW)
    cfar_kf_lat: list = []
    cnn_kf_lat:  list = []
    gru_kf_lat:  list = []

    for _ in range(n_trials):
        seed = int(rng.integers(1, 1_000_000))
        p = {
            "radar": params["radar"],
            "target": dict(
                snr_db=float(snr_db),
                range_bin=float(rng.integers(10, N_RANGE - 10)),
                azimuth_deg=float(rng.uniform(0, 360)),
                radial_velocity=float(rng.uniform(-3, 3)),
                tangential_velocity=float(rng.uniform(-4, 4)),
            ),
        }
        ppi_t, _, pos_t = generate_ppi_sequence(p, seed=seed)
        path_az = [int(az_d / 360 * N_AZ) % N_AZ for az_d, _ in pos_t]
        path_r  = [int(np.clip(r, 0, N_RANGE - 1)) for _, r in pos_t]

        # CFAR standalone
        cfar_seq = cfar.detect_sequence(ppi_t)
        cfar_hit_ever = False
        for k in range(N_SW):
            if cfar_seq[k, path_az[k], path_r[k]]:
                cfar_hit_ever = True
            if cfar_hit_ever:
                cfar_sa_cum[k] += 1

        # CFAR + KF
        kf_c = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
        cfar_kf_ever = False
        cfar_kf_lat_trial = None
        for k in range(N_SW):
            kf_c.update(ppi_t[k])
            if cfar_kf_ever:
                cfar_kf_cum[k] += 1
                continue
            az_b_k, r_b_k = path_az[k], path_r[k]
            for tr in kf_c._tracks:
                if tr.hits >= kf_c.min_hits:
                    dr  = abs(float(tr.x[0]) - r_b_k)
                    daz = min(abs(float(tr.x[2]) % N_AZ - az_b_k),
                              N_AZ - abs(float(tr.x[2]) % N_AZ - az_b_k))
                    if np.sqrt(dr**2 + daz**2) < 6:
                        cfar_kf_ever = True
                        cfar_kf_lat_trial = k + 1
                        cfar_kf_cum[k] += 1
                        break
        if cfar_kf_lat_trial is not None:
            cfar_kf_lat.append(cfar_kf_lat_trial)

        # CNN + KF
        if session is not None:
            kf_ml = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            cnn_kf_ever = False
            cnn_kf_lat_trial = None
            for k in range(N_SW):
                feat_k   = temporal_features(ppi_t[:k + 1])[np.newaxis].astype(np.float32)
                ml_out_k = session.run(None, {"ppi_sequence": feat_k})[0][0]
                peaks    = _cnn_peaks(ml_out_k, cnn_thr)
                kf_ml.update_from_peaks(peaks, N_AZ, N_RANGE)
                if cnn_kf_ever:
                    cnn_kf_cum[k] += 1
                    continue
                az_b_k, r_b_k = path_az[k], path_r[k]
                for tr in kf_ml._tracks:
                    if tr.hits >= kf_ml.min_hits:
                        dr  = abs(float(tr.x[0]) - r_b_k)
                        daz = min(abs(float(tr.x[2]) % N_AZ - az_b_k),
                                  N_AZ - abs(float(tr.x[2]) % N_AZ - az_b_k))
                        if np.sqrt(dr**2 + daz**2) < 6:
                            cnn_kf_ever = True
                            cnn_kf_lat_trial = k + 1
                            cnn_kf_cum[k] += 1
                            break
            if cnn_kf_lat_trial is not None:
                cnn_kf_lat.append(cnn_kf_lat_trial)

        # GRU + KF (streaming)
        if gru_session is not None:
            kf_gru = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            h1, h2 = _gru_zero_state(GRU_CH1, GRU_CH2, N_AZ, N_RANGE)
            nf = np.percentile(ppi_t[0], 10, axis=0).clip(1e-3)
            gru_kf_ever = False
            gru_kf_lat_trial = None
            for k in range(N_SW):
                pm, h1, h2, nf = _gru_step(ppi_t[k], h1, h2, nf, gru_session)
                peaks = _cnn_peaks(pm, gru_thr)
                kf_gru.update_from_peaks(peaks, N_AZ, N_RANGE)
                if gru_kf_ever:
                    gru_kf_cum[k] += 1
                    continue
                az_b_k, r_b_k = path_az[k], path_r[k]
                for tr in kf_gru._tracks:
                    if tr.hits >= kf_gru.min_hits:
                        dr  = abs(float(tr.x[0]) - r_b_k)
                        daz = min(abs(float(tr.x[2]) % N_AZ - az_b_k),
                                  N_AZ - abs(float(tr.x[2]) % N_AZ - az_b_k))
                        if np.sqrt(dr**2 + daz**2) < 6:
                            gru_kf_ever = True
                            gru_kf_lat_trial = k + 1
                            gru_kf_cum[k] += 1
                            break
            if gru_kf_lat_trial is not None:
                gru_kf_lat.append(gru_kf_lat_trial)

    return {
        "snr_db":          snr_db,
        "sweeps":          list(range(1, N_SW + 1)),
        "cfar_sa_cum_pct": (cfar_sa_cum / n_trials * 100).tolist(),
        "cfar_kf_cum_pct": (cfar_kf_cum / n_trials * 100).tolist(),
        "cnn_kf_cum_pct":  (cnn_kf_cum  / n_trials * 100).tolist(),
        "gru_kf_cum_pct":  (gru_kf_cum  / n_trials * 100).tolist(),
        "cfar_kf_latency": float(np.mean(cfar_kf_lat)) if cfar_kf_lat else None,
        "cnn_kf_latency":  float(np.mean(cnn_kf_lat))  if cnn_kf_lat  else None,
        "gru_kf_latency":  float(np.mean(gru_kf_lat))  if gru_kf_lat  else None,
    }


# ── Step 4: ROC curves ────────────────────────────────────────────────────────

def _cfar_path_score(detect_seq: np.ndarray, t_params: dict, N_AZ: int) -> float:
    n_sw, _, n_range = detect_seq.shape
    tgt = t_params["target"]
    r0, az0 = float(tgt["range_bin"]), float(tgt["azimuth_deg"])
    vr, vaz  = float(tgt["radial_velocity"]), float(tgt["tangential_velocity"])
    hits = 0
    for sw in range(n_sw):
        r_t   = r0 + sw * vr
        az_t  = (az0 + sw * vaz) % 360.0
        r_bin = int(np.round(r_t).clip(0, n_range - 1))
        az_bin = int(np.round(az_t * N_AZ / 360.0)) % N_AZ
        for daz in (-1, 0, 1):
            for dr in (-1, 0, 1):
                az_c = (az_bin + daz) % N_AZ
                r_c  = int(np.clip(r_bin + dr, 0, n_range - 1))
                if detect_seq[sw, az_c, r_c]:
                    hits += 1
                    break
            else:
                continue
            break
    return hits / n_sw


def roc_data(
    params: dict,
    session,
    snr_db: float = 10.0,
    n: int = 100,
) -> dict:
    """
    Build scores for ROC comparison across n trials (n/2 target, n/2 clutter).

    Returns dict with:
      ml_scores, cfar_scores, labels (all lists of length n)
    """
    N_SW    = params["radar"]["n_sweeps"]
    N_AZ    = params["radar"]["n_azimuths"]
    N_RANGE = params["radar"]["n_ranges"]

    rng    = np.random.default_rng(77)
    cfar_d = PPICFARDetector()
    ml_sc, cfar_sc, lbs = [], [], []

    print(f"  ROC data: {n} trials at SNR={snr_db:+.0f} dB...")
    for i in range(n):
        has_t = (i % 2 == 0)
        seed  = int(rng.integers(1, 1_000_000))

        if has_t:
            p = {
                "radar": params["radar"],
                "target": dict(
                    snr_db=snr_db,
                    range_bin=float(rng.integers(10, N_RANGE - 10)),
                    azimuth_deg=float(rng.uniform(0, 360)),
                    radial_velocity=float(rng.uniform(-3, 3)),
                    tangential_velocity=float(rng.uniform(-4, 4)),
                ),
            }
            ppi_t, _, _ = generate_ppi_sequence(p, seed=seed)
        else:
            ppi_t = generate_clutter_only(params, seed=seed)

        if session is not None:
            feat   = temporal_features(ppi_t)[np.newaxis].astype(np.float32)
            ml_out = session.run(None, {"ppi_sequence": feat})[0][0]
        else:
            ml_out = np.zeros((N_AZ, N_RANGE), dtype=np.float32)

        detect_seq = cfar_d.detect_sequence(ppi_t)

        if has_t:
            tgt   = p["target"]
            r0_t  = float(tgt["range_bin"])
            az0_t = float(tgt["azimuth_deg"])
            vr_t  = float(tgt["radial_velocity"])
            vaz_t = float(tgt["tangential_velocity"])

            cfar_sc.append(_cfar_path_score(detect_seq, p, N_AZ))

            ml_path_max = 0.0
            for sw in range(N_SW):
                r_t  = r0_t + sw * vr_t
                az_t = (az0_t + sw * vaz_t) % 360.0
                r_b  = int(np.round(r_t).clip(0, N_RANGE - 1))
                az_b = int(np.round(az_t * N_AZ / 360.0)) % N_AZ
                win  = ml_out[max(0, az_b - 1):az_b + 2,
                               max(0, r_b  - 1):r_b  + 2]
                ml_path_max = max(ml_path_max,
                                  float(win.max()) if win.size else 0.0)
            ml_sc.append(ml_path_max)
        else:
            r_rand  = int(rng.integers(10, N_RANGE - 10))
            az_rand = int(rng.integers(0, N_AZ))
            hits = 0
            for sw in range(N_SW):
                for daz in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        az_c = (az_rand + daz) % N_AZ
                        r_c  = int(np.clip(r_rand + dr, 0, N_RANGE - 1))
                        if detect_seq[sw, az_c, r_c]:
                            hits += 1
                            break
                    else:
                        continue
                    break
            cfar_sc.append(hits / N_SW)
            win = ml_out[max(0, az_rand - 1):az_rand + 2,
                         max(0, r_rand  - 1):r_rand  + 2]
            ml_sc.append(float(win.max()) if win.size else 0.0)

        lbs.append(1 if has_t else 0)

    return {"ml_scores": ml_sc, "cfar_scores": cfar_sc, "labels": lbs}


def _roc_curve(scores: list, labels: list):
    s, l = np.array(scores), np.array(labels)
    lo, hi = s.min(), s.max()
    thrs = np.linspace(hi + 1e-6, lo - 1e-6, 300)
    pos, neg = (l == 1).sum(), (l == 0).sum()
    fprs, tprs = [], []
    for t in thrs:
        pred = s >= t
        tprs.append((pred & (l == 1)).sum() / max(pos, 1))
        fprs.append((pred & (l == 0)).sum() / max(neg, 1))
    fpr = np.array(fprs)
    tpr = np.array(tprs)
    o   = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[o], fpr[o]))
    return fpr, tpr, auc


# ── Step 5: Runtime profiling ─────────────────────────────────────────────────

def profile_runtime(params: dict, session, n_reps: int = 50) -> dict:
    """
    Time each operation in both pipelines (ms per sweep, median over n_reps).

    Operations timed:
      cfar_detect    — PPICFARDetector.detect() on one sweep
      cfar_kf_sweep  — full CFAR+KF update (detect + associate) per sweep
      cnn_infer      — ONNX session.run() per call (batch=1, 10-sweep features)
      cnn_kf_sweep   — CNN inference + peak extraction + KF update per sweep
    """
    N_SW    = params["radar"]["n_sweeps"]
    N_AZ    = params["radar"]["n_azimuths"]
    N_RANGE = params["radar"]["n_ranges"]

    rng = np.random.default_rng(7)
    p = {
        "radar": params["radar"],
        "target": dict(snr_db=10.0, range_bin=32.0, azimuth_deg=90.0,
                       radial_velocity=1.0, tangential_velocity=1.0),
    }
    ppi_t, _, _ = generate_ppi_sequence(p, seed=42)

    cfar_det = PPICFARDetector(threshold_factor=2.5)

    # CFAR detect per sweep
    times_cfar = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cfar_det.detect(ppi_t[0])
        times_cfar.append((time.perf_counter() - t0) * 1000)

    # CFAR + KF full pipeline per sweep (amortised — run full sequence, divide by N_SW)
    times_cfar_kf = []
    for _ in range(n_reps):
        kf = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
        t0 = time.perf_counter()
        for sw in range(N_SW):
            kf.update(ppi_t[sw])
        times_cfar_kf.append((time.perf_counter() - t0) * 1000 / N_SW)

    # CNN inference (full 10-sweep features, batch=1)
    times_cnn = []
    feat_full = temporal_features(ppi_t)[np.newaxis].astype(np.float32)
    if session is not None:
        for _ in range(n_reps):
            t0 = time.perf_counter()
            session.run(None, {"ppi_sequence": feat_full})
            times_cnn.append((time.perf_counter() - t0) * 1000)

    # CNN + KF per sweep (incremental features, peak extraction, KF update)
    times_cnn_kf = []
    if session is not None:
        for _ in range(n_reps):
            kf = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            t0 = time.perf_counter()
            for k in range(N_SW):
                feat_k = temporal_features(ppi_t[:k + 1])[np.newaxis].astype(np.float32)
                ml_out = session.run(None, {"ppi_sequence": feat_k})[0][0]
                peaks  = _cnn_peaks(ml_out, 0.30)
                kf.update_from_peaks(peaks, N_AZ, N_RANGE)
            times_cnn_kf.append((time.perf_counter() - t0) * 1000 / N_SW)

    def _stats(ts):
        a = np.array(ts)
        return {"median_ms": round(float(np.median(a)), 3),
                "p95_ms":    round(float(np.percentile(a, 95)), 3),
                "min_ms":    round(float(a.min()), 3)}

    return {
        "cfar_detect_per_sweep":    _stats(times_cfar),
        "cfar_kf_per_sweep":        _stats(times_cfar_kf),
        "cnn_infer_full_10sweep":   _stats(times_cnn) if times_cnn else None,
        "cnn_kf_per_sweep":         _stats(times_cnn_kf) if times_cnn_kf else None,
        "n_reps": n_reps,
        "grid": f"{N_AZ}az × {N_RANGE}r",
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_snr(snr_data: dict, out_path: Path) -> None:
    snr  = snr_data["snr"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(snr, [v * 100 for v in snr_data["cfar_sa"]],
            "o--", color="tab:orange", alpha=0.7, lw=1.5, ms=4,
            label="CFAR standalone (single-look ref)")
    ax.plot(snr, [v * 100 for v in snr_data["cfar_kf"]],
            "o-",  color="tab:orange", lw=2.5, ms=6,
            label="CFAR + KF (classical pipeline)")
    ax.plot(snr, [v * 100 for v in snr_data["cnn_batch"]],
            "s--", color="steelblue", alpha=0.7, lw=1.5, ms=4,
            label="CNN batch (standalone ref)")
    ax.plot(snr, [v * 100 for v in snr_data["cnn_kf"]],
            "s-",  color="steelblue", lw=2.5, ms=6,
            label="CNN + KF (ML pipeline)")
    if snr_data.get("gru_kf"):
        ax.plot(snr, [v * 100 for v in snr_data["gru_kf"]],
                "^-", color="tab:green", lw=2.5, ms=6,
                label="GRU + KF (streaming)")
    ax.axhline(50, ls=":", color="#888", lw=1)
    ax.axhline(90, ls=":", color="#666", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection probability (%)")
    ax.set_title("Pd vs SNR — all four pipelines")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(snr, snr_data["cfar_kf_rmse"], "o-", color="tab:orange",
             lw=2, ms=6, label="CFAR + KF")
    ax2.plot(snr, snr_data["cnn_kf_rmse"],  "s-", color="steelblue",
             lw=2, ms=6, label="CNN + KF")
    if snr_data.get("gru_kf_rmse"):
        ax2.plot(snr, snr_data["gru_kf_rmse"], "^-", color="tab:green",
                 lw=2, ms=6, label="GRU + KF")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Position error (bins)")
    ax2.set_title("Track position RMSE vs SNR")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_sweep(profiles: list, out_path: Path) -> None:
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, prof in zip(axes, profiles):
        sw = prof["sweeps"]
        ax.plot(sw, prof["cfar_sa_cum_pct"], "o--", color="tab:orange",
                alpha=0.7, lw=1.5, ms=4, label="CFAR standalone")
        ax.plot(sw, prof["cfar_kf_cum_pct"], "o-",  color="tab:orange",
                lw=2.5, ms=6, label="CFAR + KF")
        ax.plot(sw, prof["cnn_kf_cum_pct"],  "s-",  color="steelblue",
                lw=2.5, ms=6, label="CNN + KF")
        if prof.get("gru_kf_cum_pct"):
            ax.plot(sw, prof["gru_kf_cum_pct"], "^-", color="tab:green",
                    lw=2.5, ms=6, label="GRU + KF")
        ax.axhline(50, ls=":", color="#888", lw=1)
        ax.axhline(90, ls=":", color="#666", lw=1)
        lat_c = prof.get("cfar_kf_latency")
        lat_m = prof.get("cnn_kf_latency")
        lat_g = prof.get("gru_kf_latency")
        caption = f"CFAR+KF lat: {lat_c:.1f}sw" if lat_c else "CFAR+KF: —"
        caption += f"  |  CNN+KF lat: {lat_m:.1f}sw" if lat_m else "  |  CNN+KF: —"
        caption += f"  |  GRU+KF lat: {lat_g:.1f}sw" if lat_g else ""
        ax.set_title(f"SNR = {prof['snr_db']:+.0f} dB\n{caption}", fontsize=10)
        ax.set_xlabel("Sweep number")
        ax.set_ylabel("Cumulative Pd (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(sw)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Evidence accumulation — cumulative Pd vs sweep count", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_roc(roc: dict, out_path: Path) -> None:
    ml_fpr, ml_tpr, ml_auc     = _roc_curve(roc["ml_scores"],   roc["labels"])
    cf_fpr, cf_tpr, cf_auc     = _roc_curve(roc["cfar_scores"], roc["labels"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC 0.500)")
    ax.plot(ml_fpr, ml_tpr, color="steelblue", lw=2.5,
            label=f"ML CNN   AUC {ml_auc:.3f}")
    ax.plot(cf_fpr, cf_tpr, color="tab:orange", lw=2.5, ls="--",
            label=f"CA-CFAR  AUC {cf_auc:.3f}")
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Detection Rate")
    ax.set_title(f"ROC — ML vs CA-CFAR  (SNR = {roc.get('snr_db', '?'):+.0f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CSV helper ────────────────────────────────────────────────────────────────

def _save_csv(snr_data: dict, out_path: Path) -> None:
    import csv
    has_gru = bool(snr_data.get("gru_kf"))
    gru_kf     = snr_data.get("gru_kf",     [0.0] * len(snr_data["snr"]))
    gru_rmse   = snr_data.get("gru_kf_rmse",[0.0] * len(snr_data["snr"]))
    rows = list(zip(
        snr_data["snr"],
        snr_data["cfar_sa"],
        snr_data["cfar_kf"],
        snr_data["cnn_batch"],
        snr_data["cnn_kf"],
        gru_kf,
        snr_data["cfar_kf_rmse"],
        snr_data["cnn_kf_rmse"],
        gru_rmse,
    ))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "cfar_sa_pd", "cfar_kf_pd",
                    "cnn_batch_pd", "cnn_kf_pd", "gru_kf_pd",
                    "cfar_kf_rmse_bins", "cnn_kf_rmse_bins", "gru_kf_rmse_bins"])
        for r in rows:
            w.writerow([f"{v:.4f}" for v in r])
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials",   type=int,   default=30,
                    help="Trials per SNR point (default 30)")
    ap.add_argument("--n-scenes",   type=int,   default=200,
                    help="Clutter-only scenes for Pfa calibration (default 200)")
    ap.add_argument("--roc-trials", type=int,   default=100,
                    help="Total trials for ROC curves (default 100, half target/half clutter)")
    ap.add_argument("--roc-snr",    type=float, default=10.0,
                    help="SNR for ROC evaluation (default 10)")
    ap.add_argument("--sweep-snr",  type=float, nargs="+", default=[0.0, 10.0, 20.0],
                    help="SNR values for per-sweep profile (default: 0 10 20)")
    ap.add_argument("--runtime-reps", type=int, default=50,
                    help="Repetitions for runtime profiling (default 50)")
    ap.add_argument("--out-dir",    type=str,   default="artifacts",
                    help="Output directory (default: artifacts)")
    return ap.parse_args()


def main():
    args = _parse_args()
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading params and ONNX model...")
    params      = _load_params()
    session     = _load_session()
    gru_session = _load_gru_session()

    results = {}

    # ── 1. Calibration ────────────────────────────────────────────────────────
    print("\n[1/5] Operating-point calibration")
    cal = calibrate_pfa(params, session, gru_session=gru_session, n_scenes=args.n_scenes)
    results["calibration"] = cal
    cnn_thr = cal["cnn_thr"]
    gru_thr = cal["gru_thr"]
    print(f"  CFAR Pfa = {cal['cfar_pfa']*100:.3f}%  |  CNN thr = {cnn_thr:.3f}  |  GRU thr = {gru_thr:.3f}")
    print(f"  False tracks/scene — CFAR+KF: {cal['cfar_kf_ft_rate']:.2f}  "
          f"CNN+KF: {cal['cnn_kf_ft_rate']:.2f}  "
          f"GRU+KF: {cal['gru_kf_ft_rate']:.2f}")

    # ── 2. SNR sweep ──────────────────────────────────────────────────────────
    print("\n[2/5] SNR sweep")
    snr_data = snr_sweep(params, session, gru_session=gru_session,
                         cnn_thr=cnn_thr, gru_thr=gru_thr, n_trials=args.n_trials)
    results["snr_sweep"] = snr_data
    _save_csv(snr_data, out / "comparison_snr.csv")
    _plot_snr(snr_data, out / "comparison_snr.png")

    # Print summary table
    snr_arr = np.array(snr_data["snr"])
    for target_snr in [0.0, 10.0, 20.0]:
        idx = int(np.argmin(np.abs(snr_arr - target_snr)))
        gru_pd = snr_data['gru_kf'][idx]*100 if snr_data.get("gru_kf") else float("nan")
        print(f"  SNR={target_snr:+.0f} dB: "
              f"CFAR+KF Pd={snr_data['cfar_kf'][idx]*100:.0f}%  "
              f"CNN+KF Pd={snr_data['cnn_kf'][idx]*100:.0f}%  "
              f"GRU+KF Pd={gru_pd:.0f}%  "
              f"CFAR rmse={snr_data['cfar_kf_rmse'][idx]:.1f}bin  "
              f"CNN rmse={snr_data['cnn_kf_rmse'][idx]:.1f}bin")

    # ── 3. Per-sweep profiles ─────────────────────────────────────────────────
    print("\n[3/5] Per-sweep cumulative Pd profiles")
    profiles = []
    for snr_val in args.sweep_snr:
        print(f"  Profile at SNR = {snr_val:+.0f} dB...")
        prof = sweep_profile(params, session, gru_session=gru_session,
                             snr_db=snr_val, cnn_thr=cnn_thr, gru_thr=gru_thr,
                             n_trials=args.n_trials)
        profiles.append(prof)
        lat_c = prof.get("cfar_kf_latency")
        lat_m = prof.get("cnn_kf_latency")
        lat_g = prof.get("gru_kf_latency")
        print(f"    CFAR+KF latency: {lat_c:.1f} sweeps" if lat_c else "    CFAR+KF: never confirmed")
        print(f"    CNN+KF  latency: {lat_m:.1f} sweeps" if lat_m else "    CNN+KF:  never confirmed")
        print(f"    GRU+KF  latency: {lat_g:.1f} sweeps" if lat_g else "    GRU+KF:  never confirmed")
    results["sweep_profiles"] = profiles
    _plot_sweep(profiles, out / "comparison_sweep.png")

    # ── 4. ROC curves ─────────────────────────────────────────────────────────
    print(f"\n[4/5] ROC curves at SNR = {args.roc_snr:+.0f} dB")
    roc = roc_data(params, session, snr_db=args.roc_snr, n=args.roc_trials)
    roc["snr_db"] = args.roc_snr
    _, _, ml_auc   = _roc_curve(roc["ml_scores"],   roc["labels"])
    _, _, cf_auc   = _roc_curve(roc["cfar_scores"], roc["labels"])
    roc["ml_auc"]   = ml_auc
    roc["cfar_auc"] = cf_auc
    results["roc"]  = roc
    _plot_roc(roc, out / "comparison_roc.png")
    print(f"  ML CNN AUC:  {ml_auc:.3f}")
    print(f"  CA-CFAR AUC: {cf_auc:.3f}")

    # ── 5. Runtime profiling ──────────────────────────────────────────────────
    print("\n[5/5] Runtime profiling")
    rt = profile_runtime(params, session, n_reps=args.runtime_reps)
    results["runtime"] = rt
    print(f"  CFAR detect/sweep:   {rt['cfar_detect_per_sweep']['median_ms']:.2f} ms (median)")
    print(f"  CFAR+KF/sweep:       {rt['cfar_kf_per_sweep']['median_ms']:.2f} ms (median)")
    if rt["cnn_infer_full_10sweep"]:
        print(f"  CNN infer (10-sw):   {rt['cnn_infer_full_10sweep']['median_ms']:.2f} ms (median)")
    if rt["cnn_kf_per_sweep"]:
        print(f"  CNN+KF/sweep:        {rt['cnn_kf_per_sweep']['median_ms']:.2f} ms (median)")
    with open(out / "comparison_runtime.json", "w") as f:
        json.dump(rt, f, indent=2)
    print(f"  Saved: {out / 'comparison_runtime.json'}")

    # ── Save full results ──────────────────────────────────────────────────────
    results_path = out / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
