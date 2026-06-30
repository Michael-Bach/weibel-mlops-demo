"""
radar/detection.py — gen, cfar_sweeps, kf_result, build_kf_history_*, calibrate_pfa,
                     snr_sweep, sweep_profile, roc_data, roc, auc, cfar_path_score,
                     het_clutter_demo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st
from scipy.ndimage import maximum_filter

from radar.sessions import (
    N_SW, N_AZ, N_RANGE,
    load_params, load_cnn_session, load_gru_session,
)
from radar.inference import _ml_map, _gru_step_raw, _cnn_peaks, _gru_peaks, _tf_map
from radar.sessions import load_transformer_session
from src.data.ppi_generator import (
    generate_clutter_only,
    generate_heterogeneous_ppi,
    generate_ppi_sequence,
    temporal_features,
)
from src.baseline.ppi_cfar_kf import PPICFARDetector, PPIKalmanTracker


@st.cache_data
def _gen(snr, rb, az, vr, vt):
    params = load_params()
    p = {
        "radar": params["radar"],
        "target": dict(snr_db=snr, range_bin=rb, azimuth_deg=az,
                       radial_velocity=vr, tangential_velocity=vt),
    }
    return generate_ppi_sequence(p, seed=42)


@st.cache_data(show_spinner="Running CA-CFAR...")
def _cfar_sweeps(snr, rb, az, vr, vt):
    """Per-sweep CFAR boolean maps: bool (n_sweeps, n_az, n_range)."""
    ppi, _, _ = _gen(snr, rb, az, vr, vt)
    return PPICFARDetector().detect_sequence(ppi)


@st.cache_data
def _kf_result(snr, rb, az, vr, vt):
    ppi, _, _ = _gen(snr, rb, az, vr, vt)
    kf = PPIKalmanTracker()
    prob = kf.process_sequence(ppi)
    return prob, kf.track_positions()   # track_positions: list of (az_bin, r_bin)


@st.cache_data
def _build_kf_history_cfar(snr, rb, az, vr, vt):
    """Per-sweep CFAR+KF confirmed track positions. Returns list of N_SW track lists."""
    ppi, _, _ = _gen(snr, rb, az, vr, vt)
    kf = PPIKalmanTracker()
    history = []
    for k in range(N_SW):
        kf.update(ppi[k])
        history.append(kf.track_positions())
    return history


@st.cache_data
def _build_kf_history_cnn(snr, rb, az, vr, vt, threshold: float = 0.30):
    """Per-sweep CNN+KF confirmed track positions. Returns list of N_SW track lists."""
    session = load_cnn_session()
    if session is None:
        return [[] for _ in range(N_SW)]
    ppi, _, _ = _gen(snr, rb, az, vr, vt)
    kf = PPIKalmanTracker()
    history = []
    for k in range(N_SW):
        ml_out = _ml_map(ppi, k + 1)
        peaks = _cnn_peaks(ml_out, threshold)
        kf.update_from_peaks(peaks, N_AZ, N_RANGE)
        history.append(kf.track_positions())
    return history


@st.cache_data(show_spinner="Calibrating matched false-alarm threshold…")
def _calibrate_pfa(n_scenes: int = 200, _v: int = 3):
    """
    Empirically measure CFAR Pfa at threshold_factor=2.5 on clutter-only scenes,
    then find the CNN and GRU thresholds that match it (same cells-above-threshold rate).
    Also counts confirmed false tracks per clutter-only scene for all three pipelines.

    Returns
    -------
    cfar_pfa          : float  — CFAR Pfa per cell per sweep
    cnn_thr           : float  — CNN threshold giving same cell-level Pfa
    cfar_kf_ft_rate   : float  — mean confirmed false tracks per scene (CFAR+KF)
    cnn_kf_ft_rate    : float  — mean confirmed false tracks per scene (CNN+KF)
    gru_thr           : float  — GRU threshold giving same cell-level Pfa
    gru_kf_ft_rate    : float  — mean confirmed false tracks per scene (GRU+KF)
    """
    params = load_params()
    session = load_cnn_session()
    gru_session = load_gru_session()

    rng  = np.random.default_rng(13)
    cfar = PPICFARDetector(threshold_factor=2.5)

    cfar_fa_cells      = 0
    cfar_kf_false_trk  = 0
    cnn_vals: list     = []
    gru_vals: list     = []

    for _ in range(n_scenes):
        seed  = int(rng.integers(1, 1_000_000))
        ppi_c = generate_clutter_only(params, seed=seed)

        # CFAR false alarm count
        det_seq = cfar.detect_sequence(ppi_c)
        cfar_fa_cells += int(det_seq.sum())

        # CFAR+KF false track count
        kf_c = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
        kf_c.process_sequence(ppi_c)
        cfar_kf_false_trk += len(kf_c.track_positions())

        # Collect CNN output values across all k=1..N_SW incremental calls so the
        # threshold reflects the actual per-cell-per-sweep distribution the CNN+KF
        # pipeline sees, matching the CFAR Pfa denominator (n_scenes × N_SW × cells).
        if session is not None:
            for k in range(N_SW):
                feat_k = temporal_features(ppi_c[:k + 1])[np.newaxis].astype(np.float32)
                out_k  = session.run(None, {"ppi_sequence": feat_k})[0][0]
                cnn_vals.extend(out_k.ravel().tolist())

        # Collect GRU output values streaming per-sweep (same denominator)
        if gru_session is not None:
            h_g = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
            nf  = np.percentile(ppi_c[0], 10, axis=0).clip(1e-3)
            for k in range(N_SW):
                pm, h_g, nf = _gru_step_raw(ppi_c[k], h_g, nf)
                gru_vals.extend(pm.ravel().tolist())

    cfar_pfa = cfar_fa_cells / (n_scenes * N_SW * N_AZ * N_RANGE)

    if cnn_vals and session is not None:
        arr     = np.array(cnn_vals, dtype=np.float32)
        cnn_thr = float(np.clip(np.percentile(arr, 100.0 * (1.0 - cfar_pfa)),
                                0.01, 0.99))
    else:
        cnn_thr = 0.30

    if gru_vals and gru_session is not None:
        arr_g   = np.array(gru_vals, dtype=np.float32)
        gru_thr = float(np.clip(np.percentile(arr_g, 100.0 * (1.0 - cfar_pfa)),
                                0.01, 0.99))
    else:
        gru_thr = 0.20

    # CNN+KF and GRU+KF false track rates at matched thresholds (second pass, same seeds)
    rng2 = np.random.default_rng(13)
    cnn_kf_false_trk = 0
    gru_kf_false_trk = 0
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
            h_g = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
            nf  = np.percentile(ppi_c[0], 10, axis=0).clip(1e-3)
            for k in range(N_SW):
                pm, h_g, nf = _gru_step_raw(ppi_c[k], h_g, nf)
                peaks = _gru_peaks(pm, gru_thr)
                kf_gru.update_from_peaks(peaks, N_AZ, N_RANGE)
            gru_kf_false_trk += len(kf_gru.track_positions())

    return (cfar_pfa,
            cnn_thr,
            cfar_kf_false_trk / n_scenes,
            cnn_kf_false_trk  / n_scenes,
            gru_thr,
            gru_kf_false_trk  / n_scenes)


@st.cache_data(show_spinner="Running SNR sweep (30 trials × 16 points)…")
def _snr_sweep(cnn_thr: float = 0.30, gru_thr: float = 0.20,
               n_trials: int = 30, _v: int = 11):
    """
    Compare five pipelines across 16 SNR points (30 trials each):

      1. CFAR standalone   — single-look Pd per (trial, sweep)
      2. CFAR + KF         — confirmed track within 6 bins of target final position
      3. CNN batch         — CNN on all 10 sweeps, threshold at matched Pfa
      4. CNN  + KF         — CNN peaks (incremental, sweeps 1..k) feeding same KF
      5. GRU  + KF         — ConvGRU streaming per-sweep, peaks feeding same KF

    Also records position error (bins) for detected trials of pipelines 2, 4, and 5.
    """
    params = load_params()
    session = load_cnn_session()
    gru_session = load_gru_session()

    rng      = np.random.default_rng(0)
    snr_vals = np.linspace(-20, 40, 16)
    cfar     = PPICFARDetector(threshold_factor=2.5)

    out_cfar_sa, out_cfar_kf, out_cnn_b, out_cnn_kf, out_gru_kf = [], [], [], [], []
    out_cfar_kf_rmse, out_cnn_kf_rmse, out_gru_kf_rmse = [], [], []

    for snr in snr_vals:
        cfar_sa_hits = 0          # (trial × sweep) pairs
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

            # ── 1. CFAR standalone (single-look Pd) ───────────────────────────
            cfar_seq = cfar.detect_sequence(ppi_t)
            for t, (az_b, r_b) in enumerate(zip(path_az, path_r)):
                if cfar_seq[t, az_b, r_b]:
                    cfar_sa_hits += 1

            # ── 2. CFAR + KF (incremental — check per-sweep target position) ──
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
                        err  = np.sqrt(dr ** 2 + daz ** 2)
                        if err < 6:
                            cfar_kf_h   += 1
                            cfar_kf_err += err
                            cfar_kf_det += 1
                            cfar_kf_found = True
                            break

            if session is not None:
                feat   = temporal_features(ppi_t)[np.newaxis].astype(np.float32)
                ml_out = session.run(None, {"ppi_sequence": feat})[0][0]

                # ── 3. CNN batch (all 10 sweeps, threshold at matched Pfa) ────
                for az_b, r_b in zip(path_az, path_r):
                    win = ml_out[max(0, az_b - 1):az_b + 2,
                                 max(0, r_b  - 1):r_b  + 2]
                    if float(win.max()) > cnn_thr:
                        cnn_b_h += 1
                        break

                # ── 4. CNN + KF (incremental — check per-sweep target position) ─
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
                            err  = np.sqrt(dr ** 2 + daz ** 2)
                            if err < 6:
                                cnn_kf_h   += 1
                                cnn_kf_err += err
                                cnn_kf_det += 1
                                cnn_kf_found = True
                                break

            # ── 5. GRU + KF (streaming per-sweep) ────────────────────────────
            if gru_session is not None:
                h_g  = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
                nf   = np.percentile(ppi_t[0], 10, axis=0).clip(1e-3)
                kf_g = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
                gru_kf_found = False
                for k in range(N_SW):
                    pm, h_g, nf = _gru_step_raw(ppi_t[k], h_g, nf)
                    peaks = _gru_peaks(pm, gru_thr)
                    kf_g.update_from_peaks(peaks, N_AZ, N_RANGE)
                    if gru_kf_found:
                        continue
                    az_b_k, r_b_k = path_az[k], path_r[k]
                    for tr in kf_g._tracks:
                        if tr.hits >= kf_g.min_hits:
                            az_w = float(tr.x[2]) % N_AZ
                            daz  = min(abs(az_w - az_b_k), N_AZ - abs(az_w - az_b_k))
                            dr   = abs(float(tr.x[0]) - r_b_k)
                            err  = np.sqrt(dr ** 2 + daz ** 2)
                            if err < 6:
                                gru_kf_h   += 1
                                gru_kf_err += err
                                gru_kf_det += 1
                                gru_kf_found = True
                                break

        out_cfar_sa.append(cfar_sa_hits / (n_trials * N_SW))
        out_cfar_kf.append(cfar_kf_h / n_trials)
        out_cnn_b.append(cnn_b_h   / n_trials)
        out_cnn_kf.append(cnn_kf_h / n_trials)
        out_gru_kf.append(gru_kf_h / n_trials)
        out_cfar_kf_rmse.append(cfar_kf_err / max(cfar_kf_det, 1))
        out_cnn_kf_rmse.append(cnn_kf_err  / max(cnn_kf_det,  1))
        out_gru_kf_rmse.append(gru_kf_err  / max(gru_kf_det,  1))

    return (snr_vals,
            np.array(out_cfar_sa),
            np.array(out_cfar_kf),
            np.array(out_cnn_b),
            np.array(out_cnn_kf),
            np.array(out_cfar_kf_rmse),
            np.array(out_cnn_kf_rmse),
            np.array(out_gru_kf),
            np.array(out_gru_kf_rmse))


@st.cache_data(show_spinner="Computing per-sweep profiles (30 trials)…")
def _sweep_profile(snr_db_val: float, cnn_thr: float = 0.30, gru_thr: float = 0.20,
                   n_trials: int = 30, _v: int = 5):
    """
    For each sweep k (1..N_SW) compute cumulative Pd and mean track latency
    for four pipelines at a matched operating point:

      • CFAR standalone  — cumulative Pd (at least one hit at target cell by sweep k)
      • CFAR + KF        — cumulative Pd (confirmed track within 6 bins by sweep k)
      • CNN  + KF        — same criterion, with incremental CNN feeding KF
      • GRU  + KF        — same criterion, with streaming GRU feeding KF

    Track initiation latency = mean sweep index of first confirmed track
    (averaged over trials that eventually confirm, reported for KF pipelines).
    """
    params = load_params()
    session = load_cnn_session()
    gru_session = load_gru_session()

    rng  = np.random.default_rng(99)
    cfar = PPICFARDetector(threshold_factor=2.5)

    cfar_sa_cum  = np.zeros(N_SW)
    cfar_kf_cum  = np.zeros(N_SW)
    cnn_kf_cum   = np.zeros(N_SW)
    gru_kf_cum   = np.zeros(N_SW)
    cfar_kf_lat  = []
    cnn_kf_lat   = []
    gru_kf_lat   = []

    for _ in range(n_trials):
        seed = int(rng.integers(1, 1_000_000))
        p = {
            "radar": params["radar"],
            "target": dict(
                snr_db=float(snr_db_val),
                range_bin=float(rng.integers(10, N_RANGE - 10)),
                azimuth_deg=float(rng.uniform(0, 360)),
                radial_velocity=float(rng.uniform(-3, 3)),
                tangential_velocity=float(rng.uniform(-4, 4)),
            ),
        }
        ppi_t, _, _ = generate_ppi_sequence(p, seed=seed)
        tgt = p["target"]
        r0, az0 = float(tgt["range_bin"]), float(tgt["azimuth_deg"])
        vr_t, vaz_t = float(tgt["radial_velocity"]), float(tgt["tangential_velocity"])

        path_az, path_r = [], []
        for sw in range(N_SW):
            r_t = float(np.clip(r0 + sw * vr_t, 0, N_RANGE - 1))
            az_t = (az0 + sw * vaz_t) % 360.0
            path_r.append(int(np.round(r_t)))
            path_az.append(int(np.round(az_t * N_AZ / 360.0)) % N_AZ)

        # ── CFAR standalone ───────────────────────────────────────────────────
        cfar_seq = cfar.detect_sequence(ppi_t)
        cfar_hit_ever = False
        for k in range(N_SW):
            if cfar_seq[k, path_az[k], path_r[k]]:
                cfar_hit_ever = True
            if cfar_hit_ever:
                cfar_sa_cum[k] += 1

        # ── CFAR + KF ─────────────────────────────────────────────────────────
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
                    if np.sqrt(dr ** 2 + daz ** 2) < 6:
                        cfar_kf_ever = True
                        cfar_kf_lat_trial = k + 1
                        cfar_kf_cum[k] += 1
                        break
        if cfar_kf_lat_trial is not None:
            cfar_kf_lat.append(cfar_kf_lat_trial)

        # ── CNN + KF (incremental) ────────────────────────────────────────────
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
                        if np.sqrt(dr ** 2 + daz ** 2) < 6:
                            cnn_kf_ever = True
                            cnn_kf_lat_trial = k + 1
                            cnn_kf_cum[k] += 1
                            break
            if cnn_kf_lat_trial is not None:
                cnn_kf_lat.append(cnn_kf_lat_trial)

        # ── GRU + KF (streaming per-sweep) ───────────────────────────────────
        if gru_session is not None:
            h_g  = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
            nf   = np.percentile(ppi_t[0], 10, axis=0).clip(1e-3)
            kf_g = PPIKalmanTracker(cfar=PPICFARDetector(threshold_factor=2.5))
            gru_kf_ever = False
            gru_kf_lat_trial = None
            for k in range(N_SW):
                pm, h_g, nf = _gru_step_raw(ppi_t[k], h_g, nf)
                peaks = _gru_peaks(pm, gru_thr)
                kf_g.update_from_peaks(peaks, N_AZ, N_RANGE)
                if gru_kf_ever:
                    gru_kf_cum[k] += 1
                    continue
                az_b_k, r_b_k = path_az[k], path_r[k]
                for tr in kf_g._tracks:
                    if tr.hits >= kf_g.min_hits:
                        dr  = abs(float(tr.x[0]) - r_b_k)
                        daz = min(abs(float(tr.x[2]) % N_AZ - az_b_k),
                                  N_AZ - abs(float(tr.x[2]) % N_AZ - az_b_k))
                        if np.sqrt(dr ** 2 + daz ** 2) < 6:
                            gru_kf_ever = True
                            gru_kf_lat_trial = k + 1
                            gru_kf_cum[k] += 1
                            break
            if gru_kf_lat_trial is not None:
                gru_kf_lat.append(gru_kf_lat_trial)

    cfar_kf_latency = float(np.mean(cfar_kf_lat)) if cfar_kf_lat else float("nan")
    cnn_kf_latency  = float(np.mean(cnn_kf_lat))  if cnn_kf_lat  else float("nan")
    gru_kf_latency  = float(np.mean(gru_kf_lat))  if gru_kf_lat  else float("nan")

    return (cfar_sa_cum  / n_trials * 100,
            cfar_kf_cum  / n_trials * 100,
            cnn_kf_cum   / n_trials * 100,
            gru_kf_cum   / n_trials * 100,
            cfar_kf_latency,
            cnn_kf_latency,
            gru_kf_latency)


def _cfar_path_score(detect_seq: np.ndarray, t_params: dict) -> float:
    """
    Per-cell CFAR score along the target's actual path across sweeps.
    Returns the fraction of sweeps where CFAR fires within a ±1-bin
    neighbourhood of the target's instantaneous position.
    This is the canonical single-cell Pd that CFAR ROC curves measure.
    """
    n_sw, n_az, n_range = detect_seq.shape
    tgt = t_params["target"]
    r0, az0 = float(tgt["range_bin"]), float(tgt["azimuth_deg"])
    vr, vaz  = float(tgt["radial_velocity"]), float(tgt["tangential_velocity"])
    hits = 0
    for sw in range(n_sw):
        r_t   = r0 + sw * vr
        az_t  = (az0 + sw * vaz) % 360.0
        r_bin = int(np.round(r_t).clip(0, n_range - 1))
        az_bin = int(np.round(az_t * n_az / 360.0)) % n_az
        for daz in (-1, 0, 1):
            for dr in (-1, 0, 1):
                az_c = (az_bin + daz) % n_az
                r_c  = int(np.clip(r_bin + dr, 0, n_range - 1))
                if detect_seq[sw, az_c, r_c]:
                    hits += 1
                    break
            else:
                continue
            break
    return hits / n_sw


def _lrt_score(ppi: np.ndarray) -> np.ndarray:
    """
    Non-coherent square-law integrator — NP-optimal at low SNR for Rayleigh clutter.
    Normalises by the per-range-bin noise floor (axis=(0,1): across all sweeps and
    azimuths) so the score is range-invariant.  Using axis=(1,2) gives a single
    per-sweep scalar dominated by bright near-range clutter, which grossly inflates
    scores and breaks the threshold calibration.
    Returns a (n_az, n_range) score map.
    """
    noise_floor = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-6)
    normed = ppi / noise_floor
    return (normed ** 2).sum(axis=0)


def _dp_tbd_score(ppi: np.ndarray, max_vr: int = 3, max_vaz: int = 2) -> np.ndarray:
    """
    Dynamic-programming Track-Before-Detect (Barniv / Viterbi-style).
    Uses the same per-range-bin noise floor as _lrt_score for consistency.
    Returns a (n_az, n_range) score map where high values mark trajectory endpoints.
    """
    noise_floor = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-6)
    normed = (ppi / noise_floor).astype(np.float32)
    S = normed[0].copy()
    size = (2 * max_vaz + 1, 2 * max_vr + 1)
    for k in range(1, len(normed)):
        best_prev = maximum_filter(S, size=size, mode=('wrap', 'nearest'))
        S = normed[k] + best_prev
    return S


@st.cache_data(show_spinner="Computing ROC curves (100 trials)…")
def _roc_data(snr_db_val: float, n: int = 100, _v: int = 12):
    params = load_params()
    session = load_cnn_session()
    gru_session = load_gru_session()

    rng = np.random.default_rng(77)
    ml_sc, cfar_sc, gru_sc, lrt_sc, tbd_sc, tf_sc, lbs = [], [], [], [], [], [], []
    cfar_d = PPICFARDetector()
    tf_session = load_transformer_session()

    for i in range(n):
        has_t = i % 2 == 0
        seed  = int(rng.integers(1, 1_000_000))

        if has_t:
            p = {
                "radar": params["radar"],
                "target": dict(
                    snr_db=snr_db_val,
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

        # All three detectors use the same oracle-path scoring so the ROC is
        # a fair cell-level Pd vs Pfa comparison (Neyman-Pearson framing):
        #   target trial  -> score at each sweep's known target position
        #   clutter trial -> score at a fixed random reference cell
        # Using ml_out.max() (global map max) for clutter trials inflated the
        # clutter score over ~16k cells and suppressed ML AUC artificially.
        detect_seq = cfar_d.detect_sequence(ppi_t)

        if has_t:
            tgt   = p["target"]
            r0_t  = float(tgt["range_bin"])
            az0_t = float(tgt["azimuth_deg"])
            vr_t  = float(tgt["radial_velocity"])
            vaz_t = float(tgt["tangential_velocity"])

            cfar_sc.append(_cfar_path_score(detect_seq, p))

            # ML: max output in +-1-bin window around each sweep's target cell
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

            # CFAR: fraction of sweeps firing in +-1-bin around reference cell
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

            # ML: output in +-1-bin window around the same reference cell
            win = ml_out[max(0, az_rand - 1):az_rand + 2,
                         max(0, r_rand  - 1):r_rand  + 2]
            ml_sc.append(float(win.max()) if win.size else 0.0)

        # GRU: accumulate hidden state over all sweeps, score at oracle position
        if gru_session is not None:
            h = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
            for sw in range(N_SW):
                sweep_norm = ppi_t[sw][np.newaxis, np.newaxis].astype(np.float32)
                gru_out = gru_session.run(
                    None, {"sweep_norm": sweep_norm, "h_in": h}
                )
                h = gru_out[1]
            gru_map = h[0, 0]
        else:
            gru_map = np.zeros((N_AZ, N_RANGE), dtype=np.float32)

        if has_t:
            gru_path_max = 0.0
            for sw in range(N_SW):
                r_t  = r0_t + sw * vr_t
                az_t = (az0_t + sw * vaz_t) % 360.0
                r_b  = int(np.round(r_t).clip(0, N_RANGE - 1))
                az_b = int(np.round(az_t * N_AZ / 360.0)) % N_AZ
                win  = gru_map[max(0, az_b - 1):az_b + 2,
                                max(0, r_b  - 1):r_b  + 2]
                gru_path_max = max(gru_path_max,
                                   float(win.max()) if win.size else 0.0)
            gru_sc.append(gru_path_max)
        else:
            win = gru_map[max(0, az_rand - 1):az_rand + 2,
                          max(0, r_rand  - 1):r_rand  + 2]
            gru_sc.append(float(win.max()) if win.size else 0.0)

        # ── LRT and DP-TBD ────────────────────────────────────────────────────
        tbd_map = _dp_tbd_score(ppi_t)

        if has_t:
            # LRT: path-integrated score (sum per-sweep window-max^2 along oracle path)
            lrt_nf     = np.percentile(ppi_t, 10, axis=(0, 1), keepdims=True).clip(1e-6)
            lrt_normed = ppi_t / lrt_nf
            lrt_score  = 0.0
            az_b = 0
            r_b  = 0
            for sw in range(N_SW):
                r_t  = r0_t + sw * vr_t
                az_t = (az0_t + sw * vaz_t) % 360.0
                r_b  = int(np.round(r_t).clip(0, N_RANGE - 1))
                az_b = int(np.round(az_t * N_AZ / 360.0)) % N_AZ
                win  = lrt_normed[sw, max(0, az_b - 1):az_b + 2,
                                      max(0, r_b  - 1):r_b  + 2]
                lrt_score += float((win ** 2).max()) if win.size else 0.0
            lrt_sc.append(lrt_score)
            # TBD: score at final oracle position in DP map
            win_t = tbd_map[max(0, az_b - 1):az_b + 2,
                            max(0, r_b  - 1):r_b  + 2]
            tbd_sc.append(float(win_t.max()) if win_t.size else 0.0)
        else:
            # LRT: path-integral at the same random position for all sweeps
            lrt_nf     = np.percentile(ppi_t, 10, axis=(0, 1), keepdims=True).clip(1e-6)
            lrt_normed = ppi_t / lrt_nf
            lrt_score  = 0.0
            for sw in range(N_SW):
                win = lrt_normed[sw, max(0, az_rand - 1):az_rand + 2,
                                     max(0, r_rand  - 1):r_rand  + 2]
                lrt_score += float((win ** 2).max()) if win.size else 0.0
            lrt_sc.append(lrt_score)
            win_t = tbd_map[max(0, az_rand - 1):az_rand + 2,
                            max(0, r_rand  - 1):r_rand  + 2]
            tbd_sc.append(float(win_t.max()) if win_t.size else 0.0)

        # ── Transformer ───────────────────────────────────────────────────────
        if tf_session is not None:
            tf_out = _tf_map(ppi_t, N_SW)
            if has_t:
                tf_path_max = 0.0
                for sw in range(N_SW):
                    r_t  = r0_t + sw * vr_t
                    az_t = (az0_t + sw * vaz_t) % 360.0
                    r_b  = int(np.round(r_t).clip(0, N_RANGE - 1))
                    az_b = int(np.round(az_t * N_AZ / 360.0)) % N_AZ
                    win  = tf_out[max(0, az_b - 1):az_b + 2,
                                  max(0, r_b  - 1):r_b  + 2]
                    tf_path_max = max(tf_path_max, float(win.max()) if win.size else 0.0)
                tf_sc.append(tf_path_max)
            else:
                win = tf_out[max(0, az_rand - 1):az_rand + 2,
                             max(0, r_rand  - 1):r_rand  + 2]
                tf_sc.append(float(win.max()) if win.size else 0.0)
        else:
            tf_sc.append(0.0)

        lbs.append(1 if has_t else 0)

    return (np.array(ml_sc), np.array(cfar_sc),
            np.array(gru_sc), np.array(lrt_sc),
            np.array(tbd_sc), np.array(tf_sc),
            np.array(lbs))


def _roc(scores: np.ndarray, labels: np.ndarray):
    lo, hi = scores.min(), scores.max()
    thrs = np.linspace(hi + 1e-6, lo - 1e-6, 300)
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    fprs, tprs = [], []
    for t in thrs:
        pred = scores >= t
        tprs.append((pred & (labels == 1)).sum() / max(pos, 1))
        fprs.append((pred & (labels == 0)).sum() / max(neg, 1))
    return np.array(fprs, dtype=float), np.array(tprs, dtype=float)


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    o = np.argsort(fpr)
    return float(np.trapezoid(tpr[o], fpr[o]))


@st.cache_data(show_spinner="Generating heterogeneous clutter scene…")
def het_clutter_demo(gru_thr_val: float, cfar_thr_factor: float = 2.5):
    params = load_params()
    gru_session = load_gru_session()

    # Target stays at az_bin≈53 (outside the patch at 60-100), approaching
    ppi_h, positions_h, patch_mask = generate_heterogeneous_ppi(
        params, seed=7, patch_gain=5.0,
        target_azimuth_deg=106.0, target_vaz=0.0,
        target_range_bin=35.0, target_vr=-0.8,
        target_snr_db=10.0,
    )

    # CFAR detections — accumulate Boolean hits across all sweeps
    cfar_h = PPICFARDetector(threshold_factor=cfar_thr_factor)
    cfar_hits = np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    for sw in range(N_SW):
        det = cfar_h.detect(ppi_h[sw])
        cfar_hits += det.astype(np.float32)

    # GRU probability map — final hidden state after streaming all sweeps
    if gru_session is not None:
        h_g = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
        nf  = np.percentile(ppi_h[0], 10, axis=0).clip(1e-3)
        for sw in range(N_SW):
            pm, h_g, nf = _gru_step_raw(ppi_h[sw], h_g, nf)
        gru_prob = pm
    else:
        gru_prob = np.zeros((N_AZ, N_RANGE), dtype=np.float32)

    # Mean amplitude across sweeps (for background)
    mean_amp = ppi_h.mean(axis=0)
    return cfar_hits, gru_prob, mean_amp, patch_mask, positions_h
