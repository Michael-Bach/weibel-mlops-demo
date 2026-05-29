"""
radar/inference.py — ML inference: ml_map, gru_step, gru_step_raw, gru_seq, cnn_peaks, gru_peaks.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

from radar.sessions import N_SW, N_AZ, N_RANGE, load_cnn_session, load_gru_session
from src.data.ppi_generator import temporal_features


def _ml_map(ppi_seq: np.ndarray, n_sw: int) -> np.ndarray:
    session = load_cnn_session()
    if session is None:
        return np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    # Always feed a full N_SW window so inference stays in-distribution.
    # Unseen sweeps are zero-padded at the front; the target sweeps occupy
    # the trailing slots, matching the temporal ordering seen during training.
    padded = np.zeros_like(ppi_seq)          # (N_SW, N_AZ, N_RANGE), all zeros
    padded[N_SW - n_sw:] = ppi_seq[:n_sw]   # place available sweeps at end
    feat = temporal_features(padded)[np.newaxis].astype(np.float32)
    # Input validation: guard against shape/dtype mismatches at the inference boundary
    assert feat.ndim == 4 and feat.shape[1] == 3, \
        f"Expected (1, 3, n_az, n_range), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"
    t0 = time.perf_counter()
    out = session.run(None, {"ppi_sequence": feat})[0][0]
    st.session_state["_last_onnx_ms"] = (time.perf_counter() - t0) * 1000
    return out


def _gru_step(
    sweep: np.ndarray,
    h: np.ndarray,
    noise_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance the ConvGRU by one sweep.

    h is the confidence heatmap (1, 1, n_az, n_range) ∈ [0, 1] — it IS
    the detection probability map, updated in-place each sweep.

    Args:
        sweep       : (n_az, n_range) raw amplitude
        h           : (1, 1, n_az, n_range) confidence heatmap — current hidden state
        noise_floor : (n_range,) current EMA noise floor estimate
    Returns:
        prob_map (n_az, n_range), h_next, noise_floor_next
    """
    gru_session = load_gru_session()
    if gru_session is None:
        return np.zeros((N_AZ, N_RANGE), dtype=np.float32), h, noise_floor

    sweep_norm = (sweep / noise_floor.clip(1e-3)).astype(np.float32)
    noise_floor_next = 0.9 * noise_floor + 0.1 * np.percentile(sweep, 10, axis=0)

    sweep_in = sweep_norm[np.newaxis, np.newaxis]  # (1, 1, n_az, n_range)
    t0 = time.perf_counter()
    prob_map, h_next = gru_session.run(
        None, {"sweep_norm": sweep_in, "h_in": h}
    )
    st.session_state["_last_gru_ms"] = (time.perf_counter() - t0) * 1000
    return prob_map[0, 0], h_next, noise_floor_next


def _gru_step_raw(sweep: np.ndarray, h: np.ndarray,
                  noise_floor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GRU step for benchmarks — same logic as _gru_step, no Streamlit side-effects."""
    gru_session = load_gru_session()
    sweep_norm = (sweep / noise_floor.clip(1e-3)).astype(np.float32)
    noise_floor_next = 0.9 * noise_floor + 0.1 * np.percentile(sweep, 10, axis=0)
    sweep_in = sweep_norm[np.newaxis, np.newaxis]
    prob_map, h_next = gru_session.run(None, {"sweep_norm": sweep_in, "h_in": h})
    return prob_map[0, 0], h_next, noise_floor_next


def _gru_seq(ppi_seq: np.ndarray, positions: list) -> tuple[list[float], list[np.ndarray]]:
    """
    Run the GRU sequentially over all N_SW sweeps.

    Returns:
        conf_history    : per-sweep confidence at the true target cell
        heatmap_history : per-sweep (n_az, n_range) confidence heatmap — the hidden state
    """
    gru_session = load_gru_session()
    if gru_session is None:
        return [0.0] * N_SW, None

    h = np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32)
    noise_floor = np.percentile(ppi_seq[0], 10, axis=0).clip(1e-3)

    conf_history: list[float] = []
    heatmap_history: list[np.ndarray] = []

    for n in range(N_SW):
        prob_map, h, noise_floor = _gru_step(ppi_seq[n], h, noise_floor)
        az_b = int(positions[n][0] / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(positions[n][1], 0, N_RANGE - 1))
        conf_history.append(float(prob_map[az_b, r_b]))
        heatmap_history.append(prob_map.copy())

    return conf_history, heatmap_history


def _cnn_peaks(ml_out: np.ndarray, threshold: float,
               max_peaks: int = 10) -> np.ndarray:
    """
    Threshold CNN probability map and return top-N peaks as (r, az) array.
    Format matches PPIKalmanTracker._cfar_peaks output so the KF can consume
    CNN detections identically to CFAR detections.
    """
    az_idxs, r_idxs = np.where(ml_out > threshold)
    valid = r_idxs >= 5   # match _cfar_peaks near-range clutter exclusion
    az_idxs, r_idxs = az_idxs[valid], r_idxs[valid]
    if len(az_idxs) == 0:
        return np.empty((0, 2))
    scores = ml_out[az_idxs, r_idxs]
    order  = np.argsort(-scores)[:max_peaks]
    return np.column_stack([r_idxs[order], az_idxs[order]]).astype(float)


def _gru_peaks(h_map: np.ndarray, threshold: float = 0.30,
               max_peaks: int = 10) -> np.ndarray:
    """Extract (r, az) peaks from GRU confidence heatmap for KF ingestion."""
    az_idxs, r_idxs = np.where(h_map > threshold)
    valid = r_idxs >= 5
    az_idxs, r_idxs = az_idxs[valid], r_idxs[valid]
    if len(az_idxs) == 0:
        return np.empty((0, 2))
    scores = h_map[az_idxs, r_idxs]
    order = np.argsort(-scores)[:max_peaks]
    return np.column_stack([r_idxs[order], az_idxs[order]]).astype(float)
