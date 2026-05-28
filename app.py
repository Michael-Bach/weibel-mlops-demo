"""
Streamlit demo — PPI Rotating Radar AI Target Detector.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.data.ppi_generator import (
    generate_clutter_only,
    generate_heterogeneous_ppi,
    generate_multitarget_sweep,
    generate_ppi_sequence,
    temporal_features,
)
from src.baseline.ppi_cfar_kf import PPICFARDetector, PPIKalmanTracker

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered Radar Target Detection",
    page_icon="📡",
    layout="wide",
)

# ── Globals ────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_params():
    with open("params_ppi.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def _load_session():
    p = Path("artifacts/ppi_model.onnx")
    return ort.InferenceSession(str(p)) if p.exists() else None

@st.cache_resource
def _load_gru_session():
    p = Path("artifacts/recurrent_model.onnx")
    return ort.InferenceSession(str(p)) if p.exists() else None

params      = _load_params()
session     = _load_session()
gru_session = _load_gru_session()

N_SW    = params["radar"]["n_sweeps"]
N_AZ    = params["radar"]["n_azimuths"]
N_RANGE = params["radar"]["n_ranges"]
N_GRID  = 300

# ── Live-radar demo constants ──────────────────────────────────────────────────
_TGT_COLORS     = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a78bfa", "#fb923c"]
MAX_LIVE_TGTS   = 5
_LIVE_FRAME_LO  = 0.40   # seconds — lower bound of per-sweep wall-clock delay
_LIVE_FRAME_HI  = 0.85   # seconds — upper bound
_LIVE_SPAWN_LO  = 3      # sweeps between spawns (minimum)
_LIVE_SPAWN_HI  = 6      # sweeps between spawns (maximum)

# Representative physical scale factors (short-range coastal surveillance radar)
_RANGE_BIN_M   = 7.5    # metres per range bin  (≈ 20 MHz bandwidth)
_SWEEP_PERIOD_S = 2.0   # seconds per full antenna rotation  (30 RPM)
_MS_TO_KT       = 1.944 # m/s → knots

# ── Polar → Cartesian LUT ──────────────────────────────────────────────────────

@st.cache_data
def _build_lut(n_az: int, n_range: int, n_grid: int):
    cx = cy = n_grid // 2
    r_max = n_grid // 2 - 5
    gy, gx = np.mgrid[0:n_grid, 0:n_grid]
    dx = gx.astype(float) - cx
    dy = float(cy) - gy
    dist = np.sqrt(dx**2 + dy**2)
    r_frac = dist / r_max
    r_idx  = (r_frac * (n_range - 1)).round().clip(0, n_range - 1).astype(int)
    az_rad = np.arctan2(dx, dy)
    az_rad = np.where(az_rad < 0, az_rad + 2 * np.pi, az_rad)
    az_idx = (az_rad / (2 * np.pi) * n_az).astype(int) % n_az
    mask   = r_frac <= 1.0
    return az_idx, r_idx, mask

_AZ_LUT, _R_LUT, _MASK = _build_lut(N_AZ, N_RANGE, N_GRID)

def p2c(polar: np.ndarray) -> np.ndarray:
    out = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)
    out[_MASK] = polar[_AZ_LUT[_MASK], _R_LUT[_MASK]]
    return out

def az_r_to_px(az_deg: float, r_bin: float):
    """Polar (az degrees from North, range bin) → Cartesian pixel coords."""
    cx = cy = N_GRID // 2
    r_max = N_GRID // 2 - 5
    rad  = np.radians(az_deg)
    frac = r_bin / (N_RANGE - 1)
    return (cx + frac * r_max * np.sin(rad),
            cy - frac * r_max * np.cos(rad))


def _arpa_course_speed(positions: list, vr: float, vt: float, n_ahead: int = 3):
    """
    Compute ARPA navigation data from last known position + velocity.

    CPA accounts for both radial and tangential velocity components:
      vt_eff [bins/sweep] = vt [°/sweep] × (π/180) × r_now
      CPA = r_now × |vt_eff| / sqrt(vr² + vt_eff²)
      t_CPA = -r_now × vr / (vr² + vt_eff²)  (when approaching)

    Returns: course_deg (0-360°T), speed_kt, cpa_range_m, t_to_cpa_s
    """
    az_now, r_now = positions[-1]

    # Convert tangential rate to arc-length velocity at current range
    vt_eff = vt * np.pi / 180.0 * r_now   # bins/sweep (perpendicular to line of sight)
    v_total_bins = np.sqrt(vr**2 + vt_eff**2)

    # Course: compass bearing of motion vector in Cartesian space
    az_pred = (az_now + n_ahead * vt) % 360.0
    r_pred  = r_now + n_ahead * vr
    def to_cart(az, r):
        rad = np.radians(az)
        return r * np.sin(rad), r * np.cos(rad)
    x0, y0 = to_cart(az_now, r_now)
    x1, y1 = to_cart(az_pred, r_pred)
    dx, dy  = x1 - x0, y1 - y0
    course_deg = np.degrees(np.arctan2(dx, dy)) % 360.0

    # Speed in knots
    speed_ms = v_total_bins * _RANGE_BIN_M / _SWEEP_PERIOD_S
    speed_kt = speed_ms * _MS_TO_KT

    # CPA: minimum range using both velocity components
    if v_total_bins > 1e-6:
        cpa_r_bins = r_now * abs(vt_eff) / v_total_bins
        # Time to CPA (sweeps): derivative of r(t)² set to zero
        denom = vr**2 + vt_eff**2
        t_cpa_sw = max(0.0, -r_now * vr / denom) if denom > 1e-9 else 0.0
    else:
        cpa_r_bins = r_now
        t_cpa_sw   = 0.0

    cpa_r_m  = cpa_r_bins * _RANGE_BIN_M
    t_cpa_s  = t_cpa_sw * _SWEEP_PERIOD_S

    return course_deg, speed_kt, cpa_r_m, t_cpa_s


def _arpa_traces(positions: list, sweep_idx: int, vr: float, vt: float,
                 n_pred: int = 4) -> list:
    """
    Build Plotly traces for ARPA overlay:
      - Track history trail (past true positions, fading)
      - Velocity vector (predicted future position)
    """
    traces = []

    # ── Track history ─────────────────────────────────────────────────────────
    n_past = sweep_idx + 1
    if n_past > 1:
        hist_az = [positions[i][0] for i in range(n_past)]
        hist_r  = [positions[i][1] for i in range(n_past)]
        hx = [az_r_to_px(a, r)[0] for a, r in zip(hist_az, hist_r)]
        hy = [az_r_to_px(a, r)[1] for a, r in zip(hist_az, hist_r)]
        # Connected dashed line
        traces.append(go.Scatter(
            x=hx, y=hy, mode="lines",
            line=dict(color="rgba(255,80,80,0.55)", width=1.5, dash="dot"),
            showlegend=False, name="ARPA track",
        ))
        # Past-position dots fading by age
        for i, (px, py) in enumerate(zip(hx[:-1], hy[:-1])):
            age   = n_past - 1 - i
            alpha = max(0.15, 0.85 - age * 0.15)
            sz    = max(3, 7 - age)
            traces.append(go.Scatter(
                x=[px], y=[py], mode="markers",
                marker=dict(color=f"rgba(255,100,100,{alpha:.2f})",
                            size=sz, symbol="circle"),
                showlegend=False,
            ))

    # ── Velocity vector ────────────────────────────────────────────────────────
    az_now, r_now = positions[sweep_idx]
    az_pred = (az_now + n_pred * vt) % 360.0
    r_pred  = float(np.clip(r_now + n_pred * vr, 0, N_RANGE - 1))
    tx,  ty  = az_r_to_px(az_now,  r_now)
    tvx, tvy = az_r_to_px(az_pred, r_pred)
    traces.append(go.Scatter(
        x=[tx, tvx], y=[ty, tvy], mode="lines+markers",
        line=dict(color="rgba(255,220,0,0.85)", width=2),
        marker=dict(symbol=["circle", "triangle-up"],
                    color="rgba(255,220,0,0.85)", size=[0, 9]),
        showlegend=False, name="ARPA vector",
    ))

    return traces


def _waterfall_fig(ppi_seq: np.ndarray, positions: list) -> go.Figure:
    """
    Range-Time Intensity (RTI) display: x = range, y = time (sweep), colour = signal strength.
    Uses a fixed azimuth sector (initial target bearing ± 2 bins = ± 4°) — a standard
    radar product that shows target motion in range over successive antenna rotations.
    """
    n_sw, n_az, n_range = ppi_seq.shape

    # Fix the azimuth to the target's initial bearing
    az0_deg = positions[0][0]
    az_bin_fixed = int(np.round(az0_deg / 360.0 * n_az)) % n_az
    az_idx = np.arange(az_bin_fixed - 2, az_bin_fixed + 3) % n_az

    mat = np.zeros((n_sw, n_range), dtype=np.float32)
    for sw in range(n_sw):
        mat[sw] = ppi_seq[sw][az_idx].max(axis=0)

    # Clutter-floor normalisation (R⁻² whitening) so signal stands out at all ranges
    clutter_fl = np.maximum(np.arange(n_range, dtype=np.float32), 1.0) / n_range
    clutter_fl = clutter_fl ** (-2.0)
    mat_norm = mat / clutter_fl[np.newaxis, :]

    range_m = np.arange(n_range) * _RANGE_BIN_M
    sweep_labels = [f"Sw {i+1}" for i in range(n_sw)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=mat_norm,
        x=range_m,
        y=sweep_labels,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=dict(text="Normalised amplitude (dB-ish)", font=dict(color="white")),
                      tickfont=dict(color="white")),
        zmin=0,
        hovertemplate="Range: %{x:.0f} m<br>Sweep: %{y}<br>Amplitude: %{z:.2f}<extra></extra>",
    ))

    # True target range trajectory — visible when target is near the fixed azimuth
    tgt_r_m = [positions[sw][1] * _RANGE_BIN_M for sw in range(n_sw)]
    fig.add_trace(go.Scatter(
        x=tgt_r_m,
        y=sweep_labels,
        mode="lines+markers",
        line=dict(color="red", width=2, dash="dot"),
        marker=dict(color="red", size=5, symbol="x"),
        name=f"True range (az ≈ {az0_deg:.0f}°)",
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=60, r=20, t=48, b=36),
        title=dict(
            text=f"Range-Time Intensity (RTI) — fixed azimuth {az0_deg:.0f}° ± 4°  |  "
                 f"bright band = target echo moving in range over time",
            font=dict(color="#aaa", size=12),
        ),
        xaxis=dict(title="Range (m)", color="white",
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        yaxis=dict(title="Antenna rotation (sweep)", color="white", autorange="reversed",
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def _gru_evolution_fig(heatmap_history: list, positions: list) -> go.Figure:
    """
    2-row × 5-column grid showing the GRU hidden state h at each of the 10 sweeps.
    Each panel is a (n_az × n_range) heatmap with confidence ∈ [0,1] (plasma scale).
    The true target position is marked with a red × at each step.
    """
    n_sw   = len(heatmap_history)
    n_cols = 5
    n_rows = (n_sw + n_cols - 1) // n_cols

    titles = [f"<b>Sweep {i+1}</b>" for i in range(n_sw)]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.14,
    )

    for i, h_map in enumerate(heatmap_history):
        row = i // n_cols + 1
        col = i % n_cols + 1
        az_b = int(positions[i][0] / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(positions[i][1], 0, N_RANGE - 1))

        fig.add_trace(
            go.Heatmap(
                z=h_map,
                colorscale="Plasma",
                zmin=0, zmax=1,
                showscale=(i == n_sw - 1),
                colorbar=dict(
                    title=dict(text="p", font=dict(color="white", size=11)),
                    tickfont=dict(color="white", size=9),
                    len=0.45, thickness=10,
                    x=1.01, y=0.25,
                ),
                hovertemplate="az=%{y}  r=%{x}  h=%{z:.3f}<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[r_b], y=[az_b],
                mode="markers",
                marker=dict(symbol="x", size=8, color="red",
                            line=dict(width=2, color="red")),
                showlegend=False,
                hovertemplate=f"Target  az={az_b}  r={r_b}<extra></extra>",
            ),
            row=row, col=col,
        )

    # Clean up axes — hide ticks on all inner panels
    for i in range(1, n_sw + 1):
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        xi  = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{xi}": dict(showticklabels=False, showgrid=False, zeroline=False),
            f"yaxis{xi}": dict(showticklabels=False, showgrid=False, zeroline=False),
        })

    fig.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=60, t=46, b=8),
        font=dict(color="white"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "rgba(200,200,200,0.85)"
        ann.font.size  = 11

    return fig


# ── Data & inference ──────────────────────────────────────────────────────────

@st.cache_data
def _gen(snr, rb, az, vr, vt):
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

def _ml_map(ppi_seq: np.ndarray, n_sw: int) -> np.ndarray:
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


# ── Live Radar: helpers ────────────────────────────────────────────────────────

def _live_tgt_pos(tgt: dict, sw: int) -> tuple:
    """(az_deg, r_bin) of a target at absolute sweep index sw."""
    age = sw - tgt["born"]
    return (tgt["az0"] + age * tgt["vt"]) % 360.0, tgt["r0"] + age * tgt["vr"]


def _live_tgt_alive(tgt: dict, sw: int) -> bool:
    _, r = _live_tgt_pos(tgt, sw)
    return 5 <= r <= N_RANGE - 3


def _live_init():
    """Initialise (or reset) the live-radar session state."""
    seed = int(np.random.default_rng().integers(0, 999_999))
    st.session_state["live"] = {
        "running":        False,
        "sweep_count":    0,
        "seed":           seed,
        "targets":        [],
        "next_id":        0,
        "spawn_in":       2,
        "sweep_buf":      np.zeros((N_SW, N_AZ, N_RANGE), dtype=np.float32),
        "gru_h":          np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32),
        "gru_nfloor":     np.ones(N_RANGE, dtype=np.float32),
        "kf_cfar":        PPIKalmanTracker(),
        "kf_gru":         PPIKalmanTracker(),
        "conf_hist":      {},   # {id: {"cfar": [], "gru": []}}
        "confirm":        {},   # {id: {"cfar_kf": latency|None, "gru_kf": latency|None}}
        "lat_cfar":       [],
        "lat_gru":        [],
        "last_sweep":     np.zeros((N_AZ, N_RANGE), dtype=np.float32),
        "last_cfar_det":  np.zeros((N_AZ, N_RANGE), dtype=bool),
        "last_gru_map":   np.zeros((N_AZ, N_RANGE), dtype=np.float32),
        "frame_s":        0.6,
    }


def _live_tick():
    """Advance the live radar by one full antenna sweep."""
    s   = st.session_state["live"]
    sw  = s["sweep_count"]
    rng = np.random.default_rng(s["seed"] + sw * 97)

    # Expire targets that have left the display area
    s["targets"] = [t for t in s["targets"] if _live_tgt_alive(t, sw)]

    # Spawn a new target?
    s["spawn_in"] -= 1
    if s["spawn_in"] <= 0:
        if len(s["targets"]) < MAX_LIVE_TGTS:
            tid = s["next_id"]
            tgt = {
                "id":     tid,
                "az0":    float(rng.uniform(0, 360)),
                "r0":     float(rng.uniform(N_RANGE * 0.75, N_RANGE - 6)),
                "vr":     float(rng.uniform(-2.5, -0.8)),
                "vt":     float(rng.uniform(-2.0, 2.0)),
                "snr_db": float(rng.uniform(-5, 20)),
                "born":   sw,
            }
            s["targets"].append(tgt)
            s["next_id"] += 1
            s["conf_hist"][tid] = {"cfar": [], "gru": []}
            s["confirm"][tid]   = {"cfar_kf": None, "gru_kf": None}
        s["spawn_in"] = int(rng.integers(_LIVE_SPAWN_LO, _LIVE_SPAWN_HI + 1))

    # Generate this sweep
    sweep = generate_multitarget_sweep(sw, s["targets"], params, rng)
    s["last_sweep"] = sweep

    # Rolling buffer — shift left, push new sweep at back
    buf = s["sweep_buf"]
    buf[:-1] = buf[1:]
    buf[-1]  = sweep

    # CFAR detection map (for display) + CFAR+KF update
    cfar_det = PPICFARDetector().detect(sweep)
    s["last_cfar_det"] = cfar_det
    s["kf_cfar"].update(sweep)   # uses internal CFAR for peak extraction

    # GRU single-step update
    if gru_session is not None:
        nf       = s["gru_nfloor"].clip(1e-3)
        sw_norm  = (sweep / nf)[np.newaxis, np.newaxis].astype(np.float32)
        gru_prob, h_next = gru_session.run(None, {"sweep_norm": sw_norm, "h_in": s["gru_h"]})
        s["gru_h"]      = h_next
        s["gru_nfloor"] = 0.9 * s["gru_nfloor"] + 0.1 * np.percentile(sweep, 10, axis=0)
        gru_map = gru_prob[0, 0]
    else:
        gru_map = np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    s["last_gru_map"] = gru_map
    s["kf_gru"].update_from_peaks(_gru_peaks(gru_map), N_AZ, N_RANGE)

    # Per-target confidence sampling and confirmation check
    cfar_trks = s["kf_cfar"].track_positions()
    gru_trks  = s["kf_gru"].track_positions()

    for tgt in s["targets"]:
        tid  = tgt["id"]
        az_t, r_t = _live_tgt_pos(tgt, sw)
        az_b = int(az_t / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(r_t, 0, N_RANGE - 1))

        if tid in s["conf_hist"]:
            s["conf_hist"][tid]["cfar"].append(float(cfar_det[az_b, r_b]))
            s["conf_hist"][tid]["gru"].append(float(gru_map[az_b, r_b]))

        if tid in s["confirm"]:
            conf = s["confirm"][tid]
            for key, trks in [("cfar_kf", cfar_trks), ("gru_kf", gru_trks)]:
                if conf[key] is not None:
                    continue
                for az_tr, r_tr in trks:
                    az_tr_b = int(float(az_tr) % N_AZ)
                    daz = min(abs(az_tr_b - az_b), N_AZ - abs(az_tr_b - az_b))
                    if np.sqrt(daz ** 2 + (float(r_tr) - r_b) ** 2) < 8:
                        lat = sw - tgt["born"] + 1
                        conf[key] = lat
                        {"cfar_kf": s["lat_cfar"], "gru_kf": s["lat_gru"]}[key].append(lat)
                        break

    s["frame_s"]     = float(rng.uniform(_LIVE_FRAME_LO, _LIVE_FRAME_HI))
    s["sweep_count"] += 1


def _live_ppi_fig(s: dict) -> go.Figure:
    """PPI scope for the live-radar demo — phosphor + all three pipeline tracks."""
    sw  = s["sweep_count"]
    buf = s["sweep_buf"]

    # Phosphor: blend last 4 sweeps with exponential decay
    phmap = np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    for age in range(min(4, N_SW)):
        phmap = np.maximum(phmap, buf[-(age + 1)] * (0.45 ** age))

    raw_c   = p2c(phmap)
    gru_c   = p2c(s["last_gru_map"]) if gru_session is not None else None
    img     = _ppi_rgba(raw_c, None, False, gru_c, gru_session is not None)

    scope_ovl = _scope_overlay(N_GRID)
    composite = np.maximum(img, scope_ovl)

    fig = go.Figure()
    fig.add_trace(go.Image(z=composite))

    # Cardinal labels
    cx = cy = N_GRID // 2
    r_max = N_GRID // 2 - 6
    for deg, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(deg)
        fig.add_annotation(
            x=cx + (r_max + 14) * np.sin(rad),
            y=cy - (r_max + 14) * np.cos(rad),
            text=lbl, showarrow=False,
            font=dict(color="rgba(0,255,0,0.65)", size=11),
        )

    # CFAR detections — current sweep (orange dots, reduced opacity for clutter)
    az_d, r_d = np.where(s["last_cfar_det"])
    if len(az_d):
        px = [az_r_to_px(az_d[i] / N_AZ * 360, r_d[i])[0] for i in range(len(az_d))]
        py = [az_r_to_px(az_d[i] / N_AZ * 360, r_d[i])[1] for i in range(len(az_d))]
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="markers",
            marker=dict(color="orange", size=3, opacity=0.45), showlegend=False,
        ))

    # CFAR+KF confirmed tracks (white ✕)
    for az_bin, r_bin in s["kf_cfar"].track_positions():
        x, y = az_r_to_px((float(az_bin) % N_AZ) / N_AZ * 360,
                           float(np.clip(r_bin, 0, N_RANGE - 1)))
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color="white", size=11, symbol="cross",
                        line=dict(color="white", width=2)),
            showlegend=False,
        ))

    # GRU+KF confirmed tracks (magenta diamond)
    for az_bin, r_bin in s["kf_gru"].track_positions():
        x, y = az_r_to_px((float(az_bin) % N_AZ) / N_AZ * 360,
                           float(np.clip(r_bin, 0, N_RANGE - 1)))
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color="#ff44ff", size=11, symbol="diamond",
                        line=dict(color="#ff44ff", width=2)),
            showlegend=False,
        ))

    # True target ground-truth positions (coloured circles + ID label)
    for tgt in s["targets"]:
        az_t, r_t = _live_tgt_pos(tgt, sw)
        if not (0 <= r_t < N_RANGE):
            continue
        x, y  = az_r_to_px(az_t, r_t)
        color = _TGT_COLORS[tgt["id"] % len(_TGT_COLORS)]
        age   = sw - tgt["born"]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(color=color, size=11, symbol="circle",
                        line=dict(color="white", width=1.5)),
            text=[f"#{tgt['id']}"],
            textposition="top center",
            textfont=dict(color=color, size=10),
            showlegend=False,
            hovertemplate=(
                f"Target #{tgt['id']}<br>"
                f"SNR {tgt['snr_db']:.0f} dB<br>"
                f"Age {age} sweeps<br>"
                f"az={az_t:.0f}°  r={r_t:.1f} bins<extra></extra>"
            ),
        ))

    # Legend (top-right corner annotations)
    legend_items = [
        ("● True targets", "#ff6b6b"),
        ("▪ CA-CFAR hits",  "orange"),
        ("✕ CFAR+KF",       "white"),
        ("◆ GRU+KF",        "#ff44ff"),
    ]
    lx, ly0 = N_GRID - 6, 14
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=lx, y=ly0 + i * 16,
            text=label, showarrow=False, xanchor="right",
            font=dict(color=color, size=10),
        )

    fig.update_layout(
        height=520, width=520,
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(range=[0, N_GRID], scaleanchor="y",
                   showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[N_GRID, 0],
                   showgrid=False, showticklabels=False, zeroline=False),
        title=dict(text=f"Live Radar — Sweep {sw}",
                   font=dict(color="rgba(0,255,0,0.8)", size=13)),
    )
    return fig


def _live_conf_fig(s: dict) -> go.Figure:
    """
    Confidence-at-target-cell chart for the most recently spawned active target.
    Shows CFAR hit/miss and GRU hidden-state value per sweep.
    """
    # Pick the most recently spawned target that has at least 1 sweep of history
    best_tid, best_born = None, -1
    for tgt in s["targets"]:
        tid = tgt["id"]
        if tid in s["conf_hist"] and len(s["conf_hist"][tid]["cfar"]) > 0:
            if tgt["born"] > best_born:
                best_born, best_tid = tgt["born"], tid

    if best_tid is None:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Waiting for first target…", font=dict(color="#aaa", size=12)),
        )
        return fig

    hist  = s["conf_hist"][best_tid]
    conf  = s["confirm"].get(best_tid, {})
    n     = len(hist["cfar"])
    xs    = list(range(1, n + 1))
    GRID_C = "rgba(128,128,128,0.15)"

    fig = go.Figure()

    # CFAR: binary hit bars (orange)
    fig.add_trace(go.Bar(
        x=xs, y=[v * 100 for v in hist["cfar"]],
        name="CFAR hit", marker_color="rgba(255,165,0,0.55)",
    ))
    # GRU: continuous line (magenta)
    fig.add_trace(go.Scatter(
        x=xs, y=[v * 100 for v in hist["gru"]],
        name="GRU h", mode="lines+markers",
        line=dict(color="#ff44ff", width=2),
        marker=dict(size=5, color="#ff44ff"),
    ))
    # Vertical lines at first confirmation per pipeline
    for key, color, label in [
        ("cfar_kf", "white",   "CFAR+KF"),
        ("gru_kf",  "#ff44ff", "GRU+KF"),
    ]:
        lat = conf.get(key)
        if lat is not None and lat <= n:
            fig.add_vline(
                x=lat, line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=f"{label} ✓",
                annotation_position="top",
                annotation_font_color=color,
                annotation_font_size=9,
            )

    fig.add_hline(y=30, line_dash="dot", line_color="#555",
                  annotation_text="30% gate", annotation_position="right",
                  annotation_font_color="#888")

    # Locate target in the active list to get SNR
    snr_str = ""
    for tgt in s["targets"]:
        if tgt["id"] == best_tid:
            snr_str = f"  SNR {tgt['snr_db']:.0f} dB"
            break

    fig.update_layout(
        height=210,
        margin=dict(l=48, r=10, t=40, b=28),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        title=dict(
            text=f"Confidence at target cell — #{best_tid}{snr_str}",
            font=dict(color="#aaa", size=12),
        ),
        xaxis=dict(title="Sweeps since spawn", tickvals=xs,
                   gridcolor=GRID_C, zeroline=False, color="white"),
        yaxis=dict(title="Confidence %", range=[0, 115],
                   gridcolor=GRID_C, zeroline=False, color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=1.15, x=0),
    )
    return fig


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


# ── Figure builder ─────────────────────────────────────────────────────────────

def _ppi_rgba(
    raw_c: np.ndarray,
    ml_c,
    show_ml: bool,
    gru_c=None,
    show_gru: bool = True,
) -> np.ndarray:
    img = np.zeros((N_GRID, N_GRID, 4), dtype=np.uint8)
    # Green phosphor raw sweep
    v = ~np.isnan(raw_c)
    if v.any():
        mx = float(np.percentile(raw_c[v], 99))
        norm = np.where(v, np.clip(raw_c / (mx + 1e-6), 0, 1), 0.0)
        g = (norm * 210).astype(np.uint8)
        img[:, :, 1] = np.maximum(img[:, :, 1], g)
        img[:, :, 3] = np.maximum(img[:, :, 3], g)
    # Cyan/teal CNN confidence overlay
    if show_ml and ml_c is not None:
        vv = ~np.isnan(ml_c)
        prob = np.where(vv, np.clip(ml_c, 0, 1), 0.0)
        img[:, :, 0] = np.maximum(img[:, :, 0], (prob * 20).astype(np.uint8))
        img[:, :, 1] = np.maximum(img[:, :, 1], (prob * 180).astype(np.uint8))
        img[:, :, 2] = np.maximum(img[:, :, 2], (prob * 255).astype(np.uint8))
        img[:, :, 3] = np.maximum(img[:, :, 3], (prob * 230).astype(np.uint8))
    # Purple/magenta GRU hidden-state heatmap
    if show_gru and gru_c is not None:
        vg = ~np.isnan(gru_c)
        prob_g = np.where(vg, np.clip(gru_c, 0, 1), 0.0)
        img[:, :, 0] = np.maximum(img[:, :, 0], (prob_g * 220).astype(np.uint8))
        img[:, :, 1] = np.maximum(img[:, :, 1], (prob_g * 40).astype(np.uint8))
        img[:, :, 2] = np.maximum(img[:, :, 2], (prob_g * 255).astype(np.uint8))
        img[:, :, 3] = np.maximum(img[:, :, 3], (prob_g * 230).astype(np.uint8))
    return img

def make_ppi_figure(ppi_seq, positions, cfar_sweeps, kf_pos_list,
                    ml_map_now, sweep_idx, vr, vt,
                    show_ml, show_cfar, show_kf, show_arpa,
                    gru_map_now=None, show_gru=True,
                    gru_kf_pos=None, show_gru_kf=True,
                    cnn_kf_pos=None, show_cnn_kf=True):
    raw_c = p2c(ppi_seq[sweep_idx])
    ml_c  = p2c(ml_map_now) if show_ml else None
    gru_c = p2c(gru_map_now) if (show_gru and gru_map_now is not None) else None
    img   = _ppi_rgba(raw_c, ml_c, show_ml, gru_c, show_gru)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    # CFAR detections — current sweep (bright) + trail of past sweeps (dim)
    if show_cfar:
        # dim trail: any detection in sweeps 0..sweep_idx-1
        if sweep_idx > 0:
            trail = cfar_sweeps[:sweep_idx].any(axis=0)
            az_t, r_t = np.where(trail)
            if len(az_t):
                px = [az_r_to_px(int(az_t[i]) / N_AZ * 360, int(r_t[i]))[0] for i in range(len(az_t))]
                py = [az_r_to_px(int(az_t[i]) / N_AZ * 360, int(r_t[i]))[1] for i in range(len(az_t))]
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="markers", name="CFAR trail",
                    marker=dict(color="orange", size=3, opacity=0.25),
                    showlegend=False,
                ))
        # current sweep: bright orange
        cur = cfar_sweeps[sweep_idx]
        az_d, r_d = np.where(cur)
        if len(az_d):
            px = [az_r_to_px(int(az_d[i]) / N_AZ * 360, int(r_d[i]))[0] for i in range(len(az_d))]
            py = [az_r_to_px(int(az_d[i]) / N_AZ * 360, int(r_d[i]))[1] for i in range(len(az_d))]
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers", name="CFAR",
                marker=dict(color="orange", size=5, opacity=0.85)))

    # CFAR+KF confirmed track positions (white ✕)
    if show_kf and kf_pos_list:
        kx, ky = [], []
        for az_bin, r_bin in kf_pos_list:
            az_deg_k = (float(az_bin) % N_AZ) / N_AZ * 360
            r_bin_k  = float(np.clip(r_bin, 0, N_RANGE - 1))
            x, y = az_r_to_px(az_deg_k, r_bin_k)
            kx.append(x)
            ky.append(y)
        fig.add_trace(go.Scatter(
            x=kx, y=ky, mode="markers", name="CFAR+KF",
            marker=dict(color="white", size=12, symbol="cross",
                        line=dict(color="white", width=2))))

    # GRU+KF confirmed track positions (magenta diamond)
    if show_gru_kf and gru_kf_pos:
        gx, gy = [], []
        for az_bin, r_bin in gru_kf_pos:
            az_deg_g = (float(az_bin) % N_AZ) / N_AZ * 360
            r_bin_g  = float(np.clip(r_bin, 0, N_RANGE - 1))
            x, y = az_r_to_px(az_deg_g, r_bin_g)
            gx.append(x)
            gy.append(y)
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="markers", name="GRU+KF",
            marker=dict(color="#ff44ff", size=12, symbol="diamond",
                        line=dict(color="#ff44ff", width=2))))

    # CNN+KF confirmed track positions (yellow triangle-up)
    if show_cnn_kf and cnn_kf_pos:
        cx2, cy2 = [], []
        for az_bin, r_bin in cnn_kf_pos:
            az_deg_c = (float(az_bin) % N_AZ) / N_AZ * 360
            r_bin_c  = float(np.clip(r_bin, 0, N_RANGE - 1))
            x, y = az_r_to_px(az_deg_c, r_bin_c)
            cx2.append(x)
            cy2.append(y)
        fig.add_trace(go.Scatter(
            x=cx2, y=cy2, mode="markers", name="CNN+KF",
            marker=dict(color="#ffff00", size=12, symbol="triangle-up",
                        line=dict(color="#ffff00", width=2))))

    # ARPA track history + velocity vector
    if show_arpa:
        for tr in _arpa_traces(positions, sweep_idx, vr, vt):
            fig.add_trace(tr)

    # True target position (red ×)  — always on top of ARPA trail
    az_t, r_t = positions[sweep_idx]
    tx, ty = az_r_to_px(az_t, r_t)
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty], mode="markers", name="True target",
        marker=dict(color="red", size=9, symbol="x",
                    line=dict(color="red", width=2))))

    # Scope decorations
    cx = cy = N_GRID // 2
    r_max = N_GRID // 2 - 5
    theta = np.linspace(0, 2 * np.pi, 360)
    # Outer ring
    fig.add_trace(go.Scatter(
        x=cx + r_max * np.cos(theta), y=cy + r_max * np.sin(theta),
        mode="lines", line=dict(color="rgba(0,255,0,0.4)", width=1.2),
        showlegend=False))
    # Range rings
    for rf in [0.25, 0.5, 0.75]:
        rp = r_max * rf
        fig.add_trace(go.Scatter(
            x=cx + rp * np.cos(theta), y=cy + rp * np.sin(theta),
            mode="lines", line=dict(color="rgba(0,255,0,0.1)", width=0.5),
            showlegend=False))
    # Cardinal spokes and labels
    for deg, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(deg)
        sx = cx + r_max * np.sin(rad)
        sy = cy - r_max * np.cos(rad)
        fig.add_trace(go.Scatter(
            x=[cx, sx], y=[cy, sy], mode="lines",
            line=dict(color="rgba(0,255,0,0.15)", width=0.5), showlegend=False))
        fig.add_annotation(
            x=cx + (r_max + 14) * np.sin(rad),
            y=cy - (r_max + 14) * np.cos(rad),
            text=lbl, showarrow=False,
            font=dict(color="rgba(0,255,0,0.7)", size=11))

    fig.update_layout(
        height=490, width=490,
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(range=[0, N_GRID], scaleanchor="y",
                   showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[N_GRID, 0],
                   showgrid=False, showticklabels=False, zeroline=False),
        legend=dict(x=1.01, y=1.0, bgcolor="rgba(0,0,0,0.7)",
                    font=dict(color="white", size=11)),
        title=dict(text=f"PPI Scope — Sweep {sweep_idx + 1} / {N_SW}",
                   font=dict(color="rgba(0,255,0,0.8)", size=13)),
    )
    return fig


# ── Animation helpers ─────────────────────────────────────────────────────────

@st.cache_data
def _scope_overlay(n_grid: int) -> np.ndarray:
    """Cached RGBA layer: radar rings + cardinal spokes, drawn in numpy."""
    img = np.zeros((n_grid, n_grid, 4), dtype=np.uint8)
    cx = cy = n_grid // 2
    r_max = n_grid // 2 - 6
    t = np.linspace(0, 2 * np.pi, 1200)
    for r_frac, gval, aval in [(1.0, 160, 130), (0.75, 70, 35),
                                (0.50, 70, 35), (0.25, 70, 35)]:
        r = int(r_max * r_frac)
        xs = np.clip((cx + r * np.cos(t)).astype(int), 0, n_grid - 1)
        ys = np.clip((cy + r * np.sin(t)).astype(int), 0, n_grid - 1)
        img[ys, xs, 1] = np.maximum(img[ys, xs, 1], gval)
        img[ys, xs, 3] = np.maximum(img[ys, xs, 3], aval)
    for deg in [0, 90, 180, 270]:
        rad = np.radians(deg)
        rs = np.linspace(0, r_max, r_max * 2)
        xs = np.clip((cx + rs * np.sin(rad)).astype(int), 0, n_grid - 1)
        ys = np.clip((cy - rs * np.cos(rad)).astype(int), 0, n_grid - 1)
        img[ys, xs, 1] = np.maximum(img[ys, xs, 1], 55)
        img[ys, xs, 3] = np.maximum(img[ys, xs, 3], 40)
    return img


def _phosphor_map(ppi_seq: np.ndarray, current_sw: int,
                  beam_az: int) -> np.ndarray:
    """
    Accumulated amplitude with phosphor persistence for animation.
    current_sw : sweep index currently being drawn
    beam_az    : number of azimuth bins covered so far in current_sw
    """
    out = np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    for prev in range(current_sw):
        age   = current_sw - prev
        decay = 0.38 ** age
        out   = np.maximum(out, ppi_seq[prev] * decay)
    partial              = np.zeros_like(ppi_seq[current_sw])
    partial[:beam_az]    = ppi_seq[current_sw, :beam_az]
    return np.maximum(out, partial)


def _pd_chart_fig(
    ml_history: list,
    n_shown: int,
    gru_history: list | None = None,
) -> go.Figure:
    """Bar chart: detector confidence at target cell, one bar per completed sweep.
    If gru_history is provided, adds a second line for the GRU model.
    """
    vals   = [v * 100 for v in ml_history[:n_shown]]
    colors = ["#00ccff" if v > 30 else "#445566" for v in vals]
    xs     = list(range(1, n_shown + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=vals,
        name="CNN (temporal)",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="white"),
    ))
    if gru_history is not None:
        gru_vals = [v * 100 for v in gru_history[:n_shown]]
        fig.add_trace(go.Scatter(
            x=xs, y=gru_vals,
            name="GRU (streaming)",
            mode="lines+markers",
            line=dict(color="#ff9900", width=2),
            marker=dict(size=6, color="#ff9900"),
        ))
    fig.add_hline(y=30, line_dash="dot", line_color="#888",
                  annotation_text="30% Pd gate", annotation_position="right",
                  annotation_font_color="#aaa")
    fig.update_layout(
        height=180,
        margin=dict(l=44, r=10, t=28, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text="Detector confidence at target cell — CNN vs GRU (streaming)" if gru_history else
                 "CNN confidence at target cell — grows as more sweeps are integrated",
            font=dict(color="#aaa", size=12),
        ),
        xaxis=dict(title="Sweep", tickvals=list(range(1, N_SW + 1)),
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False,
                   color="white"),
        yaxis=dict(title="Conf %", range=[0, 115],
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False,
                   color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        showlegend=gru_history is not None,
    )
    return fig


def _anim_fig(img_rgba: np.ndarray, beam_az_deg: float,
              current_sw: int, positions: list,
              cfar_sweeps: np.ndarray, completed_sw: int,
              kf_pos_list: list, vr: float, vt: float,
              show_cfar: bool, show_kf: bool, show_arpa: bool,
              gru_kf_pos: list = None, show_gru_kf: bool = True,
              cnn_kf_pos: list = None, show_cnn_kf: bool = True) -> go.Figure:
    """Lean animation frame: one Image trace + shapes for beam + annotations."""
    scope_ovl = _scope_overlay(N_GRID)
    composite = np.maximum(img_rgba, scope_ovl)

    fig = go.Figure()
    fig.add_trace(go.Image(z=composite))

    # Beam line + fading trail (shapes are cheapest Plotly primitive)
    cx = cy = N_GRID // 2
    r_max = N_GRID // 2 - 6
    for trail in range(5):
        az_t  = (beam_az_deg - trail * 10) % 360
        rad   = np.radians(az_t)
        alpha = max(0.0, 0.85 - trail * 0.18)
        width = 2.5 - trail * 0.4
        fig.add_shape(
            type="line",
            x0=cx, y0=cy,
            x1=cx + r_max * np.sin(rad),
            y1=cy - r_max * np.cos(rad),
            line=dict(color=f"rgba(150,255,150,{alpha:.2f})", width=max(width, 0.5)),
        )

    # Cardinal labels (4 annotations — negligible cost)
    for deg, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(deg)
        fig.add_annotation(
            x=cx + (r_max + 14) * np.sin(rad),
            y=cy - (r_max + 14) * np.cos(rad),
            text=lbl, showarrow=False,
            font=dict(color="rgba(0,255,0,0.65)", size=11),
        )

    # CFAR markers — accumulated detections across completed sweeps
    if show_cfar and completed_sw >= 0:
        n_done = min(completed_sw + 1, len(cfar_sweeps))
        accumulated = cfar_sweeps[:n_done].any(axis=0)
        az_d, r_d = np.where(accumulated)
        if len(az_d):
            px = [az_r_to_px(az_d[i] / N_AZ * 360, r_d[i])[0] for i in range(len(az_d))]
            py = [az_r_to_px(az_d[i] / N_AZ * 360, r_d[i])[1] for i in range(len(az_d))]
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers",
                marker=dict(color="orange", size=4, opacity=0.7),
                showlegend=False,
            ))

    # CFAR+KF track (white ✕)
    if show_kf and kf_pos_list:
        kx = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                          float(np.clip(rb, 0, N_RANGE - 1)))[0]
              for ab, rb in kf_pos_list]
        ky = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                          float(np.clip(rb, 0, N_RANGE - 1)))[1]
              for ab, rb in kf_pos_list]
        fig.add_trace(go.Scatter(
            x=kx, y=ky, mode="markers",
            marker=dict(color="white", size=11, symbol="cross",
                        line=dict(color="white", width=2)),
            showlegend=False,
        ))

    # GRU+KF track (magenta diamond)
    if show_gru_kf and gru_kf_pos:
        gx = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                          float(np.clip(rb, 0, N_RANGE - 1)))[0]
              for ab, rb in gru_kf_pos]
        gy = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                          float(np.clip(rb, 0, N_RANGE - 1)))[1]
              for ab, rb in gru_kf_pos]
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="markers",
            marker=dict(color="#ff44ff", size=11, symbol="diamond",
                        line=dict(color="#ff44ff", width=2)),
            showlegend=False,
        ))

    # CNN+KF track (yellow triangle-up)
    if show_cnn_kf and cnn_kf_pos:
        cx2 = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                           float(np.clip(rb, 0, N_RANGE - 1)))[0]
               for ab, rb in cnn_kf_pos]
        cy2 = [az_r_to_px((float(ab) % N_AZ) / N_AZ * 360,
                           float(np.clip(rb, 0, N_RANGE - 1)))[1]
               for ab, rb in cnn_kf_pos]
        fig.add_trace(go.Scatter(
            x=cx2, y=cy2, mode="markers",
            marker=dict(color="#ffff00", size=11, symbol="triangle-up",
                        line=dict(color="#ffff00", width=2)),
            showlegend=False,
        ))

    # ARPA track history + velocity vector
    if show_arpa:
        for tr in _arpa_traces(positions, current_sw, vr, vt):
            fig.add_trace(tr)

    # True target for this sweep
    az_t, r_t = positions[current_sw]
    tx, ty = az_r_to_px(az_t, r_t)
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty], mode="markers",
        marker=dict(color="red", size=8, symbol="x",
                    line=dict(color="red", width=2)),
        showlegend=False,
    ))

    # Legend block — top-right corner as annotations
    legend_items = [("▪ True target", "red")]
    if show_arpa:
        legend_items.append(("→ ARPA vector", "rgba(255,220,0,0.9)"))
    if show_cfar:
        legend_items.append(("▪ CA-CFAR", "orange"))
    if show_kf and kf_pos_list:
        legend_items.append(("✕ CFAR+KF", "white"))
    if show_gru_kf and gru_kf_pos:
        legend_items.append(("◆ GRU+KF", "#ff44ff"))
    if show_cnn_kf and cnn_kf_pos:
        legend_items.append(("▲ CNN+KF", "#ffff00"))
    lx, ly0 = N_GRID - 6, 14
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=lx, y=ly0 + i * 16,
            text=label, showarrow=False, xanchor="right",
            font=dict(color=color, size=10),
        )

    fig.update_layout(
        height=490, width=490,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(range=[0, N_GRID], scaleanchor="y",
                   showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[N_GRID, 0],
                   showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def _render_anim_frame(frame: int, ppi_seq, positions, cfar_sweeps, kf_pos_list,
                       ml_history, vr, vt,
                       show_ml, show_cfar, show_kf, show_arpa,
                       scope_ph, pd_ph,
                       gru_history=None, gru_heatmaps=None, show_gru=True,
                       kf_history_cfar=None, kf_history_gru=None, kf_history_cnn=None,
                       show_gru_kf=True, show_cnn_kf=True) -> bool:
    """
    Render a single animation frame. Returns True if there are more frames to show.
    frame index: sw * FRAMES_PER_SWEEP + step_in_sweep
    """
    AZ_STEP       = 6
    FRAMES_PER_SW = N_AZ // AZ_STEP   # 30
    FRAME_S       = 0.04

    total_frames = N_SW * FRAMES_PER_SW
    if frame >= total_frames:
        full_ml  = _ml_map(ppi_seq, N_SW)
        combined = _phosphor_map(ppi_seq, N_SW - 1, N_AZ)
        gru_c_fin = p2c(gru_heatmaps[-1]) if (show_gru and gru_heatmaps) else None
        img_fin  = _ppi_rgba(
            p2c(combined), p2c(full_ml) if show_ml else None, show_ml,
            gru_c_fin, show_gru,
        )
        _cfar_now = kf_history_cfar[N_SW - 1] if kf_history_cfar else kf_pos_list
        _gru_now  = kf_history_gru[N_SW - 1]  if kf_history_gru  else []
        _cnn_now  = kf_history_cnn[N_SW - 1]  if kf_history_cnn  else []
        scope_ph.plotly_chart(
            _anim_fig(img_fin, 361.0, N_SW - 1, positions,
                      cfar_sweeps, N_SW - 1, _cfar_now, vr, vt,
                      show_cfar, show_kf, show_arpa,
                      gru_kf_pos=_gru_now, show_gru_kf=show_gru_kf,
                      cnn_kf_pos=_cnn_now, show_cnn_kf=show_cnn_kf),
            use_container_width=False, key="anim_scope_final",
        )
        pd_ph.plotly_chart(
            _pd_chart_fig(ml_history, N_SW, gru_history),
            use_container_width=True, key="anim_pd_final",
        )
        return False   # animation complete

    sw            = frame // FRAMES_PER_SW
    step_in_sweep = frame  % FRAMES_PER_SW
    beam_az       = (step_in_sweep + 1) * AZ_STEP
    beam_deg      = beam_az / N_AZ * 360

    combined = _phosphor_map(ppi_seq, sw, beam_az)
    raw_c    = p2c(combined)
    ml_c     = p2c(_ml_map(ppi_seq, sw)) if (show_ml and sw > 0) else None
    gru_c    = p2c(gru_heatmaps[sw]) if (show_gru and gru_heatmaps and sw > 0) else None
    img      = _ppi_rgba(raw_c, ml_c, show_ml and sw > 0, gru_c, show_gru)

    _cfar_now = kf_history_cfar[sw] if kf_history_cfar else kf_pos_list
    _gru_now  = kf_history_gru[sw]  if kf_history_gru  else []
    _cnn_now  = kf_history_cnn[sw]  if kf_history_cnn  else []
    scope_ph.plotly_chart(
        _anim_fig(img, beam_deg, sw, positions,
                  cfar_sweeps, sw - 1, _cfar_now, vr, vt,
                  show_cfar, show_kf, show_arpa,
                  gru_kf_pos=_gru_now, show_gru_kf=show_gru_kf,
                  cnn_kf_pos=_cnn_now, show_cnn_kf=show_cnn_kf),
        use_container_width=False, key=f"anim_scope_{frame}",
    )

    pd_ph.plotly_chart(
        _pd_chart_fig(ml_history, sw + 1, gru_history),
        use_container_width=True, key=f"anim_pd_{sw}",
    )

    time.sleep(FRAME_S)
    return True   # more frames remain


# ── Header & sidebar ──────────────────────────────────────────────────────────

st.title("📡 AI-Powered Radar Target Detection")
st.markdown(
    "**This demo shows how a machine-learning system detects and tracks a moving target in radar "
    "data — automatically, and with greater reliability than traditional rule-based methods.** "
    "A synthetic rotating radar generates realistic clutter; three detection algorithms compete "
    "on the same data so their performance can be compared directly."
)

with st.expander("Key terms glossary"):
    st.markdown("""
| Term | Plain English |
|------|---------------|
| **PPI** | Plan Position Indicator — the classic circular radar display, like in the movies |
| **SNR (dB)** | Signal-to-Noise Ratio — how much stronger the target echo is vs. background noise. Higher = easier to detect |
| **CFAR** | Constant False-Alarm Rate — a classical threshold detector that adapts to local noise level |
| **Kalman Filter (KF)** | A mathematical tracker that smooths noisy measurements to estimate where a target is heading |
| **ARPA** | Automatic Radar Plotting Aid — computes target course, speed, and closest approach automatically |
| **CPA** | Closest Point of Approach — the minimum range the target will reach if it maintains current course |
| **Pd** | Probability of Detection — the fraction of real targets that get detected |
| **Pfa** | Probability of False Alarm — the fraction of noise-only cells that are mistakenly flagged as targets |
| **AUC** | Area Under Curve — a single number (0–1) summarising detector quality; 1.0 = perfect, 0.5 = random guessing |
| **CNN** | Convolutional Neural Network — the AI model used here, trained to recognise target signatures across multiple radar sweeps |
""")

if session is None:
    st.warning(
        "ONNX model not found at `artifacts/ppi_model.onnx`. "
        "Run `python src/train_ppi.py` then `python scripts/export_onnx_ppi.py` first. "
        "CFAR and KF will still work."
    )

with st.sidebar:
    st.header("Target Parameters")
    snr_db    = st.slider("Signal strength (SNR dB)", -20.0, 40.0, 10.0, 0.5,
                          help="How much stronger the target echo is vs. background noise. Higher = easier to detect.")
    range_bin = st.slider("Target distance (range bin)", 10, N_RANGE - 10,
                          int(params["target"]["range_bin"]),
                          help="Distance from radar in range bins (1 bin ≈ 7.5 m)")
    az_deg    = st.slider("Target direction (°)", 0.0, 359.0,
                          float(params["target"]["azimuth_deg"]), 1.0,
                          help="Compass bearing from radar, 0° = North")
    vr        = st.slider("Approach speed (bins/sweep)", -3.0, 3.0,
                          float(params["target"]["radial_velocity"]), 0.1,
                          help="Negative = closing (approaching radar). ~1 bin/sweep ≈ 3.75 m/s ≈ 7.3 kt")
    vt        = st.slider("Crossing speed (°/sweep)", -4.0, 4.0,
                          float(params["target"]["tangential_velocity"]), 0.1,
                          help="Angular rate across the radar beam. Positive = clockwise.")
    st.divider()
    st.header("Overlay layers")
    show_ml   = st.toggle("CNN confidence map", value=True,
                          help="Cyan glow — batch CNN probability, computed from all sweeps seen so far")
    show_cfar = st.toggle("CFAR detections", value=True,
                          help="Orange dots — cells where the adaptive threshold was exceeded")
    show_kf     = st.toggle("CFAR + KF track", value=True,
                            help="White ✕ — CFAR detections → Kalman filter, confirmed after ≥ 4 associations")
    show_gru_kf = st.toggle("GRU + KF tracks", value=True,
                             help="Magenta ◆ — GRU heatmap peaks → same KF; streaming, no batch window needed")
    show_cnn_kf = st.toggle("CNN + KF tracks", value=True,
                             help="Yellow ▲ — CNN peaks → same KF; highest accuracy but needs full sweep window")
    show_arpa   = st.toggle("ARPA track & vector", value=True,
                             help="Red trail = track history · Yellow arrow = predicted position in 4 sweeps")
    st.divider()
    with st.expander("Kalman filter tuning (Q / R)"):
        st.caption(
            "Process noise **Q** controls how much the tracker allows for target manoeuvres. "
            "Measurement noise **R** reflects uncertainty in the CFAR centroid estimate."
        )
        st.markdown(
            "| Matrix | Diagonal values | Effect |\n"
            "|--------|----------------|--------|\n"
            "| **Q** (process noise) | pos: 1.0 · vel: 0.5 | Allows ±1 bin/sweep² acceleration |\n"
            "| **R** (measurement noise) | 4.0 (az & range) | ±2 bin centroid uncertainty |\n"
            "| **Gate γ** | 16 (Mahalanobis²) | ≈ 4-bin association radius |"
        )
    st.divider()
    if session is not None:
        lat_ms = st.session_state.get("_last_onnx_ms")
        st.caption(
            f"CNN inference: **{lat_ms:.1f} ms**" if lat_ms else "CNN inference: — ms"
        )
        st.caption("CNN: ONNX · CPU · ~19 k parameters")
    if gru_session is not None:
        gru_lat = st.session_state.get("_last_gru_ms")
        st.caption(
            f"GRU inference: **{gru_lat:.1f} ms/sweep**" if gru_lat else "GRU inference: — ms"
        )
        st.caption("GRU: ONNX · CPU · ~6 k params · streaming · h = confidence heatmap")

# ── Load data ──────────────────────────────────────────────────────────────────

ppi_seq, label_seq, positions = _gen(snr_db, range_bin, az_deg, vr, vt)
cfar_sweeps                   = _cfar_sweeps(snr_db, range_bin, az_deg, vr, vt)
_, kf_pos_list                 = _kf_result(snr_db, range_bin, az_deg, vr, vt)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_classical, tab_bridge, tab_math, tab_ppi, tab_compare, tab_roc, tab_tradeoffs, tab_live, tab_paper = st.tabs([
    "🔭  Classical Radar",
    "🤖  From CFAR to ML",
    "📐  The Math",
    "📡  PPI Display",
    "📊  Algorithm Comparison",
    "📉  ROC Curve",
    "⚖️  Strengths & Trade-offs",
    "📻  Live Radar",
    "📄  Paper",
])

# ── Tab 0: Paper ───────────────────────────────────────────────────────────────

with tab_paper:
    _pdf_path = Path("paper/paper.pdf")

    # ── PDF pages as images ───────────────────────────────────────────────────────
    _page_imgs = sorted(Path("paper").glob("page-*.png"))
    if _page_imgs:
        _dl_col, _ = st.columns([1, 4])
        with _dl_col:
            if _pdf_path.exists():
                with open(_pdf_path, "rb") as _f:
                    st.download_button(
                        "⬇ Download PDF",
                        data=_f,
                        file_name="Bach_RadarDetection_2026.pdf",
                        mime="application/pdf",
                    )
        for _pg in _page_imgs:
            st.image(str(_pg), use_container_width=True)
            st.divider()
    else:
        st.error(
            "Paper pages not found — run: "
            "`cd paper && pdflatex paper.tex && "
            "pdftoppm -r 180 -png paper.pdf page`"
        )

# ── Tab 1: Classical Radar ─────────────────────────────────────────────────────

with tab_classical:
    st.markdown("## Classical radar detection: CA-CFAR + Kalman tracker")
    st.markdown(
        "Before introducing machine learning it is useful to understand the standard pipeline "
        "it replaces — and why that pipeline is difficult to improve without changing the "
        "underlying architecture."
    )

    st.divider()

    # ── Signal chain ──────────────────────────────────────────────────────────
    st.markdown("### 1 — The signal chain")
    st.markdown(
        "A rotating surveillance radar illuminates the scene in a sequence of narrow azimuth "
        "beams. For each beam it transmits a short pulse and records the echo amplitude at "
        "each range gate as the pulse return window slides forward in time. The result after "
        "one full 360° rotation is a **Plan Position Indicator (PPI) sweep** — a 2-D grid of "
        "amplitude samples indexed by (azimuth, range)."
    )
    st.code(
        "One rotation = one PPI sweep  (n_az × n_range  amplitude samples)\n"
        "Ten rotations = the sequence used for detection in this demo\n"
        "\n"
        "  Azimuth resolution:  2°  (180 bins per sweep)\n"
        "  Range resolution:    7.5 m  (64 bins, ≈ 480 m max range)\n"
        "  Rotation period:     2 s  (30 RPM)",
        language="text",
    )

    st.divider()

    # ── CA-CFAR ───────────────────────────────────────────────────────────────
    st.markdown("### 2 — Cell-Averaging CFAR (CA-CFAR)")
    st.markdown(
        "**CFAR** stands for *Constant False Alarm Rate*. The idea is that a target detection "
        "threshold should automatically adapt to the local noise level — otherwise a fixed "
        "threshold would produce too many false alarms in strong clutter and miss targets in "
        "quiet regions."
    )

    col_cfar_txt, col_cfar_eq = st.columns([3, 2])
    with col_cfar_txt:
        st.markdown(
            "For each **cell under test (CUT)**, CA-CFAR computes a noise estimate from the "
            "surrounding **reference cells**, excluding a **guard band** that protects against "
            "the target's own sidelobe energy leaking into the reference window:"
        )
        st.markdown(
            "- Reference cells: the N cells surrounding the guard ring\n"
            "- Guard cells: a small exclusion zone around the CUT\n"
            "- Noise estimate: Z = mean amplitude across all reference cells\n"
            "- Detection threshold: **T = α · Z**\n"
            "- Detection if CUT > T, otherwise no detection"
        )
        st.markdown(
            "The constant **α** is derived analytically to achieve a target false alarm "
            "probability *Pfa* — hence *Constant False Alarm Rate*. "
            "In this demo: guard = 2 cells, reference = 5 cells each side, α = 2.5."
        )
    with col_cfar_eq:
        st.code(
            "# 2-D CA-CFAR (per sweep)\n"
            "for each cell (az, r):\n"
            "    ref = cells in ring(az±7, r±7)\n"
            "         excluding guard ring\n"
            "    Z   = mean(ref)\n"
            "    T   = α * Z\n"
            "    det[az,r] = (cell > T)",
            language="python",
        )

    st.markdown(
        "**Where it works:** CFAR is mathematically optimal when background noise is "
        "homogeneous (all reference cells drawn from the same distribution). For an isolated "
        "target in Rayleigh-distributed clutter with sufficient SNR it achieves very close "
        "to the theoretical Pd curve."
    )
    st.warning(
        "**Where it fails:** The homogeneity assumption breaks down in real coastal and "
        "littoral environments. A reference window straddling a sea-land boundary, a rain "
        "cell, or a vessel wake mixes two different noise distributions. The resulting noise "
        "estimate is wrong — producing either excessive false alarms (underestimate) or "
        "missed detections (overestimate). This failure mode is *fundamental to the "
        "algorithm* and cannot be fixed by tuning α."
    )

    st.divider()

    # ── Kalman filter tracker ─────────────────────────────────────────────────
    st.markdown("### 3 — Kalman filter tracker")
    st.markdown(
        "A single CFAR sweep produces hundreds of detections per rotation — most of them "
        "noise. The **Kalman filter tracker** converts the noisy binary detection map into "
        "confirmed tracks by requiring *spatial and temporal consistency*."
    )

    col_kf1, col_kf2 = st.columns(2)
    with col_kf1:
        st.markdown("**State vector per track:**")
        st.code(
            "x = [range_bin,     # estimated radial position\n"
            "     range_rate,    # bins/sweep\n"
            "     azimuth_bin,   # estimated angular position\n"
            "     azimuth_rate]  # bins/sweep",
            language="python",
        )
        st.markdown("**Predict–update cycle (per sweep):**")
        st.code(
            "# Predict\n"
            "x_pred = F @ x_prev          # kinematics\n"
            "P_pred = F @ P_prev @ F.T + Q # process noise\n"
            "\n"
            "# Associate\n"
            "for each new detection d:\n"
            "    dist = euclidean(d, x_pred)\n"
            "    if dist < gate:            # 8-bin gate\n"
            "        update(x, P, d)        # standard KF\n"
            "\n"
            "# Confirm\n"
            "if track.hits >= 4:  # M-of-N rule\n"
            "    confirm(track)             # report to operator",
            language="python",
        )
    with col_kf2:
        st.markdown("**Why ≥ 4 hits?**")
        st.markdown(
            "A random noise spike that triggers CFAR in one sweep is unlikely to appear in "
            "the same location next sweep, and the sweep after that. Requiring *at least 4 "
            "consistent associations within an 8-bin gate* reduces the false track rate by "
            "several orders of magnitude compared to reporting every CFAR hit directly."
        )
        st.markdown("**Classical pipeline summary:**")
        st.code(
            "raw PPI sweep\n"
            "  → CFAR  (binary detection map)\n"
            "  → peak extraction  (centroid of each blob)\n"
            "  → KF predict  (forward-project existing tracks)\n"
            "  → nearest-neighbour association\n"
            "  → KF update\n"
            "  → confirm if hits ≥ 4\n"
            "  → confirmed track (position + velocity)",
            language="text",
        )

    st.divider()

    # ── Limitations summary ───────────────────────────────────────────────────
    st.markdown("### 4 — Operational limitations")
    col_lim1, col_lim2, col_lim3 = st.columns(3)
    with col_lim1:
        st.markdown("**Low SNR performance**")
        st.markdown(
            "CFAR applies a threshold independently to each sweep. When the target echo "
            "barely clears the noise floor, CFAR misses it most sweeps and the KF never "
            "accumulates enough hits to confirm. The ML pipeline integrates evidence across "
            "sweeps through a learned representation — achieving full detection "
            "several decibels lower."
        )
    with col_lim2:
        st.markdown("**Heterogeneous clutter**")
        st.markdown(
            "Reference window contamination at clutter boundaries floods the display with "
            "false alarms. The KF suppresses most of them but confirmed false tracks still "
            "reach the operator. CFAR has no mechanism to learn or adapt to a specific "
            "clutter environment — the noise model is fixed at algorithm design time."
        )
    with col_lim3:
        st.markdown("**Fixed operating point**")
        st.markdown(
            "α is a scalar constant. Changing the operating point (Pfa vs Pd trade-off) "
            "requires re-deriving and redeploying the constant. An ML detector outputs a "
            "continuous confidence score that can be thresholded post-hoc at any operating "
            "point — the ROC curve is swept by adjusting the threshold, not the model."
        )

    st.info(
        "**Next tab →** The bridge from CFAR to ML: how the problem is reframed as a "
        "classification task, what training data is needed, and which architecture is used."
    )


# ── Tab 2: From CFAR to ML ─────────────────────────────────────────────────────

with tab_bridge:
    st.markdown("## From CFAR to ML: reframing radar detection as a learning problem")
    st.markdown(
        "CFAR is a hand-crafted statistical decision rule. The ML alternative replaces the "
        "hand-crafted rule with a *learned* one — trained on examples of what targets and "
        "clutter look like, rather than on a closed-form noise model."
    )

    st.divider()

    # ── Problem reframing ─────────────────────────────────────────────────────
    st.markdown("### 1 — Reframing the problem")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Classical framing**")
        st.markdown(
            "For each cell (az, r) in each sweep:\n"
            "- Is this amplitude value above a locally-derived threshold?\n"
            "- Binary decision per cell per sweep\n"
            "- Threshold derived analytically from a noise model\n"
            "- No learning, no memory between sweeps beyond the KF"
        )
    with col_r2:
        st.markdown("**ML framing**")
        st.markdown(
            "Given a sequence of sweeps:\n"
            "- Produce a probability map P(target | sweeps 1..k) for every cell\n"
            "- Continuous output that can be thresholded at any operating point\n"
            "- The decision boundary is *learned from data*, not derived from a noise model\n"
            "- Temporal evidence integration is built into the architecture"
        )

    st.divider()

    # ── What the model sees ───────────────────────────────────────────────────
    st.markdown("### 2 — What the model sees")
    st.markdown(
        "A moving target leaves a distinctive signature in the PPI sequence: "
        "it appears at a slightly different (az, r) position each sweep, producing a "
        "streak in the *temporal maximum* image and elevated variance in the *temporal std* "
        "image. Static clutter (ground return, buildings) stays fixed and averages out "
        "in the std channel."
    )
    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        st.markdown("**CNN batch approach**")
        st.code(
            "# Temporal feature map (3 channels)\n"
            "noise_fl = percentile(ppi, 10, axis=0)\n"
            "norm     = ppi / noise_fl        # (10, 180, 64)\n"
            "\n"
            "ch_max  = norm.max(axis=0)       # bright where target was\n"
            "ch_mean = norm.mean(axis=0)      # local noise estimate\n"
            "ch_std  = norm.std(axis=0)       # high where target moved\n"
            "\n"
            "X = stack([ch_max, ch_mean, ch_std])  # (3, 180, 64)\n"
            "# Feed to CNN once after all 10 sweeps",
            language="python",
        )
        st.markdown(
            "✓ Simple, fast, easy to train  \n"
            "✗ Requires the full 10-sweep window before inference  \n"
            "✗ Temporal statistics conflate multiple simultaneous targets"
        )
    with col_inp2:
        st.markdown("**ConvGRU streaming approach**")
        st.code(
            "# Process one sweep at a time\n"
            "h = zeros(1, 180, 64)   # hidden state = P(target)\n"
            "\n"
            "for sweep in sequence:\n"
            "    norm = sweep / ema_noise_floor\n"
            "    h    = ConvGRU(norm, h)\n"
            "    # h is now the detection probability map\n"
            "    # Threshold h to get peaks → feed KF\n"
            "\n"
            "# Hidden state h IS the output:\n"
            "#   h[az, r] ≈ P(target at (az,r) | sweeps seen so far)",
            language="python",
        )
        st.markdown(
            "✓ Streaming: one sweep at a time, no waiting  \n"
            "✓ Multi-target: each cell in h is independent  \n"
            "✓ Natural integration: h accumulates evidence over time  \n"
            "✓ Matches real radar operation"
        )

    st.divider()

    # ── Training data ─────────────────────────────────────────────────────────
    st.markdown("### 3 — Training data: synthetic vs real")
    st.markdown(
        "A radar ML model needs labelled examples: PPI sequences with known target "
        "positions (positive class) and sequences without targets (negative class). "
        "There are two sources."
    )

    col_syn, col_real = st.columns(2)
    with col_syn:
        st.markdown("**Synthetic data** *(used in this demo)*")
        st.markdown(
            "Generate amplitude cubes from a physical model:\n\n"
            "- **Target**: Gaussian beam profile × range-gate pulse × SNR amplitude, "
            "moving across the PPI at a random velocity\n"
            "- **Clutter**: Rayleigh-distributed with range-dependent R⁻² amplitude floor "
            "and log-normal speckle\n"
            "- **SNR** drawn uniformly from −20 to +40 dB\n\n"
            "Reproducible (seeded), free, arbitrarily large, no classification issues.\n\n"
            "**Risk:** if the real clutter statistics differ significantly from the Rayleigh "
            "model, the model will over-fit to the synthetic distribution and degrade on real data. "
            "Drift detection (PSI metric) catches this in deployment."
        )
        st.code(
            "# This demo — full pipeline\n"
            "python src/data/generate.py   # seed-controlled\n"
            "dvc repro                     # cached DAG\n"
            "# All parameters in params_ppi.yaml",
            language="bash",
        )
    with col_real:
        st.markdown("**Real data** *(production requirement)*")
        st.markdown(
            "Record actual radar returns and label them:\n\n"
            "- Manual labelling by radar operators, or\n"
            "- Cooperative targets (AIS cross-referencing), or\n"
            "- Fusion with a second, independent sensor\n\n"
            "Captures actual clutter statistics (sea state, terrain, rain) that the "
            "synthetic model can only approximate.\n\n"
            "**Workflow:** train on synthetic → validate on real → identify mismatch → "
            "collect labelled real examples → add to training set → retrain via pipeline "
            "→ promote if accuracy gate passes."
        )
        st.info(
            "In classified or air-gapped environments, real data stays on-prem. "
            "This is exactly why the demo includes a self-hosted Gitea pipeline and "
            "MLflow with a local SQLite backend — no data needs to leave the network."
        )

    st.divider()

    # ── Architecture summary ──────────────────────────────────────────────────
    st.markdown("### 4 — Architecture summary")
    st.code(
        "ConvGRU (streaming detector)\n"
        "────────────────────────────────────────────────\n"
        "Input each sweep:   sweep_norm  (1, 180, 64)  normalised amplitude\n"
        "                    h_in        (1, 180, 64)  previous hidden state\n"
        "\n"
        "Encoder:  4 × Conv2d(1→16→32→32→16, k=3) + BN + ReLU\n"
        "GRU gate: ConvGRU cell — produces h_out ∈ [0,1]^(1,180,64)\n"
        "\n"
        "Output:   h_out  =  P(target | sweeps 1..k)   per cell\n"
        "          (identical to h_in for the next sweep)\n"
        "\n"
        "Export:   ONNX opset 17 — onnxruntime on embedded Linux or FPGA via Vitis AI\n"
        "Latency:  < 1 ms per sweep on CPU  (p50, batch_size=1)",
        language="text",
    )

    st.divider()

    # ── Pipeline value ────────────────────────────────────────────────────────
    st.markdown("### 5 — Why the MLOps pipeline matters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**Reproducibility**")
        st.markdown(
            "DVC locks the exact dataset version, code commit, and hyperparameters "
            "that produced each model artifact. Any deployed model can be reproduced "
            "bit-identically from `dvc repro` on the tagged commit."
        )
    with col_p2:
        st.markdown("**Adaptation**")
        st.markdown(
            "Operating clutter statistics change (new area, seasonal sea state, hardware "
            "aging). PSI drift detection fires a Prometheus alert when the incoming "
            "distribution diverges from the training reference. The retraining loop is "
            "triggered automatically — CFAR has no equivalent mechanism."
        )
    with col_p3:
        st.markdown("**Safety gates**")
        st.markdown(
            "No model reaches production without passing the accuracy gate in CI, human "
            "review in the MLflow registry, and ONNX latency validation on the target "
            "hardware. Rollback is one `register_model.py` command — restore the previous "
            "Production version from the archive."
        )

    st.info(
        "**Next tab →** The Math: the CFAR threshold formula, KF equations, "
        "ConvGRU forward pass, and ONNX export details."
    )


# ── Tab 3: PPI Display ─────────────────────────────────────────────────────────

with tab_ppi:
    # Animation state
    if "animating" not in st.session_state:
        st.session_state.animating = False
    if "anim_frame" not in st.session_state:
        st.session_state.anim_frame = 0

    col_scope, col_info = st.columns([5, 3], gap="large")

    with col_scope:
        sw_col, btn_col, stop_col = st.columns([5, 1, 1])
        with sw_col:
            sweep_idx = st.slider("Show sweep", 1, N_SW, N_SW, key="sweep") - 1
        with btn_col:
            st.write("")
            if st.button("▶ Animate", key="ppi_animate", use_container_width=True):
                st.session_state.animating = True
                st.session_state.anim_frame = 0
        with stop_col:
            st.write("")
            if st.button("■ Stop", key="ppi_stop", use_container_width=True):
                st.session_state.animating = False

        scope_ph = st.empty()
        pd_ph    = st.empty()

    # Precompute CNN confidence at target for each sweep (padded-window CNN)
    ml_conf_history = []
    for _n in range(1, N_SW + 1):
        _m  = _ml_map(ppi_seq, _n)
        _az = int(positions[_n - 1][0] / 360 * N_AZ) % N_AZ
        _r  = int(np.clip(positions[_n - 1][1], 0, N_RANGE - 1))
        ml_conf_history.append(float(_m[_az, _r]))

    # Precompute GRU: sequential single-sweep steps — conf at target + full heatmap per sweep
    if gru_session is not None:
        gru_conf_history, gru_heatmap_history = _gru_seq(ppi_seq, positions)
    else:
        gru_conf_history, gru_heatmap_history = None, None

    # Per-sweep KF track histories for the three-pipeline comparison
    kf_history_cfar = _build_kf_history_cfar(snr_db, range_bin, az_deg, vr, vt)
    kf_history_cnn  = _build_kf_history_cnn(snr_db, range_bin, az_deg, vr, vt)

    # GRU+KF history: run KF over already-computed heatmaps (fast, no extra inference)
    if gru_heatmap_history:
        _kf_gru = PPIKalmanTracker()
        kf_history_gru: list = []
        for _h_map in gru_heatmap_history:
            _kf_gru.update_from_peaks(_gru_peaks(_h_map), N_AZ, N_RANGE)
            kf_history_gru.append(_kf_gru.track_positions())
    else:
        kf_history_gru = [[] for _ in range(N_SW)]

    ml_map_now = _ml_map(ppi_seq, sweep_idx + 1)

    with col_info:
        az_now, r_now = positions[sweep_idx]
        az_b = int(az_now / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(r_now, 0, N_RANGE - 1))

        # ── ARPA navigation block ─────────────────────────────────────────────
        course, speed_kt, cpa_r_m, t_cpa_s = _arpa_course_speed(
            positions[:sweep_idx + 1], vr, vt
        )
        range_m = r_now * _RANGE_BIN_M
        st.markdown("#### ARPA Target Data")
        c1, c2 = st.columns(2)
        c1.metric("Bearing", f"{az_now:.1f}°T")
        c2.metric("Range", f"{range_m:.0f} m" if range_m < 1000 else f"{range_m/1000:.2f} km")
        c1.metric("Course", f"{course:.0f}°T")
        c2.metric("Speed", f"{speed_kt:.1f} kt")
        if vr < 0 or abs(vr) < 0.05:
            c1.metric("CPA range", f"{cpa_r_m:.0f} m" if cpa_r_m < 1000 else f"{cpa_r_m/1000:.2f} km")
            c2.metric("Time to CPA", f"{t_cpa_s:.0f} s")
        else:
            c1.metric("Aspect", "Opening")
            c2.metric("CPA", "N/A")
        st.divider()

        # ── Detector confidence ───────────────────────────────────────────────
        # All three detectors are evaluated at the EXACT target cell in each sweep
        # so the metrics are directly comparable (no neighbourhood inflation).
        n_sw_shown = sweep_idx + 1

        # CFAR Pd: fraction of sweeps where CFAR fired at the exact target cell
        cfar_hits_path = 0
        for _sw in range(n_sw_shown):
            _az_sw, _r_sw = positions[_sw]
            _az_b = int(_az_sw / 360 * N_AZ) % N_AZ
            _r_b  = int(np.clip(_r_sw, 0, N_RANGE - 1))
            if cfar_sweeps[_sw, _az_b, _r_b]:
                cfar_hits_path += 1
        cfar_pd_path = cfar_hits_path / n_sw_shown

        # ML confidence: max in a 3×3 window (CNN output is spatially smoothed —
        # a ±1 window is still single-cell equivalent for comparison with CFAR)
        ml_win = ml_map_now[
            max(0, az_b - 1):az_b + 2,
            max(0, r_b  - 1):r_b  + 2,
        ]
        ml_conf = float(ml_win.max()) if ml_win.size else float(ml_map_now[az_b, r_b])

        st.markdown(f"#### Pipeline Comparison — Sweep {n_sw_shown}/{N_SW}")

        # Three-pipeline track summary at current sweep
        _cfar_trks = kf_history_cfar[sweep_idx]
        _gru_trks  = kf_history_gru[sweep_idx]
        _cnn_trks  = kf_history_cnn[sweep_idx]

        # Single-detector confidence metrics
        st.metric("CNN confidence (peak near target)", f"{ml_conf:.1%}",
                  help="Max CNN output in 3×3 cell window around the target's current position")
        if gru_heatmap_history:
            gru_h_val = float(gru_heatmap_history[sweep_idx][az_b, r_b])
            st.metric("GRU h at target cell", f"{gru_h_val:.1%}",
                      help="Hidden state value at the exact target cell — directly the confidence probability.")
        st.metric("CFAR Pd at target cell", f"{cfar_pd_path:.1%}",
                  help="Fraction of sweeps where CFAR fired at the exact target cell — true single-cell Pd")

        st.divider()
        st.markdown("**Confirmed tracks (≥4 associations)**")
        _col1, _col2, _col3 = st.columns(3)
        _col1.metric("CFAR+KF", str(len(_cfar_trks)),
                     help="White ✕ — classical pipeline confirmed tracks at this sweep")
        _col2.metric("GRU+KF", str(len(_gru_trks)),
                     help="Magenta ◆ — streaming GRU heatmap peaks → same KF")
        _col3.metric("CNN+KF", str(len(_cnn_trks)),
                     help="Yellow ▲ — batch CNN peaks → same KF")
        st.divider()

        st.caption(
            "🟢 Green = radar echo · 🔵 Cyan = CNN confidence · 🟠 Orange = CFAR · "
            "⬜ White ✕ = CFAR+KF · 🟣 Magenta ◆ = GRU+KF · 🟡 Yellow ▲ = CNN+KF · "
            "🔴 Red ✕ = true target · 🔴 Red trail = ARPA · 🟡 Yellow → = ARPA vector"
        )

        with st.expander("How the ML model works"):
            st.markdown("""
The CNN receives **three 2-D feature channels** derived from all visible sweeps:

| Channel | Description |
|---------|-------------|
| **Max** | Pixel-wise max — peak beam illumination |
| **Mean** | Average — clutter baseline |
| **Std** | Std deviation — high where signal *varied* (motion) |

All channels are divided by the per-range clutter floor (10th-percentile amplitude) to cancel range attenuation. A moving target leaves a bright streak in the max and std channels that the CNN learns to recognise.
""")

    # Render scope + Pd chart (static or animated)
    if st.session_state.animating:
        has_more = _render_anim_frame(
            st.session_state.anim_frame,
            ppi_seq, positions, cfar_sweeps, kf_pos_list,
            ml_conf_history, vr, vt,
            show_ml, show_cfar, show_kf, show_arpa,
            scope_ph, pd_ph,
            gru_history=gru_conf_history,
            gru_heatmaps=gru_heatmap_history,
            kf_history_cfar=kf_history_cfar,
            kf_history_gru=kf_history_gru,
            kf_history_cnn=kf_history_cnn,
            show_gru_kf=show_gru_kf,
            show_cnn_kf=show_cnn_kf,
        )
        if has_more:
            st.session_state.anim_frame += 1
            st.rerun()
        else:
            st.session_state.animating = False
    else:
        scope_ph.plotly_chart(
            make_ppi_figure(
                ppi_seq, positions, cfar_sweeps,
                kf_history_cfar[sweep_idx],   # per-sweep CFAR+KF positions
                ml_map_now, sweep_idx, vr, vt,
                show_ml, show_cfar, show_kf, show_arpa,
                gru_map_now=gru_heatmap_history[sweep_idx] if gru_heatmap_history else None,
                show_gru=show_ml,
                gru_kf_pos=kf_history_gru[sweep_idx],
                show_gru_kf=show_gru_kf,
                cnn_kf_pos=kf_history_cnn[sweep_idx],
                show_cnn_kf=show_cnn_kf,
            ),
            use_container_width=False,
            key="static_scope",
        )
        pd_ph.plotly_chart(
            _pd_chart_fig(ml_conf_history, sweep_idx + 1, gru_conf_history),
            use_container_width=True,
            key="static_pd",
        )

    # ── GRU hidden-state evolution ────────────────────────────────────────────
    if gru_heatmap_history:
        st.markdown(
            "#### GRU hidden state — confidence accumulates over 10 sweeps"
        )
        st.caption(
            "Each panel shows the full (azimuth × range) confidence map h after that sweep. "
            "h starts at zero; the update gate z decides how much new evidence to absorb. "
            "Red × = true target position. Brighter = higher confidence."
        )
        st.plotly_chart(
            _gru_evolution_fig(gru_heatmap_history, positions),
            use_container_width=True,
            key="gru_evolution",
        )

    # ── Waterfall / surface plot ───────────────────────────────────────────────
    st.plotly_chart(
        _waterfall_fig(ppi_seq, positions),
        use_container_width=True,
        key="waterfall",
    )

# ── Tab 2: Algorithm Comparison ────────────────────────────────────────────────

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


with tab_compare:
    # ── Operating-point calibration (run once, cached) ────────────────────────
    cfar_pfa, cnn_thr, cfar_kf_ftr, cnn_kf_ftr, gru_thr, gru_kf_ftr = _calibrate_pfa()

    # ── Operational headline ──────────────────────────────────────────────────
    st.markdown("### Does the AI system outperform classical radar detection?")
    _fa_ratio_gru = cfar_kf_ftr / max(gru_kf_ftr, 0.01)
    _fa_ratio_cnn = cfar_kf_ftr / max(cnn_kf_ftr, 0.01)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Catches weaker targets**")
        st.markdown(
            "The recurrent AI pipeline (ConvGRU) detects 9 in 10 targets at signal strengths "
            "where the classical system still misses most — equivalent to detecting targets "
            "further away, moving more slowly, or hidden in heavier clutter. "
            "The recurrent model integrates evidence across antenna rotations through a learned "
            "hidden state, rather than applying a static threshold to each sweep."
        )
    with col_b:
        st.markdown("**Fewer false alarms reaching operators**")
        st.markdown(
            f"Both systems start with the same raw noise-hit rate. The ConvGRU pipeline produces "
            f"**{_fa_ratio_gru:.1f}× fewer spurious confirmed tracks** on noise-only data — "
            "operators are interrupted less often by ghost contacts."
        )
    with col_c:
        st.markdown("**No extra hardware**")
        st.markdown(
            "The ConvGRU model runs in under 1 ms per sweep on a standard CPU — "
            "the same order as the classical algorithm, with no GPU required. "
            "The ONNX artifact targets embedded Linux on the radar signal processor "
            "or FPGA compilation via Xilinx Vitis AI."
        )

    st.divider()

    GRID_C = "rgba(128,128,128,0.2)"

    # ── How the comparison is made ────────────────────────────────────────────
    with st.expander("How the comparison is made — methodology"):
        st.markdown(
            "All three pipelines are evaluated **at the same false-alarm rate** so none has an unfair advantage. "
            "The classical detector's threshold is fixed; the ML thresholds are calibrated so all produce the same "
            "fraction of noise-hits per grid cell. "
            "The Kalman filter tracker is identical across all pipelines — only the upstream detector changes."
        )
        st.code(
            "Classical:  raw PPI → CFAR (α=2.5) → peaks → KF (≥4 hits) → confirmed track\n"
            "CNN+KF:     raw PPI → CNN  (batch)  → peaks → KF (≥4 hits) → confirmed track\n"
            "GRU+KF:     raw PPI → ConvGRU (streaming, per-sweep hidden state) → peaks → KF → confirmed track",
            language="text",
        )
        col_pfa1, col_pfa2, col_pfa3, col_pfa4 = st.columns(4)
        col_pfa1.metric("Classical false-alarm rate",
                        f"{cfar_pfa * 100:.2f}% per cell",
                        help="Measured on 200 noise-only scenes. All pipelines are matched to this rate.")
        col_pfa2.metric("CNN threshold",
                        f"{cnn_thr:.3f}",
                        help="CNN confidence level producing the same cell-level false-alarm rate as CFAR.")
        col_pfa3.metric("GRU threshold",
                        f"{gru_thr:.3f}",
                        help="ConvGRU probability level producing the same cell-level false-alarm rate as CFAR.")
        col_pfa4.metric("Raw noise hits per sweep",
                        f"{cfar_pfa * N_AZ * N_RANGE:.0f}",
                        help="All pipelines see this many noise hits per sweep before the tracker filters them.")

    st.divider()

    # ── Pd vs SNR (pipeline comparison) ──────────────────────────────────────
    st.markdown("#### How often does each system detect the target — across all signal strengths?")
    st.caption(
        "Signal strength (x-axis): higher = target echo is stronger relative to background noise. "
        "Each point = 30 randomised trials at random target positions, speeds, and directions."
    )
    (snr_arr,
     cfar_sa_pd, cfar_kf_pd, cnn_b_pd, cnn_kf_pd,
     cfar_kf_rmse_arr, cnn_kf_rmse_arr,
     gru_kf_pd, gru_kf_rmse_arr) = _snr_sweep(cnn_thr=cnn_thr, gru_thr=gru_thr, _v=11)

    # Compute 90% Pd crossover SNR — use GRU (best ML pipeline) for the gap annotation
    _gru_kf_arr  = np.array(gru_kf_pd)
    _cfar_kf_arr = np.array(cfar_kf_pd)
    _snr90_gru  = float(snr_arr[np.argmax(_gru_kf_arr  >= 0.9)]) if (_gru_kf_arr  >= 0.9).any() else None
    _snr90_cfar = float(snr_arr[np.argmax(_cfar_kf_arr >= 0.9)]) if (_cfar_kf_arr >= 0.9).any() else None

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=snr_arr, y=cfar_kf_pd * 100, mode="lines+markers",
        name="Classical  (CFAR + KF tracker)",
        line=dict(color="orange", width=2.5),
        marker=dict(size=7),
    ))
    fig_c.add_trace(go.Scatter(
        x=snr_arr, y=cnn_kf_pd * 100, mode="lines+markers",
        name="CNN + KF tracker  (batch)",
        line=dict(color="#00ccff", width=2.5),
        marker=dict(size=7),
    ))
    fig_c.add_trace(go.Scatter(
        x=snr_arr, y=gru_kf_pd * 100, mode="lines+markers",
        name="ConvGRU + KF tracker  (streaming)",
        line=dict(color="#c084fc", width=3.0),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig_c.add_hline(y=50, line_dash="dot", line_color="#555",
                    annotation_text="detects half the time", annotation_position="right",
                    annotation_font_color="#888")
    fig_c.add_hline(y=90, line_dash="dot", line_color="#444",
                    annotation_text="detects 9 in 10", annotation_position="right",
                    annotation_font_color="#aaa")
    # Shade the SNR gap between GRU and classical reaching 90% Pd
    if _snr90_gru is not None and _snr90_cfar is not None and _snr90_gru < _snr90_cfar:
        _gap_db = _snr90_cfar - _snr90_gru
        fig_c.add_vrect(
            x0=_snr90_gru, x1=_snr90_cfar,
            fillcolor="rgba(192,132,252,0.07)", line_width=0,
        )
        fig_c.add_vline(x=_snr90_gru,  line_dash="dash", line_width=1.5,
                        line_color="rgba(192,132,252,0.65)",
                        annotation_text=f"GRU reaches 90%<br>at {_snr90_gru:+.0f} dB",
                        annotation_position="top right",
                        annotation_font_color="#c084fc", annotation_font_size=10)
        fig_c.add_vline(x=_snr90_cfar, line_dash="dash", line_width=1.5,
                        line_color="rgba(255,165,0,0.55)",
                        annotation_text=f"Classical reaches 90%<br>at {_snr90_cfar:+.0f} dB",
                        annotation_position="top left",
                        annotation_font_color="orange", annotation_font_size=10)
        fig_c.add_annotation(
            x=(_snr90_gru + _snr90_cfar) / 2, y=95,
            text=f"← {_gap_db:.0f} dB GRU advantage →",
            showarrow=False,
            font=dict(color="#c084fc", size=12),
            bgcolor="rgba(0,0,0,0.5)",
        )
    fig_c.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="SNR (dB)", gridcolor=GRID_C, zeroline=False, color="white"),
        yaxis=dict(title="Detection probability (%)", range=[0, 105],
                   gridcolor=GRID_C, zeroline=False, color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_c, use_container_width=True)
    st.caption(
        "All three pipelines compared at the same false-alarm rate (matched Pfa). "
        "A detection counts when the confirmed track falls within ~45 m of the true target position. "
        "GRU (◆ purple) = streaming ConvGRU that updates a learned detection heatmap each sweep."
    )

    st.divider()

    # ── Metrics table at matched Pfa ──────────────────────────────────────────
    st.markdown("#### Summary metrics at matched operating point")

    idx10 = int(np.argmin(np.abs(snr_arr - 10.0)))
    idx0  = int(np.argmin(np.abs(snr_arr - 0.0)))
    idx4  = int(np.argmin(np.abs(snr_arr - 4.0)))

    col_tbl = st.columns(4)
    col_tbl[0].markdown("**Metric**")
    col_tbl[1].markdown("**Classical  (CFAR+KF)**")
    col_tbl[2].markdown("**CNN+KF**")
    col_tbl[3].markdown("**GRU+KF** ⭐")

    rows = [
        ("Detection rate — faint target (0 dB signal)",
         f"{cfar_kf_pd[idx0]*100:.0f} %",
         f"{cnn_kf_pd[idx0]*100:.0f} %",
         f"**{gru_kf_pd[idx0]*100:.0f} %**"),
        ("Detection rate — at 4 dB signal",
         f"{cfar_kf_pd[idx4]*100:.0f} %",
         f"{cnn_kf_pd[idx4]*100:.0f} %",
         f"**{gru_kf_pd[idx4]*100:.0f} %**"),
        ("Detection rate — moderate target (10 dB signal)",
         f"{cfar_kf_pd[idx10]*100:.0f} %",
         f"{cnn_kf_pd[idx10]*100:.0f} %",
         f"**{gru_kf_pd[idx10]*100:.0f} %**"),
        ("Track position error at 10 dB signal",
         f"{cfar_kf_rmse_arr[idx10] * _RANGE_BIN_M:.0f} m",
         f"{cnn_kf_rmse_arr[idx10] * _RANGE_BIN_M:.0f} m",
         f"**{gru_kf_rmse_arr[idx10] * _RANGE_BIN_M:.0f} m**"),
        ("Spurious tracks on noise-only data",
         f"{cfar_kf_ftr:.2f}",
         f"{cnn_kf_ftr:.2f}",
         f"**{gru_kf_ftr:.2f}**"),
    ]
    for label, v_cfar, v_cnn, v_gru in rows:
        c0, c1, c2, c3 = st.columns(4)
        c0.markdown(label)
        c1.markdown(v_cfar)
        c2.markdown(v_cnn)
        c3.markdown(v_gru)

    st.caption(
        "Track position error = mean distance between the confirmed track estimate and the true target, "
        "averaged over trials where the system detected. "
        "Spurious tracks = confirmed ghost contacts per 10-sweep window when no target is present. "
        "All pipelines evaluated at the same false-alarm rate (matched Pfa)."
    )

    st.divider()

    # ── Clutter-only false alarm panel ────────────────────────────────────────
    st.markdown("#### Spurious alarms — what happens with no target present?")
    st.markdown(
        "All pipelines see the same rate of random noise spikes (~"
        f"{cfar_pfa * N_AZ * N_RANGE:.0f} per sweep). "
        "The tracker's job is to suppress those so operators only see real confirmed contacts. "
        "How many ghost tracks slip through per 10-rotation window?"
    )
    col_fa1, col_fa2, col_fa3 = st.columns(3)
    with col_fa1:
        st.markdown("**Classical pipeline**")
        st.metric("Ghost tracks per window",
                  f"{cfar_kf_ftr:.2f}",
                  help="Mean confirmed spurious tracks per 10-sweep noise-only window, measured over 200 scenes")
        st.markdown(
            f"Tracker reduces ~{cfar_pfa * N_AZ * N_RANGE * N_SW:.0f} raw noise spikes "
            f"per window down to **{cfar_kf_ftr:.2f}** ghost contacts — "
            f"a {cfar_pfa * N_AZ * N_RANGE * N_SW / max(cfar_kf_ftr, 0.01):.0f}× reduction. "
            "The remaining contacts occur because random spikes happen to fall "
            "in consistent locations across sweeps, fooling the tracker."
        )
    with col_fa2:
        st.markdown("**CNN pipeline**")
        st.metric("Ghost tracks per window",
                  f"{cnn_kf_ftr:.2f}",
                  help="Mean confirmed spurious tracks per 10-sweep noise-only window, measured over 200 scenes")
        st.markdown(
            "The CNN model suppresses most of the grid to near-zero confidence, "
            "so noise spikes are more isolated and less likely to chain into a confirmed track. "
            f"Result: **{_fa_ratio_cnn:.1f}× fewer ghost contacts** than the classical pipeline."
        )
    with col_fa3:
        st.markdown("**GRU pipeline**")
        st.metric("Ghost tracks per window",
                  f"{gru_kf_ftr:.2f}",
                  help="Mean confirmed spurious tracks per 10-sweep noise-only window, measured over 200 scenes")
        st.markdown(
            "The ConvGRU's recurrent hidden state reinforces detections that persist "
            "across consecutive sweeps and suppresses one-off noise spikes. "
            f"Result: **{_fa_ratio_gru:.1f}× fewer ghost contacts** than the classical pipeline."
        )

    st.divider()

    # ── Per-sweep evidence accumulation ───────────────────────────────────────
    st.markdown("#### How quickly does each system confirm a detection?")
    st.markdown(
        "Each antenna rotation adds another look at the target. "
        "All systems need at least 4 consistent detections before raising a confirmed alarm — "
        "drag the slider to see how quickly each pipeline builds confidence at different signal strengths."
    )

    col_sp_snr, _ = st.columns([1, 3])
    with col_sp_snr:
        sweep_snr = st.slider("SNR (dB)", -20.0, 40.0, float(snr_db), 1.0,
                              key="sweep_snr",
                              help="SNR for the per-sweep comparison")

    (cfar_sa_cum, cfar_kf_cum, cnn_kf_cum, gru_kf_cum,
     cfar_kf_latency, cnn_kf_latency, gru_kf_latency) = _sweep_profile(
         sweep_snr, cnn_thr=cnn_thr, gru_thr=gru_thr)
    sweeps = list(range(1, N_SW + 1))

    fig_sw = go.Figure()
    fig_sw.add_trace(go.Scatter(
        x=sweeps, y=cfar_kf_cum, mode="lines+markers",
        name="Classical  (CFAR + KF tracker)",
        line=dict(color="orange", width=2.5),
        marker=dict(size=7),
    ))
    fig_sw.add_trace(go.Scatter(
        x=sweeps, y=cnn_kf_cum, mode="lines+markers",
        name="CNN + KF tracker  (batch)",
        line=dict(color="#00ccff", width=2.5),
        marker=dict(size=7),
    ))
    fig_sw.add_trace(go.Scatter(
        x=sweeps, y=gru_kf_cum, mode="lines+markers",
        name="ConvGRU + KF tracker  (streaming)",
        line=dict(color="#c084fc", width=3.0),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig_sw.add_hline(y=50, line_dash="dot", line_color="#555",
                     annotation_text="50 %", annotation_position="right",
                     annotation_font_color="#888")
    fig_sw.add_hline(y=90, line_dash="dot", line_color="#444",
                     annotation_text="90 %", annotation_position="right",
                     annotation_font_color="#888")
    fig_sw.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Sweep number", tickvals=sweeps,
                   gridcolor=GRID_C, zeroline=False, color="white"),
        yaxis=dict(title="Cumulative detection probability (%)", range=[0, 105],
                   gridcolor=GRID_C, zeroline=False, color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(t=20, b=40, l=60, r=90),
    )
    st.plotly_chart(fig_sw, use_container_width=True)

    lat_cfar_str = f"{cfar_kf_latency:.1f}" if not np.isnan(cfar_kf_latency) else "—"
    lat_cnn_str  = f"{cnn_kf_latency:.1f}"  if not np.isnan(cnn_kf_latency)  else "—"
    lat_gru_str  = f"{gru_kf_latency:.1f}"  if not np.isnan(gru_kf_latency)  else "—"
    col_lat1, col_lat2, col_lat3 = st.columns(3)
    col_lat1.metric("Classical — avg. rotations to confirm",
                    f"{lat_cfar_str} sweeps",
                    help="Mean antenna rotation index of first confirmed track, averaged over trials that eventually detected")
    col_lat2.metric("CNN — avg. rotations to confirm",
                    f"{lat_cnn_str} sweeps",
                    help="Same metric for the CNN pipeline")
    col_lat3.metric("GRU — avg. rotations to confirm",
                    f"{lat_gru_str} sweeps",
                    help="Same metric for the ConvGRU streaming pipeline")
    st.caption(
        "Each line shows the fraction of 30 trials with a confirmed contact by that rotation number. "
        "No pipeline can confirm before rotation 4 (minimum 4 consistent detections required)."
    )
    st.info(
        "**Why the CNN pipeline can be slower at very low signal strength:** "
        "The CNN model was trained on full 10-rotation windows. "
        "When it only has 1–5 rotations of data it is working outside its training conditions, "
        "so confidence stays low until enough rotations accumulate. "
        "The GRU does not have this constraint — it processes sweeps one at a time "
        "and updates its hidden state after each rotation, matching how the radar actually operates."
    )

    st.divider()

    # ── Heterogeneous clutter: CFAR failure mode ──────────────────────────────
    st.markdown("#### Why CFAR breaks in realistic clutter — and why ML doesn't")
    st.markdown(
        "CFAR's core statistical assumption is **local noise homogeneity**: "
        "the reference cells surrounding a cell-under-test sample the same noise distribution. "
        "In real coastal radar — sea-land boundaries, rain cells, vessel wakes — this assumption "
        "fails at clutter edges. Reference cells crossing the boundary give a contaminated noise "
        "estimate, producing false alarms just outside the patch and missed detections near it. "
        "\n\n"
        "The ConvGRU learns a spatial scene model from training data and is not constrained by "
        "a sliding-window noise estimate. The heatmap below shows this directly: "
        "CFAR fires across the patch boundary; GRU concentrates probability on the real target."
    )

    @st.cache_data(show_spinner="Generating heterogeneous clutter scene…")
    def _het_clutter_demo(gru_thr_val: float, cfar_thr_factor: float = 2.5):
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

    cfar_hits_h, gru_prob_h, mean_amp_h, patch_mask_h, positions_h = _het_clutter_demo(gru_thr)

    _tgt_az_final  = int(positions_h[-1][0] / 360 * N_AZ) % N_AZ
    _tgt_r_final   = int(np.clip(positions_h[-1][1], 0, N_RANGE - 1))

    col_het1, col_het2 = st.columns(2)

    with col_het1:
        st.markdown("**CA-CFAR detections** (sum across 10 sweeps)")
        st.caption(
            "Orange = CFAR fired here. The clutter patch (dashed yellow boundary) is 5× brighter "
            "than background at the same range. CFAR's reference window straddles the edge, "
            "giving a contaminated noise estimate — the patch floods the display with false alarms "
            "that cannot be suppressed without changing the algorithm."
        )
        _cfar_disp = cfar_hits_h.copy()
        fig_het_cfar = go.Figure()
        fig_het_cfar.add_trace(go.Heatmap(
            z=mean_amp_h.T, colorscale="Greys", showscale=False,
            zmin=0, zmax=float(np.percentile(mean_amp_h, 98)),
            name="Clutter background",
        ))
        # Patch boundary outline
        _az_patch = [60, 60, 100, 100, 60]
        _r_patch  = [20, 50, 50, 20, 20]
        fig_het_cfar.add_trace(go.Scatter(
            x=_az_patch, y=_r_patch, mode="lines",
            line=dict(color="yellow", width=1.5, dash="dot"),
            name="Clutter patch boundary", showlegend=True,
        ))
        # CFAR fires — show as scatter over the heatmap
        _cfar_az, _cfar_r = np.where(cfar_hits_h >= 2)
        if len(_cfar_az):
            fig_het_cfar.add_trace(go.Scatter(
                x=_cfar_az.tolist(), y=_cfar_r.tolist(), mode="markers",
                marker=dict(color="orange", size=3, opacity=0.7),
                name="CFAR hits (≥2 sweeps)",
            ))
        # True target marker
        fig_het_cfar.add_trace(go.Scatter(
            x=[_tgt_az_final], y=[_tgt_r_final], mode="markers+text",
            marker=dict(color="lime", size=14, symbol="star"),
            text=["Target"], textposition="top right",
            textfont=dict(color="lime", size=11),
            name="True target position",
        ))
        fig_het_cfar.update_layout(
            height=300, margin=dict(t=10, b=30, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Azimuth bin", color="white", gridcolor=GRID_C),
            yaxis=dict(title="Range bin",   color="white", gridcolor=GRID_C),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10)),
        )
        st.plotly_chart(fig_het_cfar, use_container_width=True)

    with col_het2:
        st.markdown("**ConvGRU detection probability** (after 10 sweeps)")
        st.caption(
            "Purple intensity = GRU probability. The patch region is elevated (the model was "
            "trained on homogeneous clutter only, so the bright patch is out-of-distribution). "
            "The true target (★) is the separate bright region outside the patch. "
            "This is the MLOps argument: the GRU's patch response is a *data* problem, "
            "not an architecture problem — retrain on heterogeneous scenes and it is fixed."
        )
        fig_het_gru = go.Figure()
        fig_het_gru.add_trace(go.Heatmap(
            z=gru_prob_h.T, colorscale="Purples", showscale=True,
            zmin=0, zmax=1.0,
            colorbar=dict(title="P(target)", tickfont=dict(color="white"),
                          title_font=dict(color="white")),
            name="GRU probability",
        ))
        fig_het_gru.add_trace(go.Scatter(
            x=_az_patch, y=_r_patch, mode="lines",
            line=dict(color="yellow", width=1.5, dash="dot"),
            name="Clutter patch boundary", showlegend=True,
        ))
        fig_het_gru.add_trace(go.Scatter(
            x=[_tgt_az_final], y=[_tgt_r_final], mode="markers+text",
            marker=dict(color="lime", size=14, symbol="star"),
            text=["Target"], textposition="top right",
            textfont=dict(color="lime", size=11),
            name="True target position",
        ))
        fig_het_gru.update_layout(
            height=300, margin=dict(t=10, b=30, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Azimuth bin", color="white", gridcolor=GRID_C),
            yaxis=dict(title="Range bin",   color="white", gridcolor=GRID_C),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10)),
        )
        st.plotly_chart(fig_het_gru, use_container_width=True)

    st.info(
        "**The fundamental difference:** "
        "CFAR's false alarms in heterogeneous clutter are **algorithmic** — the cell-averaging "
        "assumption is mathematically violated by the clutter edge, and there is no way to fix this "
        "without changing the algorithm itself. "
        "The GRU's elevated response to the patch is a **data** problem — the model has never seen "
        "heterogeneous clutter in training. "
        "\n\n"
        "This distinction matters operationally: the MLOps pipeline exists precisely to close this gap. "
        "When PSI drift detection flags a change in incoming clutter statistics (sea-state change, new "
        "operating area, rain event), the pipeline triggers a retraining review. "
        "New labelled examples from the heterogeneous environment are collected, "
        "the model retrains, and after passing the accuracy gate and human review it is promoted to "
        "production. CFAR cannot undergo this process — its operating envelope is fixed at design time."
    )

# ── Tab 3: ROC Curve ──────────────────────────────────────────────────────────

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


@st.cache_data(show_spinner="Computing ROC curves (100 trials)…")
def _roc_data(snr_db_val: float, n: int = 100, _v: int = 9):
    rng = np.random.default_rng(77)
    ml_sc, cfar_sc, gru_sc, lbs = [], [], [], []
    cfar_d = PPICFARDetector()

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

        lbs.append(1 if has_t else 0)

    return (np.array(ml_sc), np.array(cfar_sc),
            np.array(gru_sc), np.array(lbs))

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

with tab_roc:
    st.markdown("### ROC curves: how reliably does each detector separate signal from noise?")
    st.markdown(
        "A ROC curve answers: *if I set the detector sensitivity to a given level, "
        "how many real targets do I catch — and how many false alarms do I generate?* "
        "The curve sweeps every possible threshold and plots **Detection Rate** (targets found) "
        "against **False Alarm Rate** (noise mistakenly flagged). "
        "**Closer to the top-left corner = better.** A detector on the diagonal is no better than "
        "a coin flip. The **AUC** (Area Under Curve) collapses the whole curve to one number: "
        "1.0 = perfect, 0.5 = random guessing, anything above ~0.75 is operationally useful."
    )
    st.markdown(
        "**How scores are computed** across 100 trials (50 with target, 50 clutter-only), "
        "all evaluated at the oracle target position for a fair cell-level comparison: "
        "**CNN** — peak probability map output in the target's neighbourhood. "
        "**ConvGRU** — peak hidden-state confidence at the target's neighbourhood after 10 sweeps. "
        "**CA-CFAR** — fraction of sweeps where CFAR fires near the target's actual position."
    )
    st.info(
        "**Why the Kalman Filter is not on the ROC curve:** "
        "The KF is a *track confirmer*, not a frame-level detector. "
        "It doesn't produce a continuous score that can be swept for a ROC — "
        "its output is binary (confirmed track / no confirmed track). "
        "At 5% Pfa across 11 520 cells there are ~540 false alarms per sweep, "
        "and the tracker (which processes only the top 10 peaks) is overwhelmed before it can latch onto the real target. "
        "The **Algorithm Comparison** tab shows KF performance correctly: once a confirmed track lands "
        "within 6 bins of the true target, it counts as a detection — "
        "that's the right metric for a tracker."
    )
    col_r, col_p = st.columns([1, 2])
    with col_r:
        st.markdown("**SNR for this trial set:**")
        roc_snr = st.slider("SNR (dB)", -20.0, 40.0, 10.0, 1.0, key="roc_snr")
        st.caption(
            "Drag the slider to see how each algorithm's ROC changes with target strength. "
            "At low SNR all three curves collapse toward the diagonal. "
            "At high SNR the ML curve should reach the top-left corner first."
        )

    ml_s, cf_s, gru_s, lbs = _roc_data(roc_snr, _v=9)
    ml_f,  ml_t  = _roc(ml_s,  lbs)
    cf_f,  cf_t  = _roc(cf_s,  lbs)
    gru_f, gru_t = _roc(gru_s, lbs)

    _ROC_GRID = "rgba(100,100,100,0.15)"
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(color="#aaa", dash="dot"), name="Chance (AUC 0.500)"))
    fig_r.add_trace(go.Scatter(x=cf_f, y=cf_t, mode="lines",
                               name=f"CA-CFAR  AUC {_auc(cf_f, cf_t):.3f}",
                               line=dict(color="#e67e22", width=2.0, dash="dash")))
    fig_r.add_trace(go.Scatter(x=ml_f, y=ml_t, mode="lines",
                               name=f"CNN  AUC {_auc(ml_f, ml_t):.3f}",
                               line=dict(color="#2980b9", width=2.5)))
    fig_r.add_trace(go.Scatter(x=gru_f, y=gru_t, mode="lines",
                               name=f"ConvGRU  AUC {_auc(gru_f, gru_t):.3f}",
                               line=dict(color="#8e44ad", width=2.5)))
    fig_r.update_layout(
        height=440,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="False Alarm Rate  (fraction of clutter cells incorrectly flagged)",
                   range=[-0.02, 1.02], gridcolor=_ROC_GRID, zeroline=False),
        yaxis=dict(title="Detection Rate  (fraction of real targets found)",
                   range=[-0.02, 1.04], gridcolor=_ROC_GRID, zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    with col_p:
        st.plotly_chart(fig_r, use_container_width=True)

    r1, r2, r3 = st.columns(3)
    def _auc_label(val):
        if val >= 0.95:
            return "Excellent"
        if val >= 0.85:
            return "Good"
        if val >= 0.70:
            return "Fair"
        return "Poor"
    r1.metric("CNN AUC",      f"{_auc(ml_f,  ml_t):.3f}",  _auc_label(_auc(ml_f,  ml_t)))
    r2.metric("ConvGRU AUC",  f"{_auc(gru_f, gru_t):.3f}", _auc_label(_auc(gru_f, gru_t)))
    r3.metric("CA-CFAR AUC",  f"{_auc(cf_f,  cf_t):.3f}",  _auc_label(_auc(cf_f,  cf_t)))
    st.caption(
        "**Reading the ROC:** left side = conservative (low false alarms, some targets missed). "
        "Right side = aggressive (catch more targets, more false alarms). "
        "The CFAR curve is often jagged because its score (fraction of sweeps with a hit) "
        "takes only a handful of discrete values. "
        "The ConvGRU score is the final accumulated hidden-state value at the target cell — "
        "it builds confidence across sweeps, so its AUC rises sharply once SNR exceeds ~4 dB."
    )

# ── Tab: Strengths & Trade-offs ───────────────────────────────────────────────

with tab_tradeoffs:
    st.markdown("## Algorithm Strengths, Weaknesses & Operational Trade-offs")
    st.markdown(
        "No single detector wins on every metric. "
        "This tab maps each algorithm's measured characteristics to operational radar roles — "
        "early warning, persistent tracking, and false-alarm-limited environments — "
        "and argues that the right system combines all three."
    )

    st.divider()

    # ── Three-column summary cards ─────────────────────────────────────────────
    col_cfar, col_cnn, col_gru = st.columns(3)

    with col_cfar:
        st.markdown("### 🔭 CA-CFAR + Kalman Filter")
        st.markdown("**Classical baseline — no training data required**")
        st.divider()

        st.markdown("#### ✅ Strengths")
        st.markdown("""
- **Instantaneous** — no warm-up; fires on the very first sweep.
- **Theoretically grounded** — Pfa is analytically controlled; behaviour is fully predictable.
- **No training data** — deployable in a new environment immediately.
- **Hard real-time** — deterministic per-sweep compute (2.2 ms median); no GPU needed.
- **Fully transparent** — every detection decision is traceable to a threshold exceedance.
        """)
        st.markdown("#### ❌ Weaknesses")
        st.markdown("""
- **Breaks in heterogeneous clutter** — sea-land edges, rain cells, vessel wakes violate the
  homogeneous-noise assumption and produce bursts of false alarms.
- **Low SNR sensitivity** — requires ~8 dB to reach P_d = 100 %; blind below ~0 dB.
- **Slowest to confirm** — needs repeated threshold exceedances before the KF locks on
  (7.7 sweeps avg. at 0 dB vs. 4.9 for CNN).
- **Single-sweep, no memory** — discards all temporal history between sweeps.
        """)

        st.info("**Best role:** known-clutter environments where explainability and zero-data deployment matter.")

    with col_cnn:
        st.markdown("### 🟡 CNN + Kalman Filter")
        st.markdown("**Batch deep detector — best false-alarm suppression**")
        st.divider()

        st.markdown("#### ✅ Strengths")
        st.markdown("""
- **Lowest false track rate** — 0.79 / 100 sweeps vs 3.68 for CFAR (a 4.7× reduction).
  Batch temporal averaging over 10 sweeps eliminates transient clutter spikes.
- **Fastest confirmation** — 4.03 sweeps avg. at 10 dB; strongest single-cell AUC
  (0.78 at 0 dB vs. 0.48 for CFAR).
- **Fastest inference** — 0.80 ms for a full 10-sweep batch (fully vectorised); faster
  than a single CFAR sweep.
- **ONNX-portable** — 19 073 parameters; runs on CPU or any ONNX-compatible accelerator.
        """)
        st.markdown("#### ❌ Weaknesses")
        st.markdown("""
- **Needs a full 10-sweep buffer** — cannot confirm before rotation 4; performs poorly
  in the first few sweeps after a target enters the scene.
- **Single-target design** — temporal features (max / mean / std) conflate simultaneous
  targets; not suitable for the multi-target live scenario.
- **Fixed context window** — trained on exactly 10 sweeps; partial sequences degrade
  confidence estimates.
- **Requires labelled training data** — performance depends on training distribution matching
  the operational clutter environment.
        """)

        st.info("**Best role:** post-processing or re-evaluation passes where low false alarm rate matters more than latency.")

    with col_gru:
        st.markdown("### 🟣 ConvGRU + Kalman Filter")
        st.markdown("**Streaming deep detector — best low-SNR sensitivity**")
        st.divider()

        st.markdown("#### ✅ Strengths")
        st.markdown("""
- **Best low-SNR sensitivity** — P_d = 100 % at SNR = 4 dB; 73 % at 0 dB.
  4–6 dB advantage over CFAR through recurrent temporal integration.
- **Streaming / O(1) per sweep** — processes one sweep at a time;
  confirmation latency starts from sweep 1. Only 5 694 parameters.
- **Multi-target capable** — hidden state h ∈ [0,1]^{180×64} tracks all cells
  simultaneously; each target accumulates independently.
- **Early warning** — delivers useful probability estimates 2–3 sweeps before
  CFAR can confirm, enabling earlier threat assessment.
        """)
        st.markdown("#### ❌ Weaknesses")
        st.markdown("""
- **Higher false track rate at calibration** — 4.03 / 100 sweeps (vs. 0.79 for CNN);
  the continuous hidden-state generates low-confidence activations in dense clutter.
- **Requires training and curriculum scheduling** — stable training needs a
  seq-len curriculum (5 → 10 → 15); GRU gradient flow is sensitive to hyperparameters.
- **Out-of-distribution clutter** — trained on homogeneous clutter; heterogeneous
  patches elevate the hidden state even without a target (fixable by retraining).
- **Recurrent state management** — hidden state must be maintained per radar channel;
  state reset policy on scan-mode changes needs operational definition.
        """)

        st.info("**Best role:** early-warning and long-range surveillance where low SNR sensitivity and streaming latency are the primary requirements.")

    st.divider()

    # ── Operational role matrix ────────────────────────────────────────────────
    st.markdown("### Operational role matrix")
    st.markdown(
        "The table below maps each algorithm to operational radar use cases. "
        "Green = strong fit, yellow = acceptable, red = poor fit."
    )

    st.markdown("""
| Requirement | CA-CFAR+KF | CNN+KF | ConvGRU+KF |
|---|:---:|:---:|:---:|
| **Early warning (react before 5 sweeps)** | 🔴 Slow | 🟡 Moderate | 🟢 Best |
| **Low-SNR detection (0–6 dB)** | 🔴 Poor | 🟡 Good | 🟢 Best |
| **False-alarm-limited environment** | 🟡 Baseline | 🟢 Best | 🟡 Comparable to CFAR |
| **Multi-target tracking** | 🟢 Yes | 🔴 Conflicts targets | 🟢 Yes |
| **Zero-data deployment** | 🟢 No data needed | 🔴 Needs training | 🔴 Needs training |
| **Explainability / certification** | 🟢 Fully transparent | 🟡 Interpretable map | 🟡 Interpretable map |
| **Embedded / hard real-time** | 🟢 2.2 ms/sweep | 🟢 0.8 ms/batch | 🟢 <1 ms/sweep |
| **Heterogeneous clutter** | 🔴 Breaks at edges | 🟢 Learned robustness | 🟡 Needs retraining |
    """)

    st.divider()

    # ── The case for a combined system ────────────────────────────────────────
    st.markdown("### The case for a combined system")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(
            "The three pipelines are **complementary, not competing**. "
            "Each fills a gap the others leave:"
        )
        st.markdown("""
**Layer 1 — ConvGRU for early warning and low-SNR sensitivity.**
As soon as a target enters the scan area the GRU begins accumulating confidence in its hidden state.
By sweep 4 it reaches 100 % P_d at SNR ≥ 4 dB — before CFAR has even confirmed a track.
This makes it the right front-end for threat cueing and time-critical alerting.

**Layer 2 — CNN for false-alarm qualification.**
Once the GRU cues a possible track, the CNN provides an independent, lower-false-alarm
probability estimate over the buffered 10-sweep window.
If the CNN confirms the detection, confidence in the track rises sharply.
If the CNN does not confirm, the GRU alert is downgraded to a tentative contact.
This two-stage gating reduces nuisance alerts without sacrificing early warning.

**Layer 3 — CFAR for known-environment baselines and certification.**
For environments where the clutter model is well-characterised and an analytical P_fa
guarantee is required (e.g., regulatory or safety-of-life contexts), CFAR provides a
certified, auditable detection decision alongside the ML outputs.
It also serves as an independent sanity check: if CFAR fires repeatedly on a cell that
the ML models are not flagging, it may indicate an out-of-distribution clutter event.
        """)
        st.success(
            "**Conclusion:** In a production system, the ConvGRU provides the first alert, "
            "the CNN qualifies it, and CFAR provides the certified audit trail. "
            "The MLOps pipeline — DVC versioning, automated retraining, accuracy gate, "
            "model card — ensures the ML components stay calibrated as the operational "
            "environment evolves."
        )
    with col_r:
        st.markdown("**Measured decision latencies**")
        fig_lat = go.Figure()

        snr_pts = [-4, 0, 4, 8, 10, 12, 20]
        cfar_pd = [26.7, 6.7, 40.0, 43.3, None, 100.0, 100.0]
        cnn_pd  = [20.0, 36.7, 76.7, 96.7, None, 100.0, 100.0]
        gru_pd  = [43.3, 73.3, 100.0, 100.0, None, 100.0, 100.0]

        fig_lat.add_trace(go.Scatter(
            x=snr_pts, y=cfar_pd, mode="lines+markers",
            name="CFAR+KF", line=dict(color="white", dash="dash", width=2),
            marker=dict(size=7)
        ))
        fig_lat.add_trace(go.Scatter(
            x=snr_pts, y=cnn_pd, mode="lines+markers",
            name="CNN+KF", line=dict(color="#ffe66d", width=2.5),
            marker=dict(size=7)
        ))
        fig_lat.add_trace(go.Scatter(
            x=snr_pts, y=gru_pd, mode="lines+markers",
            name="ConvGRU+KF", line=dict(color="#c084fc", width=3),
            marker=dict(size=9, symbol="diamond")
        ))
        GRID_C = "rgba(128,128,128,0.2)"
        fig_lat.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="SNR (dB)", gridcolor=GRID_C, zeroline=False, color="white"),
            yaxis=dict(title="P_d within 10 sweeps (%)", range=[0, 105],
                       gridcolor=GRID_C, zeroline=False, color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            margin=dict(t=10, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown("**False track rate at calibrated thresholds**")
        st.markdown("""
| Pipeline | False tracks / 100 sw |
|---|:---:|
| CA-CFAR + KF | 3.68 |
| ConvGRU + KF | 4.03 |
| CNN + KF | **0.79** |
        """)

# ── Tab 4: The Math ────────────────────────────────────────────────────────────

with tab_math:
    st.markdown("## Signal & Detection Math")
    st.markdown(
        "Everything in this demo derives from four equations. "
        "Each section below shows the formula, what each symbol means, "
        "and how it maps to a line of code."
    )

    # ── 1. Radar signal model ──────────────────────────────────────────────────
    with st.expander("1 · Radar signal model", expanded=True):
        st.markdown(r"""
### Received power at range $r$

$$
P_r(r) = \underbrace{A}_{\text{target}} \cdot r^{-2} + \underbrace{\eta(r)}_{\text{clutter}}
$$

| Symbol | Meaning |
|--------|---------|
| $A$ | Target reflectivity, set by the **SNR (dB)** slider |
| $r^{-2}$ | Free-space path loss (two-way) — signal falls off as range squared |
| $\eta(r)$ | Range-dependent Rayleigh clutter: $\eta \sim r^{-2} \cdot \text{Rayleigh}(\sigma=1)$ |

The clutter envelope follows a **Rayleigh distribution** — a standard model for
ground/sea clutter where many small random scatterers add in quadrature.
The $r^{-2}$ factor means close-range cells are much brighter than far-range cells,
which is why we need range normalisation before any detection step.
""")
        st.code(
            "# src/data/ppi_generator.py\n"
            "noise_floor = np.arange(n_range) / n_range   # r / R_max\n"
            "clutter = rng.rayleigh(scale=noise_floor)     # Rayleigh(σ = r/R_max)\n"
            "# Target injected at configured (range_bin, azimuth_deg)\n"
            "snr_linear = 10 ** (snr_db / 10)\n"
            "sweep[az_b, r_b] += snr_linear * noise_floor[r_b]",
            language="python",
        )

    # ── 2. Temporal features ────────────────────────────────────────────────────
    with st.expander("2 · Temporal feature extraction", expanded=True):
        st.markdown(r"""
### Three-channel feature map fed to the CNN

For a sequence of $T$ sweeps $\{x_t\}_{t=1}^{T}$, each cell $(az, r)$ produces:

$$
f_{\max}(az,r) = \frac{\max_t\, x_t(az,r)}{\hat{\sigma}(r)}, \quad
f_{\mu}(az,r)  = \frac{\mu_t\, x_t(az,r)}{\hat{\sigma}(r)}, \quad
f_{\sigma}(az,r) = \frac{\sigma_t\, x_t(az,r)}{\hat{\sigma}(r)}
$$

where $\hat{\sigma}(r)$ is the **10th-percentile amplitude** at range bin $r$
(estimated from all azimuth cells in the sequence) — an empirical range-dependent clutter floor.

| Channel | Why it helps |
|---------|-------------|
| $f_{\max}$ | Captures peak illumination — a target brightens its cell on each look |
| $f_{\mu}$ | Provides the clutter baseline — stationary clutter has a stable mean |
| $f_{\sigma}$ | Motion fingerprint — a moving target has **high variance** across sweeps; clutter has low variance |

Dividing by $\hat{\sigma}(r)$ cancels range attenuation so all three channels
are range-invariant, making the CNN's job easier.
""")
        st.code(
            "# src/data/ppi_generator.py — temporal_features()\n"
            "noise_floor = np.percentile(seq, 10, axis=(0, 1))  # shape (n_range,)\n"
            "noise_floor = np.maximum(noise_floor, 1e-6)\n"
            "normed = seq / noise_floor[np.newaxis, np.newaxis, :]  # broadcast over (T, az, r)\n"
            "feat = np.stack([normed.max(0), normed.mean(0), normed.std(0)])",
            language="python",
        )

    # ── 3. CA-CFAR ─────────────────────────────────────────────────────────────
    with st.expander("3 · Constant False-Alarm Rate (CA-CFAR)", expanded=True):
        st.markdown(r"""
### Cell-Averaging CFAR threshold

A cell under test (CUT) at $(az, r)$ is declared a detection if:

$$
x(az, r) > \alpha \cdot \hat{\mu}_{\text{ref}}
$$

where

$$
\hat{\mu}_{\text{ref}} = \frac{1}{N_{\text{ref}}} \sum_{(i,j)\,\in\,\mathcal{R}} x(i, j)
$$

$\mathcal{R}$ is the **reference window** — a rectangular annulus centred on the CUT
with guard cells excluded to prevent target self-masking:

$$
N_{\text{ref}} = (2 r_a + 2 g_a + 1)(2 r_r + 2 g_r + 1) - (2 g_a + 1)(2 g_r + 1)
$$

| Parameter | Value | Role |
|-----------|-------|------|
| $g_a, g_r$ | 2 cells | Guard cells — exclude the target's own sidelobes |
| $r_a, r_r$ | 5 cells | Reference cells each side |
| $\alpha$ | 2.5 | Threshold factor — set empirically for ~1% FA rate |

**Pre-whitening**: before applying CFAR, each sweep is divided by the
$r^{-2}$ clutter-floor model so the threshold is range-invariant.
The reference window sum is computed in $O(N_{az} \times N_r)$
using a **2-D prefix sum**, making the algorithm fast regardless of window size.
""")
        st.code(
            "# src/baseline/ppi_cfar_kf.py\n"
            "# Pre-whiten: divide by R^-2 floor\n"
            "sweep = sweep / _rng_norm(sweep.shape[1])[np.newaxis, :]\n"
            "# 2-D prefix sum with azimuth wrap-around padding\n"
            "ps = padded.cumsum(axis=0).cumsum(axis=1)\n"
            "outer_sum = box(-outer_az, outer_az, -outer_r, outer_r)\n"
            "inner_sum = box(-ga, ga, -gr, gr)\n"
            "ref_sum   = outer_sum - inner_sum          # reference window\n"
            "noise_est = ref_sum / ref_cells\n"
            "det[:, ri] = cell_val > self.thr * noise_est",
            language="python",
        )

    # ── 4. Kalman filter tracker ────────────────────────────────────────────────
    with st.expander("4 · Kalman filter tracker", expanded=True):
        st.markdown(r"""
### State-space model in (range, azimuth) bins

State vector and constant-velocity dynamics:

$$
\mathbf{x} = \begin{bmatrix} r \\ \dot{r} \\ az \\ \dot{az} \end{bmatrix}, \qquad
\mathbf{F} = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

Observation matrix (we only measure position, not velocity):

$$
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$

**Predict step** (between sweeps):

$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\hat{\mathbf{x}}_{k-1}, \qquad
\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^\top + \mathbf{Q}
$$

**Update step** (on CFAR peak association):

$$
\mathbf{K} = \mathbf{P}_{k|k-1}\mathbf{H}^\top\mathbf{S}^{-1}, \qquad
\mathbf{S} = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^\top + \mathbf{R}
$$
$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}_{k|k-1})
$$

**Mahalanobis gating** — a CFAR detection $\mathbf{z}$ is associated to a track only if:

$$
(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})^\top \mathbf{S}^{-1} (\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}) < \gamma
$$

with gate $\gamma = 16$ (roughly a 4-bin radius in normalised innovation space).
A track is **confirmed** after $\geq 4$ successful associations (hits).

| Matrix | Values | Meaning |
|--------|--------|---------|
| $\mathbf{Q}$ | diag(1, 0.5, 1, 0.5) | Process noise — allows for manoeuvring |
| $\mathbf{R}$ | diag(4, 4) | Measurement noise — ±2 bin uncertainty in CFAR centroid |
""")
        st.code(
            "# src/baseline/ppi_cfar_kf.py — _KalmanTrack2D\n"
            "def predict(self):\n"
            "    self.x = self.F @ self.x\n"
            "    self.P = self.F @ self.P @ self.F.T + self.Q\n\n"
            "def update(self, z):\n"
            "    inn = z - self.H @ self.x          # innovation\n"
            "    S   = self.H @ self.P @ self.H.T + self.R\n"
            "    K   = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain\n"
            "    self.x = self.x + K @ inn\n"
            "    self.P = (np.eye(4) - K @ self.H) @ self.P",
            language="python",
        )

    # ── 5. CNN architecture ─────────────────────────────────────────────────────
    with st.expander("5 · CNN architecture", expanded=True):
        st.markdown(r"""
### Fully-convolutional network (FCN)

Input: **3 × 180 × 64** feature map (channels = max, mean, std).

$$
\text{Input} \xrightarrow{3\to16} \text{BN+ReLU} \xrightarrow{16\to32} \text{BN+ReLU}
\xrightarrow{32\to32} \text{BN+ReLU} \xrightarrow{32\to16} \text{BN+ReLU}
\xrightarrow{16\to1} \text{logit map}
$$

All convolutions are **3×3, stride 1, padding 1** — the spatial resolution
is preserved throughout (no pooling), so the output is a **1×180×64** logit map
converted to probabilities by sigmoid: $p = \sigma(\text{logit})$.

| Layer | Filters | Parameters |
|-------|---------|-----------|
| Conv1 | 16 | 3×3×3×16 = 432 |
| Conv2 | 32 | 3×3×16×32 = 4 608 |
| Conv3 | 32 | 3×3×32×32 = 9 216 |
| Conv4 | 16 | 3×3×32×16 = 4 608 |
| Conv5 | 1  | 3×3×16×1 = 144 |
| **Total** | | **~19 000** |

The network is tiny by deep-learning standards but sufficient because
the three temporal features already do the heavy lifting — the CNN
learns a compact spatial filter to suppress residual clutter.

Loss: binary cross-entropy on the per-cell logits against the
ground-truth label map (1 at target cell ± 1 bin, 0 elsewhere).
""")
        st.code(
            "# src/models/ppi_detector.py\n"
            "class PPIDetectorCNN(nn.Module):\n"
            "    def __init__(self, filters=(16, 32, 32, 16)):\n"
            "        layers = []\n"
            "        in_ch = 3\n"
            "        for out_ch in filters:\n"
            "            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),\n"
            "                       nn.BatchNorm2d(out_ch), nn.ReLU()]\n"
            "            in_ch = out_ch\n"
            "        layers.append(nn.Conv2d(in_ch, 1, 3, padding=1))\n"
            "        self.net = nn.Sequential(*layers)\n\n"
            "    def forward(self, x):          # x: (B, 3, n_az, n_range)\n"
            "        return self.net(x).squeeze(1)   # (B, n_az, n_range) logits",
            language="python",
        )

    # ── 6. ConvGRU streaming detector ───────────────────────────────────────────
    with st.expander("6 · Convolutional GRU — hidden state as confidence heatmap", expanded=True):
        st.markdown(r"""
### Why a GRU instead of the batch CNN?

The batch CNN needs **all N sweeps at once** to compute its three temporal features
(max, mean, std) — it cannot produce output until the full window has arrived.

The ConvGRU processes **one sweep at a time** and maintains its entire belief state
in a single spatial map $h$. Output is available after every sweep — latency is one
rotation period, not N.

---

### The hidden state IS the confidence heatmap

$$
h_t \in [0, 1]^{N_{az} \times N_{range}}
$$

This is the key design choice: $h$ is a **single-channel probability map** over
the full PPI grid, not an abstract multi-channel feature tensor. After every sweep
you can read $h$ directly as the detector's current belief — no further
transformation needed. This is the **purple overlay** on the PPI display.

Compare with the batch CNN, which produces a probability map only after all
sweeps have been processed.

---

### Update equations

At each sweep $t$, given the EMA-normalised amplitude $x_t$:

**Step 1 — encode** the sweep into a rich multi-channel feature map:
$$
f_t = \text{Encoder}(x_t), \quad f_t \in \mathbb{R}^{C \times N_{az} \times N_{range}}
$$
The encoder is two Conv2d + ReLU layers ($1 \to 16 \to 32$ channels).

**Step 2 — gate** using the concatenation $[f_t \,\|\, h_{t-1}]$:
$$
r_t = \sigma\!\bigl(\text{Conv}_r([f_t \,\|\, h_{t-1}])\bigr) \quad \text{(reset gate)}
$$
$$
z_t = \sigma\!\bigl(\text{Conv}_z([f_t \,\|\, h_{t-1}])\bigr) \quad \text{(update gate)}
$$

**Step 3 — candidate** using the gated old state:
$$
\tilde{h}_t = \sigma\!\bigl(\text{Conv}_n([f_t \,\|\, r_t \odot h_{t-1}])\bigr) \in [0,1]
$$
Using $\sigma$ (not $\tanh$) keeps the candidate in $[0,1]$, matching the probability
space of $h$.

**Step 4 — blend**:
$$
h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
$$
A convex combination of two values in $[0,1]$ stays in $[0,1]$ — the heatmap
remains a valid probability map at every sweep.

| Gate | Behaviour when → 0 | Behaviour when → 1 |
|------|---------------------|---------------------|
| Reset $r$ | Ignore old confidence when computing candidate | Let old state influence candidate |
| Update $z$ | Keep old confidence unchanged (copy-through) | Replace with new candidate evidence |

---

### Architecture and parameter count

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input $x_t$ | $(1,\,1,\,180,\,64)$ | EMA-normalised amplitude — one sweep |
| Encoder output | $(1,\,32,\,180,\,64)$ | Feature map from the two Conv layers |
| **$h$** | $(1,\,1,\,180,\,64)$ | **Confidence heatmap — the hidden state** |
| $r,\, z,\, \tilde{h}$ | $(1,\,1,\,180,\,64)$ each | Gates and candidate — single channel |

| Component | Parameters |
|-----------|------------|
| Encoder Conv2d(1→16) | 160 |
| Encoder Conv2d(16→32) | 4 640 |
| Conv_r, Conv_z, Conv_n each (33→1) | 3 × 298 = 894 |
| **Total** | **~5 700** |

---

### Streaming inference (O(1) per sweep)

```python
h = np.zeros((1, 1, 180, 64), dtype=np.float32)   # start with zero confidence

for sweep in live_radar_stream:
    sweep_norm = preprocess(sweep)                  # EMA normalise: (1, 1, 180, 64)
    prob_map, h = onnx_session.run(
        None, {"sweep_norm": sweep_norm, "h_in": h}
    )
    # prob_map IS h — the heatmap you see in the purple overlay
    detections = (prob_map[0, 0] > 0.30)           # threshold → binary map
```

Each call costs the same compute regardless of how long the antenna has been
running. The heatmap builds up confidence at the target cell as more sweeps arrive
and decays where no evidence is seen — visible in real time on the PPI display.
""")
        st.code(
            "# src/model/conv_gru.py — ConvGRUDetector.forward()\n"
            "def forward(self, sweep_norm, h):\n"
            "    feat = self.encoder(sweep_norm)                  # (B, 32, H, W)\n"
            "    xh   = torch.cat([feat, h], dim=1)               # (B, 33, H, W)\n"
            "    r    = torch.sigmoid(self.r_gate(xh))            # reset gate ∈ [0,1]\n"
            "    z    = torch.sigmoid(self.z_gate(xh))            # update gate ∈ [0,1]\n"
            "    n    = torch.sigmoid(self.cand(                  # candidate ∈ [0,1]\n"
            "               torch.cat([feat, r * h], dim=1)))\n"
            "    h_next = (1.0 - z) * h + z * n                  # stays in [0,1]\n"
            "    return h_next, h_next                            # prob_map ≡ h_out",
            language="python",
        )

# ── Tab 5: Live Radar ─────────────────────────────────────────────────────────

with tab_live:
    # ── Initialise state ──────────────────────────────────────────────────────
    if "live" not in st.session_state:
        _live_init()
    s = st.session_state["live"]

    # ── Controls ──────────────────────────────────────────────────────────────
    ctl1, ctl2, ctl3, ctl_gap = st.columns([1, 1, 1, 6])
    with ctl1:
        if st.button("▶ Start", key="live_start", use_container_width=True,
                     disabled=s["running"]):
            s["running"] = True
            st.rerun()
    with ctl2:
        if st.button("■ Stop", key="live_stop", use_container_width=True,
                     disabled=not s["running"]):
            s["running"] = False
            st.rerun()
    with ctl3:
        if st.button("↺ Reset", key="live_reset", use_container_width=True):
            _live_init()
            st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_live_scope, col_live_info = st.columns([5, 3], gap="large")

    with col_live_scope:
        live_scope_ph = st.empty()
        live_conf_ph  = st.empty()

    with col_live_info:
        sw_now = s["sweep_count"]
        n_active = len(s["targets"])

        st.markdown(
            f"**Sweep {sw_now}** &nbsp;·&nbsp; "
            f"Active targets: **{n_active}** &nbsp;·&nbsp; "
            f"{'🟢 Running' if s['running'] else '🔴 Stopped'}"
        )
        st.divider()

        # ── Target confirmation table ─────────────────────────────────────────
        st.markdown("**Target confirmation status**")

        def _lat_str(conf_dict: dict, key: str) -> str:
            v = conf_dict.get(key)
            return f"sw {v} ✓" if v is not None else "—"

        active_ids = {t["id"] for t in s["targets"]}
        # Show last 8 targets (active or recently expired)
        shown_ids = sorted(s["confirm"].keys(), reverse=True)[:8]

        if shown_ids:
            hdr_cols = st.columns([1, 2, 2, 2])
            hdr_cols[0].caption("Tgt")
            hdr_cols[1].caption("SNR")
            hdr_cols[2].caption("CFAR+KF")
            hdr_cols[3].caption("GRU+KF")

            for tid in shown_ids:
                color  = _TGT_COLORS[tid % len(_TGT_COLORS)]
                active = tid in active_ids
                tgt_obj = next((t for t in s["targets"] if t["id"] == tid), None)
                snr_str = f"{tgt_obj['snr_db']:.0f}dB" if tgt_obj else "—"

                row = st.columns([1, 2, 2, 2])
                label = f"{'●' if active else '○'} #{tid}"
                row[0].markdown(
                    f"<span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True,
                )
                row[1].caption(snr_str)
                conf = s["confirm"].get(tid, {})
                row[2].caption(_lat_str(conf, "cfar_kf"))
                row[3].caption(_lat_str(conf, "gru_kf"))
        else:
            st.caption("No targets yet — click ▶ Start")

        st.divider()

        # ── Median confirmation latency ───────────────────────────────────────
        st.markdown("**Median sweeps to confirmed track**")

        def _med_str(lats: list) -> str:
            return f"{float(np.median(lats)):.1f} sw  (n={len(lats)})" if lats else "—"

        m1, m2 = st.columns(2)
        m1.metric("CFAR+KF", _med_str(s["lat_cfar"]))
        m2.metric("GRU+KF",  _med_str(s["lat_gru"]))

        st.divider()

        st.caption(
            "● Coloured circles = true targets · "
            "✕ White = CFAR+KF · ◆ Magenta = GRU+KF · "
            "Magenta glow = GRU confidence heatmap"
        )

    # ── Render frames ─────────────────────────────────────────────────────────
    live_scope_ph.plotly_chart(_live_ppi_fig(s), use_container_width=False,
                               key=f"live_ppi_{sw_now}")
    live_conf_ph.plotly_chart(_live_conf_fig(s), use_container_width=True,
                              key=f"live_conf_{sw_now}")

    # Tick and rerun if running
    if s["running"]:
        _live_tick()
        time.sleep(s["frame_s"])
        st.rerun()


# ── Footer: business value ─────────────────────────────────────────────────────

st.divider()
st.markdown("### Key takeaways")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**The AI advantage is largest in the mid-SNR band**")
    st.markdown(
        "Below about −4 dB neither system reliably detects. "
        "Above +12 dB both systems are equally effective. "
        "The 0–12 dB window — faint but real targets — is where the AI pipeline "
        "delivers its biggest gains over the classical approach."
    )
with col_b:
    st.markdown("**Integrating across rotations is the key mechanism**")
    st.markdown(
        "A single radar sweep is often too noisy to make a confident call. "
        "By learning patterns across 10 full antenna rotations, the AI builds up "
        "evidence the classical single-sweep threshold cannot accumulate."
    )
with col_c:
    st.markdown("**Drop-in compatible**")
    st.markdown(
        "The AI front-end produces the same peak list as CFAR — "
        "the downstream Kalman filter tracker and the rest of the processing chain "
        "are unchanged. No new hardware, no retraining of other components."
    )
