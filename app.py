"""
Streamlit demo — PPI Rotating Radar AI Target Detector.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from radar.sessions import (
    load_params,
    load_cnn_session,
    load_gru_session,
    N_RANGE,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered Radar Target Detection",
    page_icon="📡",
    layout="wide",
)

# ── Cached resources ───────────────────────────────────────────────────────────

params      = load_params()
session     = load_cnn_session()
gru_session = load_gru_session()

# ── Header ─────────────────────────────────────────────────────────────────────

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

# ── Sidebar ────────────────────────────────────────────────────────────────────

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

# ── Tabs ───────────────────────────────────────────────────────────────────────

from tabs import classical, bridge, math_tab, ppi_display, comparison, roc, tradeoffs, live_radar, paper  # noqa: E402

(tab_classical, tab_bridge, tab_math, tab_ppi, tab_compare,
 tab_roc, tab_tradeoffs, tab_live, tab_paper) = st.tabs([
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

with tab_classical:
    classical.render()

with tab_bridge:
    bridge.render()

with tab_math:
    math_tab.render()

with tab_ppi:
    ppi_display.render(snr_db, range_bin, az_deg, vr, vt,
                       show_ml, show_cfar, show_kf, show_arpa, show_gru_kf, show_cnn_kf)

with tab_compare:
    comparison.render(snr_db)

with tab_roc:
    roc.render()

with tab_tradeoffs:
    tradeoffs.render()

with tab_live:
    live_radar.render()

with tab_paper:
    paper.render()

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
