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
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered Radar Target Detection",
    page_icon="📡",
    layout="wide",
)

# ── Cached resources ───────────────────────────────────────────────────────────

params  = load_params()
session = load_cnn_session()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📡 AI-Powered Radar Target Detection")
st.markdown(
    "**This demo shows how a machine-learning system detects and tracks a moving target in radar "
    "data — automatically, and with greater reliability than traditional rule-based methods.** "
    "A synthetic rotating radar generates realistic clutter; classical and machine-learning detection algorithms compete "
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
| **GRU / ConvGRU** | Gated Recurrent Unit — a neural network that processes one radar sweep at a time and maintains a running "memory" of what it has seen; ConvGRU applies this spatially across the full radar image |
| **Transformer** | A neural network that uses "attention" to learn which radar sweeps are most informative at each location, without hand-crafting temporal features |
| **LRT** | Likelihood Ratio Test — a classical detector that adds up signal energy across multiple sweeps to improve detection of faint, stationary targets |
| **DP-TBD** | Dynamic-Programming Track-Before-Detect — a classical detector that traces the best possible target path through multiple sweeps before deciding if it is real |
| **TSPI** | Time Space Position Information — the precise 3D position, velocity, and acceleration output of an instrumentation radar; used as ground-truth labels for ML training |
| **PSI** | Population Stability Index — a metric that measures how much an incoming data distribution has shifted relative to the training reference; PSI > 0.20 triggers a retraining review |
| **MFCW** | Multi-Frequency Continuous Wave — a high-precision radar waveform used in Weibel's instrumentation radars; enables simultaneous range and velocity measurement at sub-centimetre accuracy |
| **OTA** | Over-The-Air — deploying a new model version to field units remotely, without physical access |
""")

if session is None:
    st.warning(
        "ONNX model not found at `artifacts/ppi_model.onnx`. "
        "Run `python src/train_ppi.py` then `python scripts/export_onnx_ppi.py` first. "
        "CFAR and KF will still work."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────

from tabs import classical, bridge, math_tab, comparison, roc, tradeoffs, live_radar, paper, pipeline  # noqa: E402

(tab_classical, tab_bridge, tab_compare,
 tab_roc, tab_tradeoffs, tab_pipeline, tab_math, tab_live, tab_paper) = st.tabs([
    "🔭  Classical Radar",
    "🤖  From CFAR to ML",
    "📊  Algorithm Comparison",
    "📉  ROC Curve",
    "⚖️  Strengths & Trade-offs",
    "🔁  MLOps Pipeline",
    "📐  The Math",
    "📻  Live Radar",
    "📄  Paper",
])

with tab_classical:
    classical.render()

with tab_bridge:
    bridge.render()

with tab_compare:
    comparison.render()

with tab_roc:
    roc.render()

with tab_tradeoffs:
    tradeoffs.render()

with tab_pipeline:
    pipeline.render()

with tab_math:
    math_tab.render()

with tab_live:
    live_radar.render()

with tab_paper:
    paper.render()


