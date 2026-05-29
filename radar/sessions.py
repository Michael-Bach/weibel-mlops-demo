"""
radar/sessions.py — cached ONNX sessions and shared constants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import yaml
import onnxruntime as ort

# ── Constants ──────────────────────────────────────────────────────────────────

N_SW    = 10
N_AZ    = 180
N_RANGE = 64
N_GRID  = 300

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

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_params():
    with open("params_ppi.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_cnn_session():
    p = Path("artifacts/ppi_model.onnx")
    return ort.InferenceSession(str(p)) if p.exists() else None


@st.cache_resource
def load_gru_session():
    p = Path("artifacts/recurrent_model.onnx")
    return ort.InferenceSession(str(p)) if p.exists() else None


@st.cache_resource
def load_transformer_session():
    p = Path("artifacts/transformer_model.onnx")
    return ort.InferenceSession(str(p)) if p.exists() else None
