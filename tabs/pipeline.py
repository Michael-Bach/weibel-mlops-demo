"""
tabs/pipeline.py — MLOps pipeline tab: from test range to fleet deployment.

Tells the story of how Weibel's high-precision instrumentation radars act as
a label factory for the mass-produced XENTA operational fleet — and how the
ML pipeline keeps the fleet calibrated as environments change.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.drift_detect import compute_psi  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

_MLFLOW_RUN = Path(
    "mlruns/168499014080440538/7f03b68e8ac54ad6aa9134b9462550f2"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_metric(name: str):
    """Parse MLflow metric file → (steps, values)."""
    p = _MLFLOW_RUN / "metrics" / name
    if not p.exists():
        return [], []
    steps, values = [], []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            steps.append(int(parts[2]))
            values.append(float(parts[1]))
    return steps, values


def _read_param(name: str) -> str:
    p = _MLFLOW_RUN / "params" / name
    return p.read_text().strip() if p.exists() else "?"


def _psi_from_shift(shift: float) -> dict:
    """Simulate PSI for a given combined clutter + drone-type shift."""
    rng = np.random.default_rng(42)
    ref_energy = rng.rayleigh(scale=1.0, size=2000)
    cur_energy = rng.rayleigh(scale=1.0 + shift, size=500) + shift * 0.25
    ref_centroid = rng.normal(0.30, 0.05, 2000)
    cur_centroid = rng.normal(0.30 + shift * 0.18, 0.05 + shift * 0.04, 500)

    psi_e = compute_psi(ref_energy, cur_energy)
    psi_c = compute_psi(ref_centroid, cur_centroid)
    psi_overall = max(psi_e, psi_c)
    status = "ok" if psi_overall < 0.10 else ("warn" if psi_overall < 0.20 else "alert")
    return {
        "psi_energy": psi_e,
        "psi_centroid": psi_c,
        "psi_overall": psi_overall,
        "status": status,
        "ref_energy": ref_energy,
        "cur_energy": cur_energy,
    }


def _psi_gauge(psi: float, status: str) -> go.Figure:
    color = {"ok": "#2ecc71", "warn": "#f39c12", "alert": "#e74c3c"}[status]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=psi,
        number={"font": {"color": color, "size": 28}, "valueformat": ".3f"},
        gauge={
            "axis": {
                "range": [0, 0.40],
                "tickvals": [0, 0.10, 0.20, 0.30, 0.40],
                "ticktext": ["0", "0.10", "0.20", "0.30", "0.40"],
                "tickcolor": "white",
                "tickfont": {"color": "white", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0.00, 0.10], "color": "rgba(46,204,113,0.15)"},
                {"range": [0.10, 0.20], "color": "rgba(243,156,18,0.15)"},
                {"range": [0.20, 0.40], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 3},
                "thickness": 0.75,
                "value": 0.20,
            },
        },
        title={"text": "PSI (overall)", "font": {"color": "white", "size": 13}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


_GRID = "rgba(128,128,128,0.2)"


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown("## MLOps Pipeline: Test Range → XENTA Fleet")
    st.markdown(
        "Weibel operates two distinct radar tiers. The high-precision instrumentation tier "
        "acts as a **label factory** for the mass-produced XENTA operational fleet. "
        "This tab shows how that data bridge works, how models are trained, validated, and "
        "deployed, and how the fleet stays calibrated as operating environments evolve."
    )

    # ── 1. Two-tier hardware architecture ─────────────────────────────────────
    st.divider()
    st.markdown("### 1 — Two-tier hardware architecture")

    col_t1, col_arrow, col_t2 = st.columns([10, 1, 10])

    with col_t1:
        st.markdown("#### 🎯 Tier 1 — Instrumentation Radar")
        st.markdown(
            "**MFCW / CW tracking radar** — deployed at missile test ranges globally "
            "(US Army, European ranges, Danish range). Provides **TSPI** (Time Space "
            "Position Information): real-time 3D position, velocity, acceleration, spin, "
            "and micro-motion, at sub-centimeter accuracy."
        )
        st.code(
            "TSPI output  (per target, per sweep)\n"
            "──────────────────────────────────────\n"
            "  t           UTC timestamp  (µs precision)\n"
            "  az          azimuth        ± 0.001 °\n"
            "  el          elevation      ± 0.001 °\n"
            "  r           slant range    ± 0.01 m\n"
            "  v_r         radial vel     ± 0.01 m/s\n"
            "  r_dot_dot   acceleration   m/s²\n"
            "  sigma_rcs   RCS estimate   dBsm\n"
            "  tracker_locked  bool",
            language="text",
        )
        st.success(
            "**This is the ground truth.** Every training label in the ML pipeline "
            "comes from the instrumentation radar — not from the XENTA unit under test. "
            "That separation is what makes instrument-error absorption possible."
        )

    with col_arrow:
        st.markdown("<div style='margin-top:120px;font-size:28px;text-align:center'>→</div>",
                    unsafe_allow_html=True)

    with col_t2:
        st.markdown("#### 📡 Tier 2 — XENTA-C Operational Radar")
        st.markdown(
            "**FMCW X-band** — mass-produced counter-UAS unit. "
            "500 M DKK order from Kongsberg / NATO, deliveries 2026–2027. "
            "Detects drones via micro-Doppler from spinning rotors. Each unit carries "
            "manufacturing tolerances: antenna phase errors, ADC non-linearity, "
            "oscillator phase noise, temperature-dependent gain drift."
        )
        st.code(
            "XENTA-C output  (per sweep)\n"
            "──────────────────────────────────────\n"
            "  PPI[az, r]   amplitude map  (180×64)\n"
            "               (all hardware imperfections included)\n"
            "\n"
            "Classical pipeline:\n"
            "  PPI → CFAR → KF → confirmed track\n"
            "\n"
            "ML pipeline (this project):\n"
            "  PPI → ConvGRU (ONNX) → P(target|cell) → KF → track",
            language="text",
        )
        st.info(
            "**Embedded constraint:** ONNX model must run in <10 ms per sweep "
            "on the ARM Cortex-A processor inside the XENTA signal processor — "
            "no GPU. ConvGRU at 5 694 parameters achieves <1 ms on CPU (p50)."
        )

    # ── 2. Data pipeline — DVC DAG ────────────────────────────────────────────
    st.divider()
    st.markdown("### 2 — Data pipeline")

    # Node layout: (x, y, stage_name, description, hex_color)
    _nodes = [
        (0.0, 2.0, "collect",       "Test range\ncapture session",       "#3498db"),
        (1.2, 2.0, "label_fusion",  "TSPI → XENTA\nframe projection",    "#9b59b6"),
        (2.4, 2.0, "quality_gate",  "Tracker lock\ncheck + filter",      "#e67e22"),
        (3.6, 2.0, "split",         "Train / val /\nheld-out split",      "#27ae60"),
        (4.8, 2.0, "train",         "DVC repro\n(ConvGRU / CNN / TF)",   "#2980b9"),
        (6.0, 2.0, "export",        "ONNX opset 17\n+ latency gate",     "#8e44ad"),
        (7.2, 2.0, "validate",      "Accuracy gate\n(Pd, AUC)",          "#16a085"),
        (8.4, 2.0, "register",      "MLflow registry\nStaging → Prod",   "#c0392b"),
        (8.4, 0.6, "deploy",        "OTA → XENTA\nfleet units",          "#e74c3c"),
        (4.8, 0.6, "monitor",       "PSI drift\nmonitor",                "#f39c12"),
    ]
    _edges_main     = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]
    _edges_feedback = [(8,9),(9,3)]  # deploy → monitor → retrigger split

    fig_dag = go.Figure()

    # Main pipeline edges
    for s, e in _edges_main:
        x0, y0 = _nodes[s][0], _nodes[s][1]
        x1, y1 = _nodes[e][0], _nodes[e][1]
        fig_dag.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=2.5,
            arrowcolor="#888", arrowsize=1.2,
        )

    # Feedback loop edges (orange)
    for s, e in _edges_feedback:
        x0, y0 = _nodes[s][0], _nodes[s][1]
        x1, y1 = _nodes[e][0], _nodes[e][1]
        fig_dag.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=2,
            arrowcolor="#f39c12", arrowsize=1.2,
        )

    # Nodes
    for x, y, name, label, color in _nodes:
        fig_dag.add_shape(
            type="rect",
            x0=x - 0.52, y0=y - 0.52, x1=x + 0.52, y1=y + 0.52,
            fillcolor=color + "22",
            line=dict(color=color, width=2),
        )
        fig_dag.add_annotation(
            x=x, y=y + 0.17,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(color=color, size=10),
            xref="x", yref="y",
        )
        fig_dag.add_annotation(
            x=x, y=y - 0.18,
            text=f"<span style='font-size:8px'>{label}</span>",
            showarrow=False,
            font=dict(color="#aaa", size=8),
            xref="x", yref="y",
        )

    # Legend annotation for feedback loop
    fig_dag.add_annotation(
        x=6.6, y=0.22,
        text="<span style='color:#f39c12'>── PSI alert triggers retraining ──▶</span>",
        showarrow=False, font=dict(size=10), xref="x", yref="y",
    )

    fig_dag.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.7, 9.1]),
        yaxis=dict(visible=False, range=[0.0, 2.7]),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_dag, use_container_width=True)

    with st.expander("Label fusion: how TSPI becomes a training label", expanded=True):
        col_lf1, col_lf2 = st.columns([5, 6])
        with col_lf1:
            st.markdown(
                "The instrumentation radar and the XENTA unit are co-located at the test range "
                "but measure the scene from slightly different positions and in different coordinate "
                "frames. Label fusion does three things:\n\n"
                "1. **Timestamp alignment** — match each XENTA sweep to the nearest TSPI frame "
                "(sub-millisecond tolerance; TSPI at µs precision is the master clock).\n\n"
                "2. **Coordinate transform** — project the TSPI 3D position (ECEF) into the XENTA's "
                "native polar grid (azimuth bin, range bin), accounting for the offset between the "
                "two antenna phase centres.\n\n"
                "3. **Quality gate** — discard any frame where the instrumentation tracker lost lock "
                "(`tracker_locked=False`) or where the target is outside the XENTA's coverage volume. "
                "Only high-confidence labels enter training."
            )
        with col_lf2:
            st.code(
                "# label_fusion.py\n"
                "def fuse(tspi_frame, xenta_ppi, xenta_origin_ecef, xenta_heading):\n"
                "\n"
                "    # 1. Timestamp alignment\n"
                "    dt_us = abs(tspi_frame.t_utc - xenta_ppi.t_utc)\n"
                "    if dt_us > 500:           # 500 µs max lag\n"
                "        return None\n"
                "\n"
                "    # 2. Quality gate\n"
                "    if not tspi_frame.tracker_locked:\n"
                "        return None\n"
                "\n"
                "    # 3. ECEF → XENTA polar frame\n"
                "    az_rad, r_m = ecef_to_polar(\n"
                "        tspi_frame.position_ecef,\n"
                "        xenta_origin_ecef,\n"
                "        xenta_heading,\n"
                "    )\n"
                "\n"
                "    return Label(\n"
                "        az_bin   = int(np.degrees(az_rad) / 2) % 180,  # 2° bins\n"
                "        r_bin    = int(r_m / 7.5),                     # 7.5 m bins\n"
                "        snr_db   = tspi_frame.sigma_rcs - noise_floor_db,\n"
                "        drone_id = session.drone_type,\n"
                "        unit_id  = xenta_ppi.serial_number,\n"
                "    )",
                language="python",
            )
        st.caption(
            "The `unit_id` field enables unit-specific fine-tuning: "
            "train a base model on pooled data from all test sessions, "
            "then fine-tune the final layer on data from that specific unit's serial number. "
            "The result is a personalised ONNX artifact per unit that compensates for "
            "that unit's individual hardware signature."
        )

    # ── 3. Instrument error absorption ────────────────────────────────────────
    st.divider()
    st.markdown("### 3 — Instrument error absorption")

    col_ie1, col_ie2 = st.columns([5, 6])
    with col_ie1:
        st.markdown("**The domain mismatch problem**")
        st.markdown(
            "A model trained on synthetic radar data learns the ideal physics — it has never "
            "seen the hardware distortions present in a real XENTA unit. When deployed, it faces "
            "a systematic gap between what the simulation predicts and what the hardware measures:"
        )
        st.markdown(
            "- **Antenna phase errors** → beam shape distorted from design spec\n"
            "- **ADC non-linearity** → amplitude compression at high SNR\n"
            "- **Oscillator phase noise** → Doppler lines broadened and shifted\n"
            "- **Temperature drift** → gain varies ±0.5 dB across operating range\n"
        )
        st.markdown("**How absorption works**")
        st.markdown(
            "Because the training **label** comes from the instrumentation radar (truth) but the "
            "training **feature** comes from the XENTA unit (imperfect), the model is forced to "
            "bridge that gap. It must learn: *given this distorted signal as input, predict where "
            "the target actually is.*\n\n"
            "This is physically equivalent to camera lens calibration — except it happens "
            "automatically during training, requires no explicit hardware characterisation, "
            "and generalises to targets the calibration drone did not fly."
        )
        st.success(
            "**The compounding return on hardware quality:** "
            "A higher-precision instrumentation radar produces more accurate TSPI labels. "
            "More accurate labels produce a better-calibrated ML model. "
            "Sensor accuracy at the hardware level compounds directly into model accuracy — "
            "the return on investing in Weibel's precision measurement technology."
        )

    with col_ie2:
        rng = np.random.default_rng(7)
        t = np.linspace(0, 1, 300)

        # Ground truth: micro-Doppler signature from drone rotors
        # Main rotor line (~12 Hz) + blade-passing harmonic (~85 Hz)
        clean = (
            np.sin(2 * np.pi * 12 * t) * np.exp(-4 * (t - 0.5) ** 2)
            + 0.3 * np.sin(2 * np.pi * 85 * t) * np.exp(-8 * (t - 0.3) ** 2)
        )

        # XENTA measurement: add hardware distortions
        phase_noise = np.cumsum(rng.normal(0, 0.018, 300))
        gain_drift  = 1.0 + 0.14 * np.sin(2 * np.pi * 2.5 * t)   # temperature cycle
        adc_nonlin  = clean ** 3 * 0.06                             # compression at peaks
        thermal_n   = rng.normal(0, 0.04, 300)
        distorted   = clean * gain_drift + adc_nonlin + phase_noise * 0.035 + thermal_n

        fig_ie = go.Figure()
        fig_ie.add_trace(go.Scatter(
            x=t, y=clean, mode="lines",
            name="True micro-Doppler (instrumentation TSPI label)",
            line=dict(color="#2ecc71", width=2.0),
        ))
        fig_ie.add_trace(go.Scatter(
            x=t, y=distorted, mode="lines",
            name="XENTA measurement (hardware distortions)",
            line=dict(color="#c084fc", width=1.5, dash="dot"),
        ))
        fig_ie.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Time (normalised)", gridcolor=_GRID, color="white"),
            yaxis=dict(title="Amplitude (normalised)", gridcolor=_GRID, color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10),
                        orientation="h", y=-0.25),
            margin=dict(t=10, b=60, l=60, r=10),
        )
        st.plotly_chart(fig_ie, use_container_width=True)
        st.caption(
            "Green = what physics predicts (instrumentation ground truth / training label). "
            "Purple = what the XENTA actually measures — phase noise, gain drift, and ADC "
            "non-linearity compound the distortion. The model is trained to map purple → green: "
            "hardware error absorbed into the learned weights."
        )

    # ── 4. MLflow experiment tracking ─────────────────────────────────────────
    st.divider()
    st.markdown("### 4 — MLflow experiment tracking")

    train_steps, train_loss = _read_metric("train_loss")
    val_steps,   val_f1     = _read_metric("val_f1")
    seq_steps,   seq_len    = _read_metric("seq_len")
    best_steps,  best_f1    = _read_metric("best_val_f1")

    best_f1_val = best_f1[-1] if best_f1 else 0.356
    n_params_val = _read_param("n_params") or "5 694"
    epochs_val   = _read_param("epochs") or "30"
    lr_val       = _read_param("lr") or "0.001"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model",       "ConvGRU")
    c2.metric("Parameters",  n_params_val)
    c3.metric("Best val F1", f"{best_f1_val:.4f}")
    c4.metric("Epochs",      epochs_val)
    c5.metric("LR",          lr_val)

    # Training curves with seq-len curriculum shading
    fig_ml = go.Figure()
    if train_steps:
        fig_ml.add_trace(go.Scatter(
            x=train_steps, y=train_loss, mode="lines",
            name="Train loss", yaxis="y1",
            line=dict(color="#e74c3c", width=2),
        ))
    if val_steps:
        fig_ml.add_trace(go.Scatter(
            x=val_steps, y=val_f1, mode="lines+markers",
            name="Val F1", yaxis="y2",
            line=dict(color="#c084fc", width=2.5),
            marker=dict(size=5),
        ))

    # Shade curriculum phases (seq_len 5 → 10 → 15)
    if seq_steps and seq_len:
        phases, cur_len, phase_start = [], seq_len[0], seq_steps[0]
        for i, (s, l) in enumerate(zip(seq_steps, seq_len)):
            if l != cur_len:
                phases.append((phase_start, seq_steps[i - 1], cur_len))
                cur_len, phase_start = l, s
        phases.append((phase_start, seq_steps[-1], cur_len))

        phase_colors = {
            5.0:  "rgba(52,152,219,0.07)",
            10.0: "rgba(155,89,182,0.07)",
            15.0: "rgba(46,204,113,0.07)",
        }
        for p_start, p_end, p_len in phases:
            fig_ml.add_vrect(
                x0=p_start, x1=p_end,
                fillcolor=phase_colors.get(p_len, "rgba(100,100,100,0.05)"),
                line_width=0,
                annotation_text=f"seq={int(p_len)}",
                annotation_font_color="#aaa",
                annotation_font_size=9,
                annotation_position="top left",
            )

    fig_ml.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Epoch", gridcolor=_GRID, zeroline=False, color="white"),
        yaxis=dict(
            title="Train loss", side="left", gridcolor=_GRID, zeroline=False,
            color="#e74c3c", titlefont=dict(color="#e74c3c"), tickfont=dict(color="#e74c3c"),
        ),
        yaxis2=dict(
            title="Val F1", overlaying="y", side="right", range=[0, 0.6],
            zeroline=False, color="#c084fc",
            titlefont=dict(color="#c084fc"), tickfont=dict(color="#c084fc"),
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(t=30, b=40, l=60, r=70),
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    st.caption(
        "Real ConvGRU training run from mlruns/. "
        "Shaded regions show the **sequence-length curriculum**: "
        "seq=5 (short windows, shallow temporal dependencies) → seq=10 → seq=15 "
        "(full deployment sequence length). "
        "Starting with short sequences prevents vanishing gradients in the GRU; "
        "extending progressively forces the model to learn longer-range temporal structure. "
        "Val F1 is noisy because the label structure (soft Gaussian position heatmaps) "
        "makes hard-threshold F1 sensitive to output calibration rather than raw detection quality."
    )

    st.markdown("**Model comparison across training runs (same dataset, matched Pfa)**")
    st.markdown("""
| Model | Params | Val F1 | Inference / sweep | Streaming | ONNX target |
|---|---:|---:|---:|:---:|---|
| ConvGRU ⭐ | 5 694 | 0.356 | <1 ms (CPU) | ✓ | ARM Cortex-A / embedded Linux |
| CNN (batch) | 19 073 | 0.307† | 0.8 ms (batch-10) | ✗ | ARM Cortex-A / Vitis AI FPGA |
| Transformer | 20 056 | 0.307 | ~2 ms (batch-10) | ✗ | ARM Cortex-A |
| CA-CFAR (classical) | — | — | 2.2 ms | ✓ | Any (no model) |

†CNN cell-level F1 on single sweep; 10-sweep batch AUC is meaningfully higher (see ROC tab).
    """)

    # ── 5. PSI drift monitoring ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 5 — PSI drift monitoring in deployment")
    st.markdown(
        "Each deployed XENTA unit periodically sends a sample of recent inference inputs "
        "to the on-premise monitoring backend. The PSI (Population Stability Index) compares "
        "the current signal distribution against the training reference. PSI > 0.20 triggers "
        "a retraining review — typically a new test-range session in the new environment "
        "or with the new drone type."
    )

    col_sliders, col_gauge, col_dist = st.columns([2, 2, 4])

    with col_sliders:
        clutter_shift = st.slider(
            "Clutter environment shift",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="pipe_clutter_shift",
            help="0 = training distribution · 1 = major environmental shift "
                 "(new sea state, coastal/inland change, heavy rain)"
        )
        drone_shift = st.slider(
            "New drone type fraction",
            min_value=0.0, max_value=0.5, value=0.0, step=0.05,
            key="pipe_drone_shift",
            help="Fraction of incoming detections from a drone type "
                 "not present in the training set (e.g. new FPV airframe)"
        )

    total_shift = clutter_shift * 0.85 + drone_shift * 0.65
    drift = _psi_from_shift(total_shift)
    status = drift["status"]

    with col_gauge:
        st.plotly_chart(_psi_gauge(drift["psi_overall"], status),
                        use_container_width=True)
        if status == "ok":
            st.success("✓ Stable — no action required")
        elif status == "warn":
            st.warning("⚠ Moderate drift — monitor closely")
        else:
            st.error("🚨 DRIFT ALERT — schedule test-range session")

    with col_dist:
        ref_e = drift["ref_energy"]
        cur_e = drift["cur_energy"]
        e_max = max(float(np.percentile(ref_e, 99)), float(np.percentile(cur_e, 99)))
        bins  = np.linspace(0, e_max, 42)
        bc    = (bins[:-1] + bins[1:]) / 2
        w     = bins[1] - bins[0]
        ref_h, _ = np.histogram(ref_e, bins=bins, density=True)
        cur_h, _ = np.histogram(cur_e, bins=bins, density=True)

        bar_color = {"ok": "#2ecc71", "warn": "#f39c12", "alert": "#e74c3c"}[status]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=bc, y=ref_h, name="Training reference",
            marker_color="#2980b9", opacity=0.55, width=w * 0.9,
        ))
        fig_dist.add_trace(go.Bar(
            x=bc, y=cur_h, name="Current deployment batch",
            marker_color=bar_color, opacity=0.55, width=w * 0.9,
        ))
        fig_dist.update_layout(
            height=210, barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Signal energy", gridcolor=_GRID, color="white"),
            yaxis=dict(title="Density", gridcolor=_GRID, color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10),
                        orientation="h", y=-0.35),
            margin=dict(t=10, b=50, l=50, r=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            f"PSI energy: **{drift['psi_energy']:.3f}** · "
            f"PSI spectral centroid: **{drift['psi_centroid']:.3f}** · "
            "thresholds: warn >0.10, alert >0.20"
        )

    with st.expander("Drift response playbook"):
        st.markdown("""
| PSI level | Likely cause | Recommended action |
|---|---|---|
| **<0.10** (stable) | Distribution matches training | No action |
| **0.10–0.20** (warn) | Seasonal sea-state change, new deployment area | Increase monitoring cadence; prepare test-range session |
| **>0.20** (alert) | New drone type, major clutter change, hardware fault | Schedule test-range collection; retrain; accuracy gate; promote |
| **>0.40** (critical) | Sensor hardware change or failure | Rollback to previous Production version; investigate hardware |
        """)
        st.code(
            "# Rollback to previous Production version\n"
            "python scripts/register_model.py \\\n"
            "    --run-id <previous-run-id> \\\n"
            "    --promote-to production\n"
            "# Restores the archived Production version in one command",
            language="bash",
        )

    # ── 6. Fleet deployment and continuous improvement ─────────────────────────
    st.divider()
    st.markdown("### 6 — Fleet deployment and the data flywheel")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**OTA deployment workflow**")
        st.code(
            "# Triggered by PSI alert from XENTA-C3-0142\n"
            "\n"
            "# Step 1 — Schedule test-range session\n"
            "range_ops.schedule(\n"
            "    unit_id      = 'XENTA-C3-0142',\n"
            "    drone_types  = ['DJI-Matrice4', 'FPV-250mm'],\n"
            "    environment  = 'coastal-high-sea-state',\n"
            ")\n"
            "\n"
            "# Step 2 — Pipeline re-runs on expanded dataset\n"
            "dvc repro              # all stages, cached where unchanged\n"
            "\n"
            "# Step 3 — Register and gate\n"
            "python scripts/register_model.py \\\n"
            "    --run-id <mlflow-run-id>   # blocked if val_acc < 0.95\n"
            "\n"
            "# Step 4 — Human review in MLflow UI, then promote\n"
            "python scripts/register_model.py \\\n"
            "    --run-id <mlflow-run-id> \\\n"
            "    --promote-to production\n"
            "\n"
            "# Step 5 — OTA push to PSI-flagged units\n"
            "fleet_deploy.push(\n"
            "    model        = 'radar-classifier:Production',\n"
            "    target_units = fleet.psi_alert_units,\n"
            "    rollback_on_fail = True,\n"
            ")",
            language="python",
        )

    with col_f2:
        st.markdown("**The data flywheel**")
        st.markdown(
            "Each deployment cycle makes the next one better:\n\n"
            "1. **More XENTA units deployed** → more diverse operating environments encountered\n"
            "2. **PSI alerts** identify gaps in the training distribution automatically\n"
            "3. **Test-range sessions** fill those gaps with labelled data\n"
            "4. **Retraining on the expanded dataset** lowers the detection floor further\n"
            "5. **Better model → higher Pd, fewer false alarms** → more units deployed\n\n"
            "This compounding effect is unavailable to classical signal processing: "
            "CFAR's operating parameters are fixed at algorithm design time."
        )
        st.markdown("**Air-gapped operation**")
        st.markdown(
            "For classified or NATO-restricted deployments, the entire pipeline runs on-premise: "
            "**self-hosted Gitea** for version control and CI, **local MLflow** with SQLite backend, "
            "**on-prem DVC remote** (NFS or S3-compatible MinIO). "
            "No training data, model weights, or PSI statistics leave the network boundary. "
            "OTA delivery uses authenticated HTTPS to the unit's local update endpoint — "
            "the same physical security boundary as the radar itself."
        )
        st.markdown("**Per-unit calibration**")
        st.markdown(
            "Because `unit_id` is a training metadata field, the pipeline supports two deployment modes:\n\n"
            "- **Fleet model**: single ONNX artifact trained on pooled data from all units — "
            "lowest cost, good for homogeneous environments\n"
            "- **Unit-personalised model**: base model + fine-tuned final layer on that unit's "
            "serial-specific test-range data — compensates for individual hardware variance, "
            "best performance in demanding SNR conditions"
        )

    st.success(
        "**The complete loop:** Instrumentation radar provides precision labels → XENTA provides "
        "hardware-distorted signals → model learns to correct for hardware signature → "
        "ONNX deploys to fleet → PSI monitors distribution drift → drift triggers test-range collection → "
        "pipeline retrains → improved model redeploys. "
        "Each cycle, the detection floor lowers and the false-alarm rate drops."
    )

    st.info(
        "**Next tab →** Agentic MLOps: a Claude-powered agent that interprets PSI drift reports, "
        "queries MLflow experiment history, and recommends a retraining action — "
        "demonstrating agentic AI integrated with the pipeline described here."
    )
