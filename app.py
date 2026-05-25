"""
Streamlit demo — Range-Doppler Radar Target Classifier.

Multi-scan temporal pipeline: a CNN+LSTM model classifies sequences of 8
consecutive Range-Doppler maps and decides whether a target track is present.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from src.baseline.kalman_cfar import KalmanCFARDetector
from src.data.generator import N_DOPPLER, N_RANGE, T, generate_dataset

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Radar AI Classifier",
    page_icon="📡",
    layout="wide",
)

TARGET_COLOR = "#4C9BE8"
TARGET_FILL  = "rgba(76, 155, 232, 0.15)"
CLUTTER_COLOR = "#E8924C"
CLUTTER_FILL  = "rgba(232, 146, 76, 0.15)"
GRID = "rgba(128,128,128,0.15)"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> ort.InferenceSession | None:
    path = Path("artifacts/model.onnx")
    return ort.InferenceSession(str(path)) if path.exists() else None


@st.cache_data(show_spinner="Computing accuracy across SNR range…")
def snr_benchmark(
    snr_min: float = -10.0,
    snr_max: float = 25.0,
    n_steps: int = 24,
    n_per_class: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = ort.InferenceSession("artifacts/model.onnx")
    detector = KalmanCFARDetector()
    snr_values = np.linspace(snr_min, snr_max, n_steps)
    ml_accs, cfar_accs = [], []

    for snr_db in snr_values:
        X, y = generate_dataset(n_per_class * 2, float(snr_db), seed=99)
        # ML inference: (n, T, 64, 128) → add channel → (n, T, 1, 64, 128)
        X_onnx = X[:, :, np.newaxis, :, :].astype(np.float32)
        probs = session.run(None, {"sequence": X_onnx})[0].squeeze(1)
        preds_ml = (probs >= 0.5).astype(int)
        ml_accs.append(float((preds_ml == y).mean()))

        preds_kf = detector.detect_batch(X).astype(int)
        cfar_accs.append(float((preds_kf == y).mean()))

    return snr_values, np.array(ml_accs), np.array(cfar_accs)


@st.cache_data(show_spinner="Computing ROC data…")
def roc_data(snr_db: float, n_per_class: int = 150) -> dict:
    session = ort.InferenceSession("artifacts/model.onnx")
    detector = KalmanCFARDetector()
    X, y = generate_dataset(n_per_class * 2, snr_db, seed=77)

    X_onnx = X[:, :, np.newaxis, :, :].astype(np.float32)
    ml_scores = session.run(None, {"sequence": X_onnx})[0].squeeze(1)
    classical_scores = detector.score_batch(X)

    return {"y": y, "ml_scores": ml_scores, "classical_scores": classical_scores}


def compute_roc(
    scores: np.ndarray, labels: np.ndarray, n_points: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.concatenate([[hi + 1], np.linspace(hi, lo, n_points), [lo - 1]])
    fprs, tprs = [], []
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    for t in thresholds:
        preds = scores >= t
        tprs.append(float((preds & (labels == 1)).sum()) / pos if pos else 0.0)
        fprs.append(float((preds & (labels == 0)).sum()) / neg if neg else 0.0)
    return np.array(fprs), np.array(tprs), thresholds


def trapz_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    x, y = fpr[order], tpr[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def plot_layout(height: int = 300) -> dict:
    return dict(
        height=height,
        margin=dict(t=40, b=30, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def classify_sequence(
    session: ort.InferenceSession, sequence: np.ndarray
) -> float:
    """Return P(target present) for a single sequence (T, 64, 128)."""
    x = sequence[np.newaxis, :, np.newaxis, :, :].astype(np.float32)  # (1,T,1,64,128)
    return float(session.run(None, {"sequence": x})[0][0, 0])


def _animated_rdmap(
    sequence: np.ndarray,
    track_positions: list | None = None,
    title: str = "Range-Doppler Sequence",
) -> go.Figure:
    """Plotly animated heatmap cycling through T scans with optional track overlay."""
    z_max = float(np.percentile(np.abs(sequence), 98))
    z_max = max(z_max, 1e-3)

    frames = []
    for t in range(sequence.shape[0]):
        data: list[go.BaseTraceType] = [go.Heatmap(
            z=sequence[t],
            colorscale="RdBu",
            zmin=-z_max, zmax=z_max,
            showscale=True,
            colorbar=dict(thickness=12, len=0.7, title="Amplitude"),
        )]
        if track_positions and track_positions[t] is not None:
            r_pos, d_pos = track_positions[t]
            data.append(go.Scatter(
                x=[d_pos], y=[r_pos],
                mode="markers",
                name="KF track",
                marker=dict(
                    color="yellow", size=14, symbol="cross",
                    line=dict(color="black", width=1.5),
                ),
            ))
        frames.append(go.Frame(
            data=data,
            name=str(t),
            layout=go.Layout(title_text=f"{title} — scan {t + 1}/{T}"),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"{title} — scan 1/{T}", font=dict(size=14)),
            height=380,
            margin=dict(t=50, b=60, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Doppler bin", gridcolor=GRID, zeroline=False),
            yaxis=dict(title="Range bin", gridcolor=GRID, zeroline=False),
            showlegend=bool(track_positions),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=-0.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"fromcurrent": True,
                                      "frame": {"duration": 350, "redraw": True},
                                      "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"mode": "immediate",
                                        "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[dict(
                currentvalue=dict(prefix="Scan ", font=dict(size=12)),
                pad=dict(b=10, t=0),
                steps=[dict(
                    args=[[f.name], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}],
                    label=str(int(f.name) + 1),
                    method="animate",
                ) for f in frames],
            )],
        ),
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────

session = load_model()

st.title("📡 Radar Target Classifier — Multi-Scan Range-Doppler")
st.markdown(
    "An automated pipeline that trains, validates, and deploys an AI model that separates "
    "**radar targets** (aircraft, ships, projectiles) from **background clutter** — by analysing "
    "**sequences of 8 Range-Doppler maps** and detecting whether a coherent moving track is present."
)

if session is None:
    st.error(
        "Model not found at `artifacts/model.onnx`. "
        "Run `python src/train.py && python scripts/export_onnx_rd.py` first."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scans per decision", f"T = {T}", "temporal context")
m2.metric("Map resolution", f"{N_RANGE}×{N_DOPPLER}", "range × Doppler bins")
m3.metric("Validation accuracy", "≥ 80%", "CI pipeline gate")
m4.metric("Architecture", "CNN+LSTM", "shared encoder + sequence model")

st.divider()

tab_demo, tab_signals, tab_perf, tab_roc, tab_pipeline = st.tabs([
    "🎯  Live Classifier",
    "📊  Signal Explorer",
    "📈  Detection Performance",
    "📉  ROC Curve",
    "⚙️  How the Pipeline Works",
])


# ── Tab 1: Live Classifier ─────────────────────────────────────────────────────

with tab_demo:
    with st.expander("What is a scan sequence?"):
        st.markdown(
            "A **Range-Doppler map** is a 2D image produced by a radar after processing one "
            "batch of received pulses (one *coherent processing interval*, or CPI). "
            "The horizontal axis is **Doppler frequency** — proportional to the target's "
            "radial velocity. The vertical axis is **range** — distance from the radar.\n\n"
            "A single map may contain a faint target blob buried in noise. "
            "By analysing a **sequence of T=8 consecutive maps**, the model can "
            "exploit the fact that a real target *drifts consistently* in range as it moves, "
            "while clutter and noise do not. The LSTM part of the model learns this "
            "temporal consistency pattern across scans.\n\n"
            "| | Single map | Sequence of 8 |\n"
            "|---|---|---|\n"
            "| Target blob | Weak, noisy | Drifts predictably in range |\n"
            "| Clutter | Spatially correlated | No consistent drift |\n"
            "| Model confidence | Low at marginal SNR | Much higher — uses motion |"
        )

    col_ctrl, col_out = st.columns([1, 2], gap="large")

    with col_ctrl:
        signal_choice = st.radio(
            "Sequence type",
            ["Target", "Clutter", "🎲 Random"],
            help=(
                "**Target** — moving object with coherent Doppler return drifting in range.\n\n"
                "**Clutter** — slow-moving background (ground, sea surface)."
            ),
        )
        snr = st.slider(
            "Signal-to-Noise Ratio (dB)",
            min_value=-10.0, max_value=25.0, value=10.0, step=0.5,
        )
        snr_label = (
            "very noisy — near detection limit" if snr < 0
            else "low — model must work hard" if snr < 5
            else "moderate" if snr < 12
            else "clean — strong return"
        )
        st.caption(f"SNR {snr:+.0f} dB — {snr_label}")
        with st.expander("What is dB?"):
            st.markdown(
                "**dB (decibel)** is a logarithmic ratio. For SNR: how much stronger is the "
                "signal than background noise?\n\n"
                "| dB | Signal vs noise |\n|---|---|\n"
                "| −10 dB | Signal is **10× weaker** |\n"
                "| 0 dB | Equal strength |\n"
                "| 10 dB | Signal is **10× stronger** |\n"
                "| 20 dB | Signal is **100× stronger** |"
            )
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42,
                               help="Fix this to reproduce the same sequence.")

    rng = np.random.default_rng(int(seed))
    if signal_choice == "Target":
        true_label = "Target"
    elif signal_choice == "Clutter":
        true_label = "Clutter"
    else:
        true_label = "Target" if rng.random() > 0.5 else "Clutter"

    # Generate a small batch and pick a sample with the right label
    label_int = 1 if true_label == "Target" else 0
    X_seq, y_seq = generate_dataset(10, snr, seed=int(seed))
    match_idx = np.where(y_seq == label_int)[0]
    # Fallback: generate fresh if not found (very rare with 10 samples)
    if len(match_idx) == 0:
        from src.data.generator import _clutter_sequence, _target_sequence
        fn = _target_sequence if true_label == "Target" else _clutter_sequence
        sequence = fn(rng, snr)
    else:
        sequence = X_seq[match_idx[0]]  # (T, 64, 128)

    prob = classify_sequence(session, sequence)
    pred_label = "Target" if prob >= 0.5 else "Clutter"
    correct = pred_label == true_label

    detector = KalmanCFARDetector()
    track_pos = detector.track_positions(sequence)
    kf_detected = detector.detect_sequence(sequence)
    kf_label = "Target" if kf_detected else "Clutter"
    kf_correct = kf_label == true_label
    kf_score = detector.score_sequence(sequence)

    with col_out:
        border = "#2ecc71" if correct else "#e74c3c"
        icon = "✓" if correct else "✗"
        emoji = "🎯" if pred_label == "Target" else "🌊"
        st.markdown(
            f"""
            <div style="border:2px solid {border}; border-radius:12px;
                        padding:20px; text-align:center; margin-bottom:16px;">
                <div style="font-size:40px">{emoji}</div>
                <h2 style="margin:8px 0; color:{border}">{pred_label} {icon}</h2>
                <p style="font-size:20px; margin:0">P(target) = {prob:.1%}</p>
                <p style="color:#888; font-size:13px; margin:6px 0">
                    True label: <strong>{true_label}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        kf_color = "#2ecc71" if kf_correct else "#e74c3c"
        kf_icon = "✓" if kf_correct else "✗"
        hits_str = f"{int(kf_score * T)}/{T} scans tracked"
        st.markdown(
            f"""
            <div style="border:1px solid #444; border-radius:8px;
                        padding:12px 16px; margin-top:8px;">
                <span style="font-size:11px; color:#666; text-transform:uppercase;
                             letter-spacing:0.05em">Classical baseline (KF + CFAR)</span>
                <div style="font-size:16px; font-weight:bold; color:{kf_color};
                            margin-top:4px">{kf_label} {kf_icon}</div>
                <span style="font-size:12px; color:#888">{hits_str}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Track overlay: show range positions across scans
    has_track = any(p is not None for p in track_pos)
    track_ranges = [p[0] if p is not None else None for p in track_pos]

    fig_anim = _animated_rdmap(
        sequence,
        track_positions=track_pos if has_track else None,
        title=f"{true_label} sequence",
    )
    st.plotly_chart(fig_anim, use_container_width=True)

    if has_track and true_label == "Target":
        r_vals = [r for r in track_ranges if r is not None]
        scan_ids = [i for i, r in enumerate(track_ranges) if r is not None]
        fig_track = go.Figure(go.Scatter(
            x=[s + 1 for s in scan_ids], y=r_vals,
            mode="lines+markers",
            line=dict(color="yellow", width=2),
            marker=dict(size=8, color="yellow", symbol="cross"),
            name="Track range",
        ))
        fig_track.update_layout(
            **plot_layout(180),
            xaxis=dict(title="Scan number", gridcolor=GRID, zeroline=False,
                       tickmode="array", tickvals=list(range(1, T + 1))),
            yaxis=dict(title="Range bin", gridcolor=GRID, zeroline=False),
            title="KF track — range position across scans",
        )
        st.plotly_chart(fig_track, use_container_width=True)

    if pred_label == "Target":
        st.info(
            "**Why target?** A compact bright blob drifts consistently in range across the "
            "8 scans — the signature of a moving object at constant velocity. "
            "The LSTM learns to recognise this temporal drift pattern even at low SNR."
        )
    else:
        st.info(
            "**Why clutter?** Energy is diffuse and concentrated near range=0, Doppler=0 "
            "with no consistent drift across scans — the spatial pattern of correlated "
            "ground or sea returns rather than a moving point target."
        )


# ── Tab 2: Signal Explorer ─────────────────────────────────────────────────────

with tab_signals:
    st.markdown(
        "Compare a target sequence and a clutter sequence side by side. "
        "Select a scan number to see the Range-Doppler maps at that instant."
    )
    st.write("")

    snr_exp = st.slider(
        "Signal-to-Noise Ratio (dB)", min_value=-10.0, max_value=25.0,
        value=10.0, step=1.0, key="snr_explorer",
    )
    seed_exp = st.number_input("Seed", min_value=0, max_value=9999, value=7, key="seed_exp")
    scan_sel = st.slider("Scan number", min_value=1, max_value=T, value=1, key="scan_sel")

    X_exp, y_exp = generate_dataset(20, snr_exp, seed=int(seed_exp))
    tgt_seq = X_exp[y_exp == 1][0] if (y_exp == 1).any() else X_exp[0]
    clt_seq = X_exp[y_exp == 0][0] if (y_exp == 0).any() else X_exp[-1]

    scan_t = tgt_seq[scan_sel - 1]
    scan_c = clt_seq[scan_sel - 1]
    z_max = float(np.percentile(np.abs(np.concatenate([scan_t, scan_c])), 98))
    z_max = max(z_max, 1e-3)

    col_t, col_c = st.columns(2)
    with col_t:
        st.markdown(f"**🎯 Target — scan {scan_sel}/{T}**")
        fig_t = go.Figure(go.Heatmap(
            z=scan_t, colorscale="RdBu", zmin=-z_max, zmax=z_max, showscale=False,
        ))
        fig_t.update_layout(**plot_layout(280),
                            xaxis=dict(title="Doppler bin", gridcolor=GRID),
                            yaxis=dict(title="Range bin", gridcolor=GRID))
        st.plotly_chart(fig_t, use_container_width=True)
    with col_c:
        st.markdown(f"**🌊 Clutter — scan {scan_sel}/{T}**")
        fig_c = go.Figure(go.Heatmap(
            z=scan_c, colorscale="RdBu", zmin=-z_max, zmax=z_max, showscale=False,
        ))
        fig_c.update_layout(**plot_layout(280),
                            xaxis=dict(title="Doppler bin", gridcolor=GRID),
                            yaxis=dict(title="Range bin", gridcolor=GRID))
        st.plotly_chart(fig_c, use_container_width=True)

    st.success(
        "**Target:** A bright compact blob at a Doppler bin ≥ 5 that drifts in range "
        "as you move the scan slider. The Doppler offset means the object is moving."
    )
    st.warning(
        "**Clutter:** Energy concentrated near Doppler=0 (stationary/slow returns) "
        "and near range=0, with no coherent drift across scans."
    )


# ── Tab 3: Detection Performance ──────────────────────────────────────────────

with tab_perf:
    st.markdown(
        "CNN+LSTM classifier vs the classical Kalman filter + CA-CFAR baseline. "
        "Accuracy measured on 120 test sequences per SNR point."
    )
    st.write("")

    snr_vals, ml_accs, cfar_accs = snr_benchmark()

    fig3 = go.Figure()
    fig3.add_vrect(x0=-10, x1=0, fillcolor="rgba(231,76,60,0.07)",
                   line_width=0, annotation_text="Very low SNR", annotation_position="top left")
    fig3.add_vrect(x0=0, x1=8, fillcolor="rgba(241,196,15,0.07)",
                   line_width=0, annotation_text="Marginal", annotation_position="top left")
    fig3.add_vrect(x0=8, x1=25, fillcolor="rgba(46,204,113,0.05)",
                   line_width=0, annotation_text="Operational range", annotation_position="top left")
    fig3.add_trace(go.Scatter(
        x=snr_vals, y=ml_accs * 100, mode="lines+markers",
        name="CNN+LSTM", line=dict(color=TARGET_COLOR, width=2.5), marker=dict(size=5),
        hovertemplate="SNR %{x:.1f} dB<br>ML accuracy %{y:.1f}%<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=snr_vals, y=cfar_accs * 100, mode="lines+markers",
        name="KF + CFAR baseline", line=dict(color=CLUTTER_COLOR, width=2.5, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="SNR %{x:.1f} dB<br>Classical accuracy %{y:.1f}%<extra></extra>",
    ))
    fig3.add_hline(y=80, line_dash="dash", line_color="#e74c3c",
                   annotation_text="80% CI gate", annotation_position="bottom right")
    fig3.add_hline(y=50, line_dash="dot", line_color="#666",
                   annotation_text="Chance level", annotation_position="bottom right")
    fig3.update_layout(
        **{**plot_layout(420), "showlegend": True},
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Signal-to-Noise Ratio (dB)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
    )
    fig3.update_xaxes(gridcolor=GRID, zeroline=False)
    fig3.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**Why the CNN+LSTM dominates at low SNR:** the Kalman tracker requires CFAR "
        "to produce a detection before it can initiate a track — at low SNR the target "
        "blob falls below the CFAR threshold and the tracker never starts. "
        "The ML model learns the full 2D spatial and temporal pattern jointly, "
        "finding sub-threshold signatures the CFAR misses."
    )
    st.info(
        "**Classical baseline:** CA-CFAR per range row detects local peaks above the "
        "surrounding noise level. Detected peaks are associated to a constant-velocity "
        "Kalman track using Mahalanobis gating. A sequence is declared target-present "
        "if the track is maintained across ≥ 4 of 8 scans."
    )


# ── Tab 4: ROC Curve ──────────────────────────────────────────────────────────

with tab_roc:
    st.markdown(
        "The ROC curve shows the full trade-off between **detection probability (Pd)** "
        "and **false alarm rate (Pfa)** as the decision threshold varies. "
        "AUC = 1.0 is perfect, 0.5 is a coin flip."
    )
    st.write("")

    col_ctrl_roc, col_plot_roc = st.columns([1, 2], gap="large")

    with col_ctrl_roc:
        snr_roc = st.slider(
            "Signal-to-Noise Ratio (dB)",
            min_value=-10.0, max_value=25.0, value=5.0, step=1.0,
            key="snr_roc",
        )
        ml_thr = st.slider(
            "ML threshold (P(target))",
            min_value=0.01, max_value=0.99, value=0.50, step=0.01,
            key="ml_thr",
            help="Slide to move the ML operating point along its ROC curve.",
        )
        kf_thr = st.slider(
            "KF threshold (hit fraction)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.125,
            key="kf_thr",
            help="Hit fraction = tracked_scans / T. 0.5 = ≥4/8 scans (default).",
        )

    data = roc_data(snr_roc)
    y_roc = data["y"]
    ml_fpr, ml_tpr, ml_thrs = compute_roc(data["ml_scores"], y_roc)
    cl_fpr, cl_tpr, cl_thrs = compute_roc(data["classical_scores"], y_roc)
    ml_auc = trapz_auc(ml_fpr, ml_tpr)
    cl_auc = trapz_auc(cl_fpr, cl_tpr)

    ml_op_idx = int(np.argmin(np.abs(ml_thrs - ml_thr)))
    cl_op_idx = int(np.argmin(np.abs(cl_thrs - kf_thr)))

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Chance (AUC 0.500)",
        line=dict(color="#555", width=1, dash="dot"),
    ))
    fig_roc.add_trace(go.Scatter(
        x=ml_fpr, y=ml_tpr, mode="lines",
        name=f"CNN+LSTM  (AUC {ml_auc:.3f})",
        line=dict(color=TARGET_COLOR, width=2.5),
        hovertemplate="Pfa %{x:.3f}<br>Pd %{y:.3f}<extra>ML</extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=cl_fpr, y=cl_tpr, mode="lines",
        name=f"KF + CFAR  (AUC {cl_auc:.3f})",
        line=dict(color=CLUTTER_COLOR, width=2.5, dash="dash"),
        hovertemplate="Pfa %{x:.3f}<br>Pd %{y:.3f}<extra>KF+CFAR</extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[float(ml_fpr[ml_op_idx])], y=[float(ml_tpr[ml_op_idx])],
        mode="markers", name=f"ML op. point (thr {ml_thr:.2f})",
        marker=dict(color=TARGET_COLOR, size=13, symbol="circle",
                    line=dict(color="white", width=2)),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[float(cl_fpr[cl_op_idx])], y=[float(cl_tpr[cl_op_idx])],
        mode="markers", name=f"KF op. point (thr {kf_thr:.3f})",
        marker=dict(color=CLUTTER_COLOR, size=13, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))
    fig_roc.update_layout(
        **{**plot_layout(440), "showlegend": True},
        xaxis_title="False Alarm Rate (Pfa)",
        yaxis_title="Detection Probability (Pd)",
        xaxis=dict(range=[-0.02, 1.02], gridcolor=GRID, zeroline=False),
        yaxis=dict(range=[-0.02, 1.04], gridcolor=GRID, zeroline=False),
        legend=dict(x=0.40, y=0.10, bgcolor="rgba(0,0,0,0)"),
    )

    with col_plot_roc:
        st.plotly_chart(fig_roc, use_container_width=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ML AUC", f"{ml_auc:.3f}")
    r2.metric("KF+CFAR AUC", f"{cl_auc:.3f}")
    r3.metric("ML Pd", f"{float(ml_tpr[ml_op_idx]):.1%}")
    r4.metric("ML Pfa", f"{float(ml_fpr[ml_op_idx]):.1%}")

    st.info(
        "**KF score = hit fraction:** the continuous score for the classical baseline "
        "is the fraction of T scans where a detection was associated to the track "
        "(e.g., 6/8 = 0.75). This gives a coarse-grained ROC curve with discrete steps "
        "at multiples of 1/T, in contrast to the ML model's smooth probability output."
    )


# ── Tab 5: Pipeline ────────────────────────────────────────────────────────────

with tab_pipeline:
    st.markdown(
        "The classifier is one component in an automated model factory. "
        "Every stage is versioned and reproducible — from the raw simulation parameters "
        "to the ONNX file running on the radar signal processor."
    )
    st.write("")

    steps = [
        ("1", "📡", "Synthetic Data Generation",
         "Generates thousands of labelled Range-Doppler sequences (T=8 maps each) with "
         "controllable SNR and Doppler structure. Every dataset is pinned to an exact seed — "
         "anyone can reproduce the exact same sequences from the same parameters.",
         "Python · NumPy · DVC"),
        ("2", "🔬", "Automated Testing",
         "Before training starts, unit tests verify data shapes (n, 8, 64, 128), model "
         "forward-pass contracts, ONNX export correctness, and KF tracker Pd at 10 dB SNR. "
         "A broken pipeline fails here, not in production.",
         "pytest · ruff"),
        ("3", "🧠", "GPU Training & Experiment Tracking",
         "The CNN+LSTM model trains on sequences of 8 Range-Doppler maps. The shared CNN "
         "encodes each scan; the LSTM integrates track evidence across scans. Every run is "
         "logged with T, architecture, and SNR as MLflow params.",
         "PyTorch · MLflow · Kubernetes"),
        ("4", "✅", "Quality Gate",
         "If the model does not reach 80% validation accuracy, the pipeline fails "
         "automatically. No human needs to check — a bad model cannot be promoted "
         "to the next stage.",
         "CI/CD gate · sys.exit(1)"),
        ("5", "📦", "ONNX Export & Registry",
         "The approved model is exported to ONNX with a fixed T=8 sequence length — "
         "the LSTM Python for-loop is unrolled into T replicated op-graphs so no "
         "special LSTM runtime is needed. Accepted by FPGA toolchains and embedded Linux.",
         "ONNX · MLflow Registry"),
        ("6", "🚀", "Edge Deployment",
         "The ONNX file deploys to the radar signal processor. One decision per 8 CPIs, "
         "no ML framework at runtime. The ONNX format is directly accepted by Xilinx "
         "Vitis AI for FPGA compilation; that path is defined but outside this demonstrator.",
         "ONNX Runtime · Xilinx Vitis AI (path defined)"),
    ]

    for row_start in range(0, len(steps), 3):
        cols = st.columns(3, gap="medium")
        for j, col in enumerate(cols):
            if row_start + j >= len(steps):
                break
            num, icon, title, desc, tech = steps[row_start + j]
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid #333; border-radius:10px; padding:20px;
                                min-height:200px; display:flex; flex-direction:column;">
                        <div style="font-size:26px">{icon}</div>
                        <div style="font-size:11px; color:#666; margin:4px 0 2px">STEP {num}</div>
                        <div style="font-size:15px; font-weight:bold">{title}</div>
                        <div style="font-size:13px; color:#bbb; flex:1">{desc}</div>
                        <div style="font-size:11px; color:#555; font-family:monospace;
                                    margin-top:12px; padding-top:8px; border-top:1px solid #222">
                            {tech}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")

    st.divider()
    st.subheader("Why this matters for defense environments")
    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        st.markdown("**🔒 Air-gapped operation**")
        st.write(
            "The entire pipeline — Git server, CI runner, experiment tracking, "
            "container registry — runs on sovereign infrastructure. "
            "No training data, model weights, or telemetry leaves the network."
        )
    with d2:
        st.markdown("**📋 Full auditability**")
        st.write(
            "Every deployed model traces back to its exact training data, code version, "
            "and simulation parameters. The lineage is stored in DVC lock files "
            "and MLflow run records — both cryptographically verifiable."
        )
    with d3:
        st.markdown("**♻️ Reproducibility**")
        st.write(
            "Given any model version, the exact training run can be reconstructed: "
            "same data, same code, same parameters, bit-identical output. "
            "Critical for incident investigation and safety certification."
        )
