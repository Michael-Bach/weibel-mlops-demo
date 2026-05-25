"""
Streamlit demo — Radar Signal Classifier.

Target audience: non-ML recruiters and Weibel radar/defense personnel.
Focuses on what the system does and why it makes sense for radar,
not on internal ML mechanics.

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
from src.baseline.cfar import MTIThresholdDetector
from src.data.generate import generate_clutter, generate_target

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Radar AI Classifier",
    page_icon="📡",
    layout="wide",
)

SIGNAL_LENGTH = 128

TARGET_COLOR = "#4C9BE8"
TARGET_FILL = "rgba(76, 155, 232, 0.15)"
CLUTTER_COLOR = "#E8924C"
CLUTTER_FILL = "rgba(232, 146, 76, 0.15)"
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
    n_steps: int = 36,
    n_per_class: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = ort.InferenceSession("artifacts/model.onnx")
    cfar = MTIThresholdDetector()
    rng = np.random.default_rng(99)
    snr_values = np.linspace(snr_min, snr_max, n_steps)
    ml_accs, cfar_accs = [], []
    for snr_db in snr_values:
        targets = np.stack(
            [generate_target(SIGNAL_LENGTH, snr_db, rng) for _ in range(n_per_class)]
        )
        clutters = np.stack(
            [generate_clutter(SIGNAL_LENGTH, snr_db, rng) for _ in range(n_per_class)]
        )
        X = np.concatenate([targets, clutters]).astype(np.float32)
        y = np.array([1] * n_per_class + [0] * n_per_class)

        preds_ml = session.run(None, {"signal": X})[0].argmax(axis=1)
        ml_accs.append(float((preds_ml == y).mean()))

        spectra = np.stack([
            np.abs(np.fft.rfft(x * np.hanning(SIGNAL_LENGTH))) for x in X
        ])
        preds_cfar = cfar.detect_batch(spectra).astype(int)
        cfar_accs.append(float((preds_cfar == y).mean()))

    return snr_values, np.array(ml_accs), np.array(cfar_accs)


# ── Signal helpers ────────────────────────────────────────────────────────────

def classify(
    session: ort.InferenceSession, signal: np.ndarray
) -> tuple[str, float, np.ndarray]:
    x = signal[np.newaxis, :].astype(np.float32)
    logits = session.run(None, {"signal": x})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    pred = int(np.argmax(logits))
    return ("Target" if pred == 1 else "Clutter"), float(probs[pred]), probs


def windowed_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hanning-windowed magnitude spectrum — matches the model's internal FFT."""
    mag = np.abs(np.fft.rfft(signal * np.hanning(SIGNAL_LENGTH)))
    freqs = np.fft.rfftfreq(SIGNAL_LENGTH)
    return freqs, mag


def plot_layout(height: int = 300) -> dict:
    return dict(
        height=height,
        margin=dict(t=40, b=30, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )


# ── Header ────────────────────────────────────────────────────────────────────

session = load_model()

st.title("📡 Radar Signal Classifier")
st.markdown(
    "An automated pipeline that trains, validates, and deploys an AI model that separates "
    "**radar targets** (aircraft, ships, projectiles) from **background clutter** "
    "(ground returns, sea surface, weather) — from raw sensor data to edge deployment."
)

if session is None:
    st.error(
        "Model not found at `artifacts/model.onnx`. "
        "Run `python scripts/export_onnx.py` first."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Inference latency", "0.035 ms", "p50, host CPU")
m2.metric("Model file size", "67 KB", "self-contained ONNX")
m3.metric("Validation accuracy", "≥ 80%", "CI pipeline gate")
m4.metric("Parameters", "16,838", "weights learned from data")

st.divider()

tab_demo, tab_signals, tab_perf, tab_pipeline = st.tabs([
    "🎯  Live Classifier",
    "📊  Signal Explorer",
    "📈  Detection Performance",
    "⚙️  How the Pipeline Works",
])


# ── Tab 1: Live Classifier ─────────────────────────────────────────────────────

with tab_demo:
    st.markdown(
        "Select a signal type and noise level. The model classifies it in real time "
        "using the same ONNX file that would be deployed to the radar signal processor."
    )
    st.write("")

    col_ctrl, col_out = st.columns([1, 2], gap="large")

    with col_ctrl:
        signal_choice = st.radio(
            "Signal type",
            ["Target", "Clutter", "🎲 Random"],
            help=(
                "**Target** — moving object with coherent Doppler return.\n\n"
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
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42,
                               help="Fix this to reproduce the same signal.")

    rng = np.random.default_rng(int(seed))
    if signal_choice == "Target":
        sig = generate_target(SIGNAL_LENGTH, snr, rng)
        true_label = "Target"
    elif signal_choice == "Clutter":
        sig = generate_clutter(SIGNAL_LENGTH, snr, rng)
        true_label = "Clutter"
    else:
        true_label = "Target" if rng.random() > 0.5 else "Clutter"
        gen_fn = generate_target if true_label == "Target" else generate_clutter
        sig = gen_fn(SIGNAL_LENGTH, snr, rng)

    pred_label, confidence, probs = classify(session, sig)
    correct = pred_label == true_label

    freqs_arr, spec_arr = windowed_spectrum(sig)
    cfar_detector = MTIThresholdDetector()
    cfar_detected, cfar_threshold = cfar_detector.detect(spec_arr)
    cfar_label = "Target" if cfar_detected else "Clutter"
    cfar_correct = cfar_label == true_label

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
                <p style="font-size:20px; margin:0">{confidence * 100:.1f}% confidence</p>
                <p style="color:#888; font-size:13px; margin:6px 0">
                    True signal type: <strong>{true_label}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ca, cb = st.columns(2)
        ca.metric("Target score", f"{probs[1] * 100:.1f}%")
        cb.metric("Clutter score", f"{probs[0] * 100:.1f}%")

        cfar_icon = "✓" if cfar_correct else "✗"
        cfar_color = "#2ecc71" if cfar_correct else "#e74c3c"
        st.markdown(
            f"""
            <div style="border:1px solid #444; border-radius:8px;
                        padding:12px 16px; margin-top:8px;">
                <span style="font-size:11px; color:#666; text-transform:uppercase;
                             letter-spacing:0.05em">Classical baseline (MTI + peak detector)</span>
                <div style="font-size:16px; font-weight:bold; color:{cfar_color};
                            margin-top:4px">{cfar_label} {cfar_icon}</div>
                <span style="font-size:12px; color:#888">
                    Doppler peak {"exceeds" if cfar_detected else "below"} threshold
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sig_color = TARGET_COLOR if pred_label == "Target" else CLUTTER_COLOR
    sig_fill = TARGET_FILL if pred_label == "Target" else CLUTTER_FILL
    peak_idx = int(np.argmax(spec_arr))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Time domain — raw ADC samples",
            "Frequency domain — signal and CFAR threshold",
        ],
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=np.arange(SIGNAL_LENGTH), y=sig, mode="lines",
        line=dict(color=sig_color, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=freqs_arr, y=spec_arr, mode="lines",
        name="Spectrum",
        line=dict(color=sig_color, width=1.5),
        fill="tozeroy", fillcolor=sig_fill,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=freqs_arr, y=cfar_threshold, mode="lines",
        name="CFAR threshold",
        line=dict(color="#f39c12", width=1.5, dash="dash"),
    ), row=1, col=2)
    fig.add_annotation(
        x=float(freqs_arr[peak_idx]), y=float(spec_arr[peak_idx]),
        text=f"Peak @ {freqs_arr[peak_idx]:.2f}",
        showarrow=True, arrowhead=2, arrowcolor=sig_color,
        font=dict(color=sig_color, size=11),
        row=1, col=2,
    )
    fig.update_layout(**plot_layout(300), showlegend=True,
                      legend=dict(x=0.52, y=0.97, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    if pred_label == "Target":
        st.info(
            "**Why target?** The frequency spectrum shows a sharp isolated peak — the signature "
            "of a coherent Doppler return. A moving object reflects the radar pulse at a slightly "
            "different frequency, determined by its radial velocity. "
            "That peak is what the model detects."
        )
    else:
        st.info(
            "**Why clutter?** Energy is spread across low frequencies with no dominant peak — the "
            "signature of correlated, slowly-varying ground or sea returns. Clutter lacks the "
            "coherent Doppler component produced by a moving object."
        )


# ── Tab 2: Signal Explorer ─────────────────────────────────────────────────────

with tab_signals:
    st.markdown(
        "Compare target and clutter signals side by side at the same noise level. "
        "The frequency domain reveals why the classifier works — the two signal types "
        "have fundamentally different spectral shapes."
    )
    st.write("")

    snr_exp = st.slider(
        "Signal-to-Noise Ratio (dB)",
        min_value=-10.0, max_value=25.0, value=10.0, step=1.0,
        key="snr_explorer",
        help="Drag left to add noise and watch how the spectral signatures degrade.",
    )
    seed_exp = st.number_input("Seed", min_value=0, max_value=9999, value=7, key="seed_exp")

    rng_exp = np.random.default_rng(int(seed_exp))
    sig_t = generate_target(SIGNAL_LENGTH, snr_exp, rng_exp)
    sig_c = generate_clutter(SIGNAL_LENGTH, snr_exp, rng_exp)
    freqs_t, spec_t = windowed_spectrum(sig_t)
    freqs_c, spec_c = windowed_spectrum(sig_c)
    peak_t = int(np.argmax(spec_t))
    peak_c = int(np.argmax(spec_c))

    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "🎯 Target — time domain",  "🌊 Clutter — time domain",
            "🎯 Target — frequency domain", "🌊 Clutter — frequency domain",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    # Time domain
    fig2.add_trace(go.Scatter(x=np.arange(SIGNAL_LENGTH), y=sig_t, mode="lines",
                              line=dict(color=TARGET_COLOR, width=1.5)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=np.arange(SIGNAL_LENGTH), y=sig_c, mode="lines",
                              line=dict(color=CLUTTER_COLOR, width=1.5)), row=1, col=2)
    # Frequency domain
    fig2.add_trace(go.Scatter(x=freqs_t, y=spec_t, mode="lines",
                              line=dict(color=TARGET_COLOR, width=1.5),
                              fill="tozeroy", fillcolor=TARGET_FILL), row=2, col=1)
    fig2.add_trace(go.Scatter(x=freqs_c, y=spec_c, mode="lines",
                              line=dict(color=CLUTTER_COLOR, width=1.5),
                              fill="tozeroy", fillcolor=CLUTTER_FILL), row=2, col=2)
    # Annotations
    fig2.add_annotation(
        x=float(freqs_t[peak_t]), y=float(spec_t[peak_t]),
        text="Sharp Doppler peak", showarrow=True, arrowhead=2,
        arrowcolor=TARGET_COLOR, font=dict(color=TARGET_COLOR, size=11),
        row=2, col=1,
    )
    fig2.add_annotation(
        x=float(freqs_c[peak_c]), y=float(spec_c[peak_c]),
        text="Diffuse low-freq energy", showarrow=True, arrowhead=2,
        arrowcolor=CLUTTER_COLOR, font=dict(color=CLUTTER_COLOR, size=11),
        row=2, col=2,
    )
    fig2.update_layout(**plot_layout(520))
    fig2.update_xaxes(gridcolor=GRID, zeroline=False)
    fig2.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig2, use_container_width=True)

    col_ta, col_ca = st.columns(2)
    with col_ta:
        st.success(
            "**Target:** A moving object reflects the radar pulse at a frequency shifted "
            "by its velocity (Doppler effect). This appears as a **sharp isolated peak** "
            "in the spectrum. Even at low SNR, the peak persists and the model can detect it."
        )
    with col_ca:
        st.warning(
            "**Clutter:** Ground and sea returns are correlated and slow-varying — they "
            "concentrate energy at **low frequencies** with no dominant peak. "
            "At high SNR this pattern is obvious; at low SNR clutter blends into noise."
        )


# ── Tab 3: Detection Performance ──────────────────────────────────────────────

with tab_perf:
    st.markdown(
        "The ML classifier vs the classical radar baseline — an MTI filter followed by a "
        "spectral peak threshold. Accuracy measured on 400 test signals per SNR point."
    )
    st.write("")

    snr_vals, ml_accs, cfar_accs = snr_benchmark()

    fig3 = go.Figure()
    fig3.add_vrect(x0=-10, x1=0, fillcolor="rgba(231,76,60,0.07)",
                   line_width=0, annotation_text="Very low SNR", annotation_position="top left")
    fig3.add_vrect(x0=0, x1=8, fillcolor="rgba(241,196,15,0.07)",
                   line_width=0, annotation_text="Marginal", annotation_position="top left")
    fig3.add_vrect(
        x0=8, x1=25, fillcolor="rgba(46,204,113,0.05)",
        line_width=0, annotation_text="Operational range", annotation_position="top left",
    )
    fig3.add_trace(go.Scatter(
        x=snr_vals, y=ml_accs * 100,
        mode="lines+markers",
        name="ML classifier",
        line=dict(color=TARGET_COLOR, width=2.5),
        marker=dict(size=5),
        hovertemplate="SNR %{x:.1f} dB<br>ML accuracy %{y:.1f}%<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=snr_vals, y=cfar_accs * 100,
        mode="lines+markers",
        name="Classical baseline",
        line=dict(color=CLUTTER_COLOR, width=2.5, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="SNR %{x:.1f} dB<br>Classical accuracy %{y:.1f}%<extra></extra>",
    ))
    fig3.add_hline(y=80, line_dash="dash", line_color="#e74c3c",
                   annotation_text="80% CI gate", annotation_position="bottom right")
    fig3.add_hline(y=50, line_dash="dot", line_color="#666",
                   annotation_text="Chance level", annotation_position="bottom right")
    fig3.update_layout(
        **plot_layout(420),
        showlegend=True,
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Signal-to-Noise Ratio (dB)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
    )
    fig3.update_xaxes(gridcolor=GRID, zeroline=False)
    fig3.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig3, use_container_width=True)

    p1, p2, p3 = st.columns(3)
    p1.metric("p50 inference latency", "0.035 ms", help="Single sample, host CPU")
    p2.metric("p99 inference latency", "0.055 ms", help="Worst-case single sample")
    p3.metric("FPGA path", "defined", help="ONNX → Xilinx Vitis AI compilation — not included in this repo")

    st.info(
        "**Why ML outperforms the classical baseline at low SNR:** the classical detector "
        "looks for a peak that stands above a fixed noise-floor ratio. At low SNR the peak "
        "is weak and the ratio falls below the threshold — the detector effectively gives up. "
        "The ML model learns the full spectral shape jointly: sharpness, width, and relative "
        "height together, making it more robust when the signal is faint."
    )
    st.info(
        "**Classical baseline:** the detector applies an MTI (Moving Target Indication) filter "
        "to suppress the low-frequency clutter band, then looks for a spectral peak that stands "
        "above the noise floor by a fixed ratio (peak-to-mean > 7.9, tuned on validation data). "
        "This two-stage pipeline — clutter filter then threshold detector — is standard in "
        "operational radar signal processing."
    )


# ── Tab 4: Pipeline ────────────────────────────────────────────────────────────

with tab_pipeline:
    st.markdown(
        "The classifier is one component in an automated model factory. "
        "Every stage is versioned and reproducible — from the raw simulation parameters "
        "to the ONNX file running on the radar signal processor."
    )
    st.write("")

    steps = [
        ("1", "📡", "Synthetic Data Generation",
         "Generates thousands of labelled radar returns with controllable SNR, Doppler "
         "frequency, and clutter structure. Every dataset is pinned to an exact seed — "
         "anyone can reproduce the exact same data from the same parameters.",
         "Python · NumPy · DVC"),
        ("2", "🔬", "Automated Testing",
         "Before training starts, 23 unit tests verify data shapes, model output format, "
         "and that the ONNX graph matches the PyTorch model to numerical precision. "
         "A broken pipeline fails here, not in production.",
         "pytest · ruff"),
        ("3", "🧠", "GPU Training & Experiment Tracking",
         "The model trains on a Kubernetes GPU Job. Every run — different noise levels, "
         "different architectures — is logged automatically. The team compares runs "
         "side by side and selects the best before promoting.",
         "PyTorch · MLflow · Kubernetes"),
        ("4", "✅", "Quality Gate",
         "If the model does not reach 80% validation accuracy, the pipeline fails "
         "automatically. No human needs to check — a bad model cannot be promoted "
         "to the next stage.",
         "CI/CD gate · sys.exit(1)"),
        ("5", "📦", "ONNX Export & Registry",
         "The approved model is exported to ONNX — a universal format accepted by FPGA "
         "toolchains, embedded Linux runtimes, and GPU accelerators. The registry records "
         "which version is in production and maintains a full audit trail.",
         "ONNX · MLflow Registry"),
        ("6", "🚀", "Edge Deployment",
         "The 67 KB ONNX file deploys to the radar signal processor — no retraining, "
         "no ML framework at runtime. It classifies one radar return in 0.035 ms on a "
         "host CPU. The ONNX format is directly accepted by Xilinx Vitis AI for FPGA "
         "compilation; that path is defined but outside the scope of this demonstrator.",
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
