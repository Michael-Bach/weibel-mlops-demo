"""
tabs/tradeoffs.py — render() for tab_tradeoffs.
"""

import plotly.graph_objects as go
import streamlit as st


def render():
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

        st.caption(
            "**Sliding-window deployment on a continuously rotating radar:** "
            "maintain a rolling buffer of the last 10 sweeps and run CNN inference on every new arrival — "
            "drop the oldest sweep, append the latest, recompute max/mean/std, infer. "
            "After the initial 10-sweep ramp-up, output is available every rotation at O(1) cost per sweep, "
            "the same cadence as ConvGRU. "
            "The tradeoff: a slow-moving or hovering target that occupies roughly the same cell for more than "
            "10 sweeps will appear in both the outgoing and incoming window, so the std channel loses "
            "sensitivity — ConvGRU's unbounded hidden state handles that case better."
        )
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

    with col_cfar:   # reuse the 4th column slot
        pass

    # ── Transformer card ───────────────────────────────────────────────────────
    st.divider()
    col_tf, col_gap = st.columns([2, 1])
    with col_tf:
        st.markdown("### 🟠 Patch Temporal Transformer")
        st.markdown("**Learned temporal attention — no hand-engineered features**")
        st.divider()
        col_tf1, col_tf2 = st.columns(2)
        with col_tf1:
            st.markdown("#### ✅ Strengths")
            st.markdown("""
- **No feature engineering** — receives the raw 10-sweep amplitude stack; attention learns
  which sweeps matter at each spatial location, replacing the fixed max/mean/std.
- **Interpretable attention weights** — the (10×10) attention matrix per patch shows
  exactly which sweep pairs the model uses for detection.
- **Comparable AUC to CNN** — achieves similar cell-level ROC performance with a
  different inductive bias (learned vs. fixed temporal aggregation).
- **~20 k parameters** — same order as the CNN, easily deployable.
            """)
        with col_tf2:
            st.markdown("#### ❌ Weaknesses")
            st.markdown("""
- **Batch only** — like the CNN, requires a full 10-sweep window before inference;
  not suitable for streaming.
- **Patch-level resolution** — output is decoded at 6×4 bin patch granularity
  (30×16 patches), slightly coarser than the per-cell CNN output.
- **Needs more training data** — transformers are more data-hungry than CNNs;
  the current model was trained on the same 800-sequence set, which may under-utilise
  the architecture's capacity.
- **Fixed context window** — same sliding-window limitation as the CNN for
  continuous rotation deployment.
            """)
        st.info("**Best role:** offline re-evaluation or second-opinion pass where interpretable attention maps help operators understand why a detection was flagged.")

    st.divider()

    # ── Operational role matrix ────────────────────────────────────────────────
    st.markdown("### Operational role matrix")
    st.markdown(
        "The table below maps each algorithm to operational radar use cases. "
        "Green = strong fit, yellow = acceptable, red = poor fit."
    )

    st.markdown("""
| Requirement | CA-CFAR+KF | CNN+KF | ConvGRU+KF | Transformer |
|---|:---:|:---:|:---:|:---:|
| **Early warning (react before 5 sweeps)** | 🔴 Slow | 🟡 Moderate | 🟢 Best | 🟡 Moderate |
| **Low-SNR detection (0–6 dB)** | 🔴 Poor | 🟡 Good | 🟢 Best | 🟡 Good |
| **False-alarm-limited environment** | 🟡 Baseline | 🟢 Best | 🟡 Comparable to CFAR | 🟢 Good |
| **Multi-target tracking** | 🟢 Yes | 🔴 Conflicts targets | 🟢 Yes | 🔴 Conflicts targets |
| **Zero-data deployment** | 🟢 No data needed | 🔴 Needs training | 🔴 Needs training | 🔴 Needs training |
| **Explainability / certification** | 🟢 Fully transparent | 🟡 Interpretable map | 🟡 Interpretable map | 🟢 Attention weights |
| **Embedded / hard real-time** | 🟢 2.2 ms/sweep | 🟢 0.8 ms/batch | 🟢 <1 ms/sweep | 🟡 ~2 ms/batch |
| **Heterogeneous clutter** | 🔴 Breaks at edges | 🟢 Learned robustness | 🟡 Needs retraining | 🟢 Learned robustness |
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
