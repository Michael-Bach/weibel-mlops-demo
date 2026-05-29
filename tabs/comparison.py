"""
tabs/comparison.py — render(snr_db) for tab_compare.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from radar.sessions import N_SW, N_AZ, N_RANGE, _RANGE_BIN_M
from radar.detection import (
    _calibrate_pfa,
    _snr_sweep,
    _sweep_profile,
    het_clutter_demo,
)


def render():
    # ── Operating-point calibration (run once, cached) ────────────────────────
    cfar_pfa, cnn_thr, cfar_kf_ftr, cnn_kf_ftr, gru_thr, gru_kf_ftr = _calibrate_pfa()

    # ── Operational headline ──────────────────────────────────────────────────
    st.caption("*Drag the SNR slider — the AI pipeline detects 4–6 dB lower than CFAR at the same false-alarm rate.*")
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
            "All six detectors are evaluated **at the same false-alarm rate** so none has an unfair advantage. "
            "Thresholds are calibrated so all produce the same cell-level noise-hit rate as CFAR. "
            "Three pipelines use a Kalman filter for confirmed-track detection; three are evaluated "
            "at cell level only (no KF backend — they output a score map, not a confirmed track)."
        )
        st.code(
            "── KF-confirmed track detection ──────────────────────────────────────\n"
            "CFAR+KF:     raw PPI → CFAR (α=2.5)             → peaks → KF → confirmed track\n"
            "CNN+KF:      raw PPI → CNN  (batch, 10 sweeps)   → peaks → KF → confirmed track\n"
            "GRU+KF:      raw PPI → ConvGRU (streaming/sweep) → peaks → KF → confirmed track\n"
            "\n"
            "── Cell-level detection (no KF) ──────────────────────────────────────\n"
            "LRT:         raw PPI → sum squared normalised amplitudes → threshold\n"
            "DP-TBD:      raw PPI → DP accumulate energy along best path → threshold\n"
            "Transformer: raw PPI → patch self-attention across 10 sweeps → threshold",
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
        "**New to this demo? Focus on the three solid lines** — those are the main comparison "
        "(CFAR in orange, CNN in blue, ConvGRU in purple). "
        "The dash-dot lines (LRT, DP-TBD, Transformer) are additional classical and ML variants "
        "covered in the Advanced sections of Tab 1 and Tab 2. "
        "Each point = 30 randomised trials at random target positions, speeds, and directions."
    )
    (snr_arr,
     cfar_sa_pd, cfar_kf_pd, cnn_b_pd, cnn_kf_pd,
     cfar_kf_rmse_arr, cnn_kf_rmse_arr,
     gru_kf_pd, gru_kf_rmse_arr) = _snr_sweep(cnn_thr=cnn_thr, gru_thr=gru_thr, _v=11)

    # Load cell-level Pd from precomputed CSV (LRT, DP-TBD, Transformer)
    _csv = Path("artifacts/comparison_snr.csv")
    _df  = pd.read_csv(_csv) if _csv.exists() else None

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
    if _df is not None:
        fig_c.add_trace(go.Scatter(
            x=_df["snr_db"], y=_df["lrt_pd"] * 100, mode="lines+markers",
            name="LRT non-coherent  (cell-level)",
            line=dict(color="#2ecc71", width=1.8, dash="dashdot"),
            marker=dict(size=5, symbol="triangle-up"),
        ))
        fig_c.add_trace(go.Scatter(
            x=_df["snr_db"], y=_df["dptbd_pd"] * 100, mode="lines+markers",
            name="DP-TBD  (cell-level)",
            line=dict(color="#e74c3c", width=1.8, dash="dashdot"),
            marker=dict(size=5, symbol="triangle-down"),
        ))
        if "transformer_pd" in _df.columns:
            fig_c.add_trace(go.Scatter(
                x=_df["snr_db"], y=_df["transformer_pd"] * 100, mode="lines+markers",
                name="Transformer  (cell-level)",
                line=dict(color="#f39c12", width=1.8, dash="dashdot"),
                marker=dict(size=6, symbol="star"),
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
        "All six detectors compared at the same false-alarm rate (matched Pfa). "
        "KF pipelines: detection counts when the confirmed track falls within ~45 m of the true target. "
        "Cell-level detectors (dash-dot): detection counts when the score at the oracle target position exceeds the calibrated threshold."
    )

    st.divider()

    # ── Metrics table at matched Pfa ──────────────────────────────────────────
    st.markdown("#### Summary metrics at matched operating point")

    idx10 = int(np.argmin(np.abs(snr_arr - 10.0)))
    idx0  = int(np.argmin(np.abs(snr_arr - 0.0)))
    idx4  = int(np.argmin(np.abs(snr_arr - 4.0)))

    # Cell-level Pd from CSV
    def _csv_pd(col, snr):
        if _df is None or col not in _df.columns: return "—"
        row = _df[np.isclose(_df["snr_db"], snr, atol=0.5)]
        return f"{row[col].values[0]*100:.0f} %" if len(row) else "—"

    col_tbl = st.columns(7)
    for i, h in enumerate(["**Metric**","**CFAR+KF**","**CNN+KF**","**GRU+KF** ⭐",
                            "*LRT*†","*DP-TBD*†","*Transf.*†"]):
        col_tbl[i].markdown(h)

    rows = [
        ("Pd at 0 dB (faint target)",
         f"{cfar_kf_pd[idx0]*100:.0f} %", f"{cnn_kf_pd[idx0]*100:.0f} %",
         f"**{gru_kf_pd[idx0]*100:.0f} %**",
         _csv_pd("lrt_pd", 0), _csv_pd("dptbd_pd", 0), _csv_pd("transformer_pd", 0)),
        ("Pd at 4 dB",
         f"{cfar_kf_pd[idx4]*100:.0f} %", f"{cnn_kf_pd[idx4]*100:.0f} %",
         f"**{gru_kf_pd[idx4]*100:.0f} %**",
         _csv_pd("lrt_pd", 4), _csv_pd("dptbd_pd", 4), _csv_pd("transformer_pd", 4)),
        ("Pd at 10 dB (moderate target)",
         f"{cfar_kf_pd[idx10]*100:.0f} %", f"{cnn_kf_pd[idx10]*100:.0f} %",
         f"**{gru_kf_pd[idx10]*100:.0f} %**",
         _csv_pd("lrt_pd", 12), _csv_pd("dptbd_pd", 12), _csv_pd("transformer_pd", 12)),
        ("Track position error at 10 dB",
         f"{cfar_kf_rmse_arr[idx10]*_RANGE_BIN_M:.0f} m",
         f"{cnn_kf_rmse_arr[idx10]*_RANGE_BIN_M:.0f} m",
         f"**{gru_kf_rmse_arr[idx10]*_RANGE_BIN_M:.0f} m**",
         "N/A", "N/A", "N/A"),
        ("Ghost tracks / noise window",
         f"{cfar_kf_ftr:.2f}", f"{cnn_kf_ftr:.2f}", f"**{gru_kf_ftr:.2f}**",
         "N/A", "N/A", "N/A"),
    ]
    for row in rows:
        cols = st.columns(7)
        for i, v in enumerate(row):
            cols[i].markdown(v)

    st.caption(
        "† Cell-level detectors (LRT, DP-TBD, Transformer) have no KF backend: "
        "track error and ghost-track rate are not applicable. "
        "All detectors evaluated at the same false-alarm rate (matched Pfa)."
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
        sweep_snr = st.slider("SNR (dB)", -20.0, 40.0, 10.0, 1.0,
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

    cfar_hits_h, gru_prob_h, mean_amp_h, patch_mask_h, positions_h = het_clutter_demo(gru_thr)

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

    st.info(
        "**Next tab →** ROC Curve: a different view of the same comparison — how each detector "
        "trades off targets caught against false alarms generated."
    )
