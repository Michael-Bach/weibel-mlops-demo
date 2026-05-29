"""
components/charts.py — pd_chart_fig, waterfall_fig, gru_evolution_fig,
                        render_anim_frame, live_ppi_fig, live_conf_fig.
"""

import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from radar.sessions import N_SW, N_AZ, N_RANGE, N_GRID, _RANGE_BIN_M, _TGT_COLORS
from radar.geometry import p2c, az_r_to_px
from radar.sessions import load_gru_session
from components.ppi_canvas import (
    _ppi_rgba, _scope_overlay, _phosphor_map, anim_fig,
)
from radar.inference import _ml_map


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


def waterfall_fig(ppi_seq: np.ndarray, positions: list) -> go.Figure:
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


def gru_evolution_fig(heatmap_history: list, positions: list) -> go.Figure:
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


def render_anim_frame(frame: int, ppi_seq, positions, cfar_sweeps, kf_pos_list,
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
            anim_fig(img_fin, 361.0, N_SW - 1, positions,
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
        anim_fig(img, beam_deg, sw, positions,
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


def live_ppi_fig(s: dict) -> go.Figure:
    """PPI scope for the live-radar demo — phosphor + all three pipeline tracks."""
    gru_session = load_gru_session()

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
    from radar.live import _live_tgt_pos
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


def live_conf_fig(s: dict) -> go.Figure:
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
