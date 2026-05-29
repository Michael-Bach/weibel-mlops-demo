"""
components/ppi_canvas.py — ppi_rgba, scope_overlay, phosphor_map, make_ppi_figure, anim_fig.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from radar.sessions import N_SW, N_AZ, N_RANGE, N_GRID
from radar.geometry import p2c, az_r_to_px
from radar.arpa import _arpa_traces


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


def anim_fig(img_rgba: np.ndarray, beam_az_deg: float,
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
