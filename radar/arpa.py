"""
radar/arpa.py — ARPA computation and Plotly traces.
"""

import numpy as np
import plotly.graph_objects as go

from radar.sessions import N_RANGE, _RANGE_BIN_M, _SWEEP_PERIOD_S, _MS_TO_KT
from radar.geometry import az_r_to_px


def _arpa_course_speed(positions: list, vr: float, vt: float, n_ahead: int = 3):
    """
    Compute ARPA navigation data from last known position + velocity.

    CPA accounts for both radial and tangential velocity components:
      vt_eff [bins/sweep] = vt [°/sweep] × (π/180) × r_now
      CPA = r_now × |vt_eff| / sqrt(vr² + vt_eff²)
      t_CPA = -r_now × vr / (vr² + vt_eff²)  (when approaching)

    Returns: course_deg (0-360°T), speed_kt, cpa_range_m, t_to_cpa_s
    """
    az_now, r_now = positions[-1]

    # Convert tangential rate to arc-length velocity at current range
    vt_eff = vt * np.pi / 180.0 * r_now   # bins/sweep (perpendicular to line of sight)
    v_total_bins = np.sqrt(vr**2 + vt_eff**2)

    # Course: compass bearing of motion vector in Cartesian space
    az_pred = (az_now + n_ahead * vt) % 360.0
    r_pred  = r_now + n_ahead * vr
    def to_cart(az, r):
        rad = np.radians(az)
        return r * np.sin(rad), r * np.cos(rad)
    x0, y0 = to_cart(az_now, r_now)
    x1, y1 = to_cart(az_pred, r_pred)
    dx, dy  = x1 - x0, y1 - y0
    course_deg = np.degrees(np.arctan2(dx, dy)) % 360.0

    # Speed in knots
    speed_ms = v_total_bins * _RANGE_BIN_M / _SWEEP_PERIOD_S
    speed_kt = speed_ms * _MS_TO_KT

    # CPA: minimum range using both velocity components
    if v_total_bins > 1e-6:
        cpa_r_bins = r_now * abs(vt_eff) / v_total_bins
        # Time to CPA (sweeps): derivative of r(t)² set to zero
        denom = vr**2 + vt_eff**2
        t_cpa_sw = max(0.0, -r_now * vr / denom) if denom > 1e-9 else 0.0
    else:
        cpa_r_bins = r_now
        t_cpa_sw   = 0.0

    cpa_r_m  = cpa_r_bins * _RANGE_BIN_M
    t_cpa_s  = t_cpa_sw * _SWEEP_PERIOD_S

    return course_deg, speed_kt, cpa_r_m, t_cpa_s


def _arpa_traces(positions: list, sweep_idx: int, vr: float, vt: float,
                 n_pred: int = 4) -> list:
    """
    Build Plotly traces for ARPA overlay:
      - Track history trail (past true positions, fading)
      - Velocity vector (predicted future position)
    """
    traces = []

    # ── Track history ─────────────────────────────────────────────────────────
    n_past = sweep_idx + 1
    if n_past > 1:
        hist_az = [positions[i][0] for i in range(n_past)]
        hist_r  = [positions[i][1] for i in range(n_past)]
        hx = [az_r_to_px(a, r)[0] for a, r in zip(hist_az, hist_r)]
        hy = [az_r_to_px(a, r)[1] for a, r in zip(hist_az, hist_r)]
        # Connected dashed line
        traces.append(go.Scatter(
            x=hx, y=hy, mode="lines",
            line=dict(color="rgba(255,80,80,0.55)", width=1.5, dash="dot"),
            showlegend=False, name="ARPA track",
        ))
        # Past-position dots fading by age
        for i, (px, py) in enumerate(zip(hx[:-1], hy[:-1])):
            age   = n_past - 1 - i
            alpha = max(0.15, 0.85 - age * 0.15)
            sz    = max(3, 7 - age)
            traces.append(go.Scatter(
                x=[px], y=[py], mode="markers",
                marker=dict(color=f"rgba(255,100,100,{alpha:.2f})",
                            size=sz, symbol="circle"),
                showlegend=False,
            ))

    # ── Velocity vector ────────────────────────────────────────────────────────
    az_now, r_now = positions[sweep_idx]
    az_pred = (az_now + n_pred * vt) % 360.0
    r_pred  = float(np.clip(r_now + n_pred * vr, 0, N_RANGE - 1))
    tx,  ty  = az_r_to_px(az_now,  r_now)
    tvx, tvy = az_r_to_px(az_pred, r_pred)
    traces.append(go.Scatter(
        x=[tx, tvx], y=[ty, tvy], mode="lines+markers",
        line=dict(color="rgba(255,220,0,0.85)", width=2),
        marker=dict(symbol=["circle", "triangle-up"],
                    color="rgba(255,220,0,0.85)", size=[0, 9]),
        showlegend=False, name="ARPA vector",
    ))

    return traces
