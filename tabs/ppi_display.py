"""
tabs/ppi_display.py — render() for tab_ppi.
"""

import numpy as np
import streamlit as st

from radar.sessions import N_SW, N_AZ, N_RANGE, _RANGE_BIN_M, load_gru_session
from radar.inference import _ml_map, _gru_seq, _gru_peaks
from radar.arpa import _arpa_course_speed
from radar.detection import _gen, _cfar_sweeps, _kf_result, _build_kf_history_cfar, _build_kf_history_cnn
from components.ppi_canvas import make_ppi_figure
from components.charts import _pd_chart_fig, waterfall_fig, gru_evolution_fig, render_anim_frame
from src.baseline.ppi_cfar_kf import PPIKalmanTracker


def render(snr_db, range_bin, az_deg, vr, vt,
           show_ml, show_cfar, show_kf, show_arpa, show_gru_kf, show_cnn_kf):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    gru_session = load_gru_session()

    ppi_seq, label_seq, positions = _gen(snr_db, range_bin, az_deg, vr, vt)
    cfar_sweeps_arr               = _cfar_sweeps(snr_db, range_bin, az_deg, vr, vt)
    _, kf_pos_list                 = _kf_result(snr_db, range_bin, az_deg, vr, vt)

    # Animation state
    if "animating" not in st.session_state:
        st.session_state.animating = False
    if "anim_frame" not in st.session_state:
        st.session_state.anim_frame = 0

    col_scope, col_info = st.columns([5, 3], gap="large")

    with col_scope:
        sw_col, btn_col, stop_col = st.columns([5, 1, 1])
        with sw_col:
            sweep_idx = st.slider("Show sweep", 1, N_SW, N_SW, key="sweep") - 1
        with btn_col:
            st.write("")
            if st.button("▶ Animate", key="ppi_animate", use_container_width=True):
                st.session_state.animating = True
                st.session_state.anim_frame = 0
        with stop_col:
            st.write("")
            if st.button("■ Stop", key="ppi_stop", use_container_width=True):
                st.session_state.animating = False

        scope_ph = st.empty()
        pd_ph    = st.empty()

    # Precompute CNN confidence at target for each sweep (padded-window CNN)
    ml_conf_history = []
    for _n in range(1, N_SW + 1):
        _m  = _ml_map(ppi_seq, _n)
        _az = int(positions[_n - 1][0] / 360 * N_AZ) % N_AZ
        _r  = int(np.clip(positions[_n - 1][1], 0, N_RANGE - 1))
        ml_conf_history.append(float(_m[_az, _r]))

    # Precompute GRU: sequential single-sweep steps — conf at target + full heatmap per sweep
    if gru_session is not None:
        gru_conf_history, gru_heatmap_history = _gru_seq(ppi_seq, positions)
    else:
        gru_conf_history, gru_heatmap_history = None, None

    # Per-sweep KF track histories for the three-pipeline comparison
    kf_history_cfar = _build_kf_history_cfar(snr_db, range_bin, az_deg, vr, vt)
    kf_history_cnn  = _build_kf_history_cnn(snr_db, range_bin, az_deg, vr, vt)

    # GRU+KF history: run KF over already-computed heatmaps (fast, no extra inference)
    if gru_heatmap_history:
        _kf_gru = PPIKalmanTracker()
        kf_history_gru: list = []
        for _h_map in gru_heatmap_history:
            _kf_gru.update_from_peaks(_gru_peaks(_h_map), N_AZ, N_RANGE)
            kf_history_gru.append(_kf_gru.track_positions())
    else:
        kf_history_gru = [[] for _ in range(N_SW)]

    ml_map_now = _ml_map(ppi_seq, sweep_idx + 1)

    with col_info:
        az_now, r_now = positions[sweep_idx]
        az_b = int(az_now / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(r_now, 0, N_RANGE - 1))

        # ── ARPA navigation block ─────────────────────────────────────────────
        course, speed_kt, cpa_r_m, t_cpa_s = _arpa_course_speed(
            positions[:sweep_idx + 1], vr, vt
        )
        range_m = r_now * _RANGE_BIN_M
        st.markdown("#### ARPA Target Data")
        c1, c2 = st.columns(2)
        c1.metric("Bearing", f"{az_now:.1f}°T")
        c2.metric("Range", f"{range_m:.0f} m" if range_m < 1000 else f"{range_m/1000:.2f} km")
        c1.metric("Course", f"{course:.0f}°T")
        c2.metric("Speed", f"{speed_kt:.1f} kt")
        if vr < 0 or abs(vr) < 0.05:
            c1.metric("CPA range", f"{cpa_r_m:.0f} m" if cpa_r_m < 1000 else f"{cpa_r_m/1000:.2f} km")
            c2.metric("Time to CPA", f"{t_cpa_s:.0f} s")
        else:
            c1.metric("Aspect", "Opening")
            c2.metric("CPA", "N/A")
        st.divider()

        # ── Detector confidence ───────────────────────────────────────────────
        # All three detectors are evaluated at the EXACT target cell in each sweep
        # so the metrics are directly comparable (no neighbourhood inflation).
        n_sw_shown = sweep_idx + 1

        # CFAR Pd: fraction of sweeps where CFAR fired at the exact target cell
        cfar_hits_path = 0
        for _sw in range(n_sw_shown):
            _az_sw, _r_sw = positions[_sw]
            _az_b = int(_az_sw / 360 * N_AZ) % N_AZ
            _r_b  = int(np.clip(_r_sw, 0, N_RANGE - 1))
            if cfar_sweeps_arr[_sw, _az_b, _r_b]:
                cfar_hits_path += 1
        cfar_pd_path = cfar_hits_path / n_sw_shown

        # ML confidence: max in a 3×3 window (CNN output is spatially smoothed —
        # a ±1 window is still single-cell equivalent for comparison with CFAR)
        ml_win = ml_map_now[
            max(0, az_b - 1):az_b + 2,
            max(0, r_b  - 1):r_b  + 2,
        ]
        ml_conf = float(ml_win.max()) if ml_win.size else float(ml_map_now[az_b, r_b])

        st.markdown(f"#### Pipeline Comparison — Sweep {n_sw_shown}/{N_SW}")

        # Three-pipeline track summary at current sweep
        _cfar_trks = kf_history_cfar[sweep_idx]
        _gru_trks  = kf_history_gru[sweep_idx]
        _cnn_trks  = kf_history_cnn[sweep_idx]

        # Single-detector confidence metrics
        st.metric("CNN confidence (peak near target)", f"{ml_conf:.1%}",
                  help="Max CNN output in 3×3 cell window around the target's current position")
        if gru_heatmap_history:
            gru_h_val = float(gru_heatmap_history[sweep_idx][az_b, r_b])
            st.metric("GRU h at target cell", f"{gru_h_val:.1%}",
                      help="Hidden state value at the exact target cell — directly the confidence probability.")
        st.metric("CFAR Pd at target cell", f"{cfar_pd_path:.1%}",
                  help="Fraction of sweeps where CFAR fired at the exact target cell — true single-cell Pd")

        st.divider()
        st.markdown("**Confirmed tracks (≥4 associations)**")
        _col1, _col2, _col3 = st.columns(3)
        _col1.metric("CFAR+KF", str(len(_cfar_trks)),
                     help="White ✕ — classical pipeline confirmed tracks at this sweep")
        _col2.metric("GRU+KF", str(len(_gru_trks)),
                     help="Magenta ◆ — streaming GRU heatmap peaks → same KF")
        _col3.metric("CNN+KF", str(len(_cnn_trks)),
                     help="Yellow ▲ — batch CNN peaks → same KF")
        st.divider()

        st.caption(
            "🟢 Green = radar echo · 🔵 Cyan = CNN confidence · 🟠 Orange = CFAR · "
            "⬜ White ✕ = CFAR+KF · 🟣 Magenta ◆ = GRU+KF · 🟡 Yellow ▲ = CNN+KF · "
            "🔴 Red ✕ = true target · 🔴 Red trail = ARPA · 🟡 Yellow → = ARPA vector"
        )

        with st.expander("How the ML model works"):
            st.markdown("""
The CNN receives **three 2-D feature channels** derived from all visible sweeps:

| Channel | Description |
|---------|-------------|
| **Max** | Pixel-wise max — peak beam illumination |
| **Mean** | Average — clutter baseline |
| **Std** | Std deviation — high where signal *varied* (motion) |

All channels are divided by the per-range clutter floor (10th-percentile amplitude) to cancel range attenuation. A moving target leaves a bright streak in the max and std channels that the CNN learns to recognise.
""")

    # Render scope + Pd chart (static or animated)
    if st.session_state.animating:
        has_more = render_anim_frame(
            st.session_state.anim_frame,
            ppi_seq, positions, cfar_sweeps_arr, kf_pos_list,
            ml_conf_history, vr, vt,
            show_ml, show_cfar, show_kf, show_arpa,
            scope_ph, pd_ph,
            gru_history=gru_conf_history,
            gru_heatmaps=gru_heatmap_history,
            kf_history_cfar=kf_history_cfar,
            kf_history_gru=kf_history_gru,
            kf_history_cnn=kf_history_cnn,
            show_gru_kf=show_gru_kf,
            show_cnn_kf=show_cnn_kf,
        )
        if has_more:
            st.session_state.anim_frame += 1
            st.rerun()
        else:
            st.session_state.animating = False
    else:
        scope_ph.plotly_chart(
            make_ppi_figure(
                ppi_seq, positions, cfar_sweeps_arr,
                kf_history_cfar[sweep_idx],   # per-sweep CFAR+KF positions
                ml_map_now, sweep_idx, vr, vt,
                show_ml, show_cfar, show_kf, show_arpa,
                gru_map_now=gru_heatmap_history[sweep_idx] if gru_heatmap_history else None,
                show_gru=show_ml,
                gru_kf_pos=kf_history_gru[sweep_idx],
                show_gru_kf=show_gru_kf,
                cnn_kf_pos=kf_history_cnn[sweep_idx],
                show_cnn_kf=show_cnn_kf,
            ),
            use_container_width=False,
            key="static_scope",
        )
        pd_ph.plotly_chart(
            _pd_chart_fig(ml_conf_history, sweep_idx + 1, gru_conf_history),
            use_container_width=True,
            key="static_pd",
        )

    # ── GRU hidden-state evolution ────────────────────────────────────────────
    if gru_heatmap_history:
        st.markdown(
            "#### GRU hidden state — confidence accumulates over 10 sweeps"
        )
        st.caption(
            "Each panel shows the full (azimuth × range) confidence map h after that sweep. "
            "h starts at zero; the update gate z decides how much new evidence to absorb. "
            "Red × = true target position. Brighter = higher confidence."
        )
        st.plotly_chart(
            gru_evolution_fig(gru_heatmap_history, positions),
            use_container_width=True,
            key="gru_evolution",
        )

    # ── Waterfall / surface plot ───────────────────────────────────────────────
    st.plotly_chart(
        waterfall_fig(ppi_seq, positions),
        use_container_width=True,
        key="waterfall",
    )
