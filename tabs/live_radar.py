"""
tabs/live_radar.py — render() for tab_live.
"""

import numpy as np
import streamlit as st

from radar.sessions import _TGT_COLORS
from radar.live import live_init, live_tick
from components.charts import live_ppi_fig, live_conf_fig, live_gru_heatmap_fig


def _lat_str(conf_dict: dict, key: str) -> str:
    v = conf_dict.get(key)
    return f"sw {v} ✓" if v is not None else "—"


def _med_str(lats: list) -> str:
    return f"{float(np.median(lats)):.1f} sw  (n={len(lats)})" if lats else "—"


def render():
    # ── Initialise state ──────────────────────────────────────────────────────
    if "live" not in st.session_state:
        live_init()
    s = st.session_state["live"]

    # ── Intro ─────────────────────────────────────────────────────────────────
    st.caption("*Click ▶ Start — watch the AI track appear before CFAR reacts, especially at low signal strength.*")
    st.markdown(
        "Multiple targets enter and cross the radar scan area at random speeds and signal strengths. "
        "**Watch how quickly each system raises a confirmed alarm** — the classical CFAR pipeline "
        "(white ✕) and the ConvGRU AI pipeline (magenta ◆) run side by side on identical data. "
        "At low signal strength the AI track appears several antenna rotations before the classical "
        "system reacts; the magenta heatmap shows the AI's growing confidence building sweep by sweep."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctl1, ctl2, ctl3, ctl_gap = st.columns([1, 1, 1, 6])
    with ctl1:
        if st.button("▶ Start", key="live_start", use_container_width=True,
                     disabled=s["running"]):
            s["running"] = True
            st.rerun()
    with ctl2:
        if st.button("■ Stop", key="live_stop", use_container_width=True,
                     disabled=not s["running"]):
            s["running"] = False
            st.rerun()
    with ctl3:
        if st.button("↺ Reset", key="live_reset", use_container_width=True):
            live_init()
            st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_live_scope, col_live_info = st.columns([5, 3], gap="large")

    with col_live_scope:
        live_scope_ph = st.empty()
        live_conf_ph  = st.empty()
        live_gru_ph   = st.empty()

    with col_live_info:
        sw_now = s["sweep_count"]
        n_active = len(s["targets"])

        st.markdown(
            f"**Sweep {sw_now}** &nbsp;·&nbsp; "
            f"Active targets: **{n_active}** &nbsp;·&nbsp; "
            f"{'🟢 Running' if s['running'] else '🔴 Stopped'}"
        )
        st.divider()

        # ── Target confirmation table ─────────────────────────────────────────
        st.markdown("**Target confirmation status**")

        active_ids = {t["id"] for t in s["targets"]}
        # Show last 8 targets (active or recently expired)
        shown_ids = sorted(s["confirm"].keys(), reverse=True)[:8]

        if shown_ids:
            hdr_cols = st.columns([1, 2, 2, 2])
            hdr_cols[0].caption("Tgt")
            hdr_cols[1].caption("SNR")
            hdr_cols[2].caption("CFAR+KF")
            hdr_cols[3].caption("GRU+KF")

            for tid in shown_ids:
                color  = _TGT_COLORS[tid % len(_TGT_COLORS)]
                active = tid in active_ids
                tgt_obj = next((t for t in s["targets"] if t["id"] == tid), None)
                snr_str = f"{tgt_obj['snr_db']:.0f}dB" if tgt_obj else "—"

                row = st.columns([1, 2, 2, 2])
                label = f"{'●' if active else '○'} #{tid}"
                row[0].markdown(
                    f"<span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True,
                )
                row[1].caption(snr_str)
                conf = s["confirm"].get(tid, {})
                row[2].caption(_lat_str(conf, "cfar_kf"))
                row[3].caption(_lat_str(conf, "gru_kf"))
        else:
            st.caption("No targets yet — click ▶ Start")

        st.divider()

        # ── Median confirmation latency ───────────────────────────────────────
        st.markdown("**Median sweeps to confirmed track**")

        m1, m2 = st.columns(2)
        m1.metric("CFAR+KF", _med_str(s["lat_cfar"]))
        m2.metric("GRU+KF",  _med_str(s["lat_gru"]))

        st.divider()

        st.caption(
            "● Coloured circles = true targets · "
            "✕ White = CFAR+KF · ◆ Magenta = GRU+KF · "
            "Magenta glow = GRU confidence heatmap"
        )

    # ── Render frames ─────────────────────────────────────────────────────────
    live_scope_ph.plotly_chart(live_ppi_fig(s), use_container_width=False,
                               key=f"live_ppi_{sw_now}")
    live_conf_ph.plotly_chart(live_conf_fig(s), use_container_width=True,
                              key=f"live_conf_{sw_now}")
    live_gru_ph.plotly_chart(live_gru_heatmap_fig(s), use_container_width=True,
                             key=f"live_gru_{sw_now}")

    # Tick and rerun if running
    if s["running"]:
        import time
        live_tick()
        time.sleep(s["frame_s"])
        st.rerun()
