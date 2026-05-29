"""
tabs/fleet.py — Fleet learning loop visualisation.

Illustrates the proposed continuous learning cycle: XENTA units spread across
deployment environments, streaming detection data to a central training hub,
receiving updated ONNX models, and collectively classifying an ever-growing
catalogue of drone types.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Fleet definition ──────────────────────────────────────────────────────────

_UNITS = [
    {"id": "C3-0041", "env": "Inland",         "psi": 0.04, "status": "ok",    "model": "v8",   "detections": 47},
    {"id": "C3-0042", "env": "Coastal (high)",  "psi": 0.22, "status": "alert", "model": "v7",   "detections": 31},
    {"id": "C3-0051", "env": "Coastal",         "psi": 0.17, "status": "warn",  "model": "v8",   "detections": 28},
    {"id": "C3-0089", "env": "Border",          "psi": 0.06, "status": "ok",    "model": "v8",   "detections": 62},
    {"id": "C1-0012", "env": "Urban",           "psi": 0.03, "status": "ok",    "model": "v8",   "detections": 19},
    {"id": "C3-0103", "env": "Inland",          "psi": 0.08, "status": "ok",    "model": "v8",   "detections": 53},
    {"id": "C3-0121", "env": "Coastal",         "psi": 0.11, "status": "warn",  "model": "v8",   "detections": 22},
    {"id": "C1-0018", "env": "Urban",           "psi": 0.05, "status": "ok",    "model": "v8",   "detections": 38},
]

_STATUS_COLOR = {"ok": "#2ecc71", "warn": "#f39c12", "alert": "#e74c3c"}

# Drone detections near specific units
_DETECTIONS = [
    {"unit_idx": 0, "dx":  0.75, "dy":  0.45, "label": "DJI M4 ✓",      "conf": 0.97, "known": True},
    {"unit_idx": 1, "dx": -0.55, "dy": -0.60, "label": "FPV-250 ⚠",     "conf": 0.43, "known": False},
    {"unit_idx": 3, "dx":  0.60, "dy": -0.50, "label": "DJI Spark ✓",   "conf": 0.91, "known": True},
    {"unit_idx": 5, "dx": -0.70, "dy":  0.30, "label": "Autel EVO ✓",   "conf": 0.88, "known": True},
    {"unit_idx": 6, "dx": -0.60, "dy":  0.55, "label": "? queued",       "conf": 0.38, "known": False},
]

# Simulated event log entries
_EVENTS = [
    ("📤", "C3-0103",  "uploaded 531 PSI samples — hub ingesting"),
    ("🟢", "C3-0103",  "classified Autel EVO II (conf 0.88)  ← new type absorbed in v8"),
    ("📤", "C3-0042",  "uploaded 487 PSI samples — PSI alert 0.22 flagged"),
    ("⚠️", "C3-0042",  "unknown FPV airframe detected — queued for test-range labelling"),
    ("🤖", "Agent",    "recommended coastal test-range session (units 0042, 0051, 0121)"),
    ("🔄", "Hub",      "retraining on +44 labelled coastal sequences — epoch 1/30"),
    ("🔄", "Hub",      "training converged — val F1 0.371 (prev 0.356) — accuracy gate passed"),
    ("⬇️", "Hub",      "deployed model v8.1-coastal → C3-0042, C3-0051, C3-0121"),
    ("🟢", "C3-0042",  "PSI normalising (0.14 → 0.09) after v8.1 model update"),
    ("📤", "C3-0089",  "uploaded 651 PSI samples — highest-volume unit in fleet"),
    ("🟢", "C3-0089",  "classified DJI Spark (conf 0.91)"),
    ("🟢", "C1-0012",  "classified DJI M4 (conf 0.97)"),
    ("📤", "C3-0051",  "uploaded 223 PSI samples — warn 0.17 steady"),
    ("🟢", "C3-0121",  "classified FPV-250 (conf 0.89)  ← absorbed in v8.1"),
    ("⬇️", "Hub",      "deployed model v8.2 → full fleet (inland + urban refresh)"),
]


# ── Visualisation ─────────────────────────────────────────────────────────────

def _fleet_fig() -> go.Figure:
    """Network diagram of the continuous fleet learning loop."""

    # Position units in a ring, starting from the top
    n = len(_UNITS)
    angles = [i * 2 * np.pi / n - np.pi / 2 for i in range(n)]
    R = 3.1
    units = []
    for i, u in enumerate(_UNITS):
        units.append({**u, "x": R * np.cos(angles[i]), "y": R * np.sin(angles[i])})

    HX, HY = 0.0, 0.0
    fig = go.Figure()

    # ── Background edges (all units ↔ hub, dim) ───────────────────────────────
    for u in units:
        fig.add_trace(go.Scatter(
            x=[u["x"], HX], y=[u["y"], HY], mode="lines",
            line=dict(color="rgba(120,120,120,0.18)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Data-upload pulses (alert/warn units → hub, orange dashed) ────────────
    _legend_upload_done = False
    for u in units:
        if u["status"] in ("alert", "warn"):
            mx, my = (u["x"] + HX) / 2, (u["y"] + HY) / 2
            fig.add_trace(go.Scatter(
                x=[u["x"], mx], y=[u["y"], my], mode="lines",
                line=dict(color="rgba(243,156,18,0.75)", width=2.5, dash="dot"),
                name="Data uploading ↑" if not _legend_upload_done else None,
                showlegend=not _legend_upload_done,
                hoverinfo="skip",
            ))
            _legend_upload_done = True

    # ── Model-push lines (ok units ← hub, purple) ─────────────────────────────
    _legend_model_done = False
    for u in units:
        if u["status"] == "ok":
            mx, my = (u["x"] + HX) / 2, (u["y"] + HY) / 2
            fig.add_trace(go.Scatter(
                x=[HX, mx], y=[HY, my], mode="lines",
                line=dict(color="rgba(192,132,252,0.55)", width=2.5),
                name="Model push ↓" if not _legend_model_done else None,
                showlegend=not _legend_model_done,
                hoverinfo="skip",
            ))
            _legend_model_done = True

    # ── Unit nodes ────────────────────────────────────────────────────────────
    for u in units:
        label_pos = "top center" if u["y"] >= 0 else "bottom center"
        fig.add_trace(go.Scatter(
            x=[u["x"]], y=[u["y"]],
            mode="markers+text",
            marker=dict(
                size=32, color=_STATUS_COLOR[u["status"]], opacity=0.92,
                line=dict(color="white", width=2),
            ),
            text=[f"XENTA-{u['id']}<br><span style='font-size:9px'>{u['env']}<br>PSI {u['psi']:.2f} · {u['model']}</span>"],
            textposition=label_pos,
            textfont=dict(color="white", size=9),
            hovertemplate=(
                f"<b>XENTA-{u['id']}</b><br>"
                f"Environment: {u['env']}<br>"
                f"PSI: {u['psi']}<br>"
                f"Model: {u['model']}<br>"
                f"Detections today: {u['detections']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # ── Hub node ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[HX], y=[HY],
        mode="markers",
        marker=dict(size=64, color="#f39c12", opacity=0.95,
                    line=dict(color="white", width=3)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_annotation(
        x=HX, y=HY, text="Training Hub<br><b>MLflow · DVC</b><br>Model v8 live",
        showarrow=False, font=dict(color="white", size=9), align="center",
    )

    # ── Drone detection markers ───────────────────────────────────────────────
    for d in _DETECTIONS:
        u = units[d["unit_idx"]]
        dx, dy = d["dx"], d["dy"]
        px, py = u["x"] + dx, u["y"] + dy
        color = "#00ccff" if d["known"] else "#ffdd44"
        suffix = f" {d['conf']:.2f}" if d["known"] else ""
        fig.add_trace(go.Scatter(
            x=[px], y=[py],
            mode="markers+text",
            marker=dict(size=13, color=color, symbol="triangle-up",
                        line=dict(color="white", width=1)),
            text=[d["label"] + suffix],
            textposition="top right" if dx > 0 else "top left",
            textfont=dict(color=color, size=8),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Legend entries for drone types ────────────────────────────────────────
    for color, label in [("#00ccff", "Known drone classified"), ("#ffdd44", "Unknown / queued")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="triangle-up"),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-4.8, 4.8]),
        yaxis=dict(visible=False, range=[-4.8, 4.8], scaleanchor="x"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", font=dict(color="white", size=10),
            x=0.01, y=0.01, xanchor="left", yanchor="bottom",
        ),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


def _event_log(seed: int) -> list[tuple]:
    """Return a shuffled subset of the event log."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(_EVENTS))[:10]
    return [_EVENTS[i] for i in sorted(idx)]


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown("## Proposed: Continuous Fleet Learning")
    st.markdown(
        "Each XENTA unit in the field sees a slightly different world — different coastlines, "
        "different drone types, different clutter. The idea behind this architecture is that "
        "the fleet's collective experience becomes the training set: data flows inward to a "
        "central hub, models improve, and updated ONNX artifacts push back out. "
        "Every new drone type one unit encounters eventually makes the whole fleet smarter."
    )
    st.info(
        "**Proposed design** — the visualisation below shows what this loop would look like in operation. "
        "Unit positions are illustrative; colors show the current PSI status from the simulated scenarios "
        "in the Agent tab. Drone detections are simulated examples of the classification output."
    )

    st.divider()

    # ── Fleet diagram ─────────────────────────────────────────────────────────
    st.markdown("### The learning loop")

    col_diag, col_key = st.columns([3, 1])

    with col_diag:
        st.plotly_chart(_fleet_fig(), use_container_width=True)

    with col_key:
        st.markdown("**Node colour**")
        for status, color, label in [
            ("ok",    "#2ecc71", "Stable (PSI < 0.10)"),
            ("warn",  "#f39c12", "Warn (PSI 0.10–0.20)"),
            ("alert", "#e74c3c", "Alert (PSI > 0.20)"),
        ]:
            st.markdown(
                f"<span style='color:{color};font-size:18px'>●</span> {label}",
                unsafe_allow_html=True,
            )

        st.markdown("**Flow lines**")
        st.markdown(
            "<span style='color:#f39c12'>▶ ···</span> Data upload  \n"
            "<span style='color:#c084fc'>▶ ───</span> Model push  \n"
            "<span style='color:#888'>▶ ───</span> Idle connection",
            unsafe_allow_html=True,
        )

        st.markdown("**Detections**")
        st.markdown(
            "<span style='color:#00ccff'>▲</span> Known — classified  \n"
            "<span style='color:#ffdd44'>▲</span> Unknown — queued",
            unsafe_allow_html=True,
        )

        st.markdown("**The hub**")
        st.markdown(
            "Central training node: "
            "MLflow tracks every run, "
            "DVC versions the data, "
            "accuracy gate blocks bad models, "
            "ONNX artifacts push OTA."
        )

    st.divider()

    # ── Why it compounds ──────────────────────────────────────────────────────
    st.markdown("### Why it gets smarter over time")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**More units → more diversity**")
        st.markdown(
            "Each deployment environment contributes a distribution of signals the others "
            "haven't seen: coastal sea clutter, urban multipath, border-terrain reflections. "
            "A model trained on the union of these distributions is more robust than one "
            "trained on any single environment."
        )
    with col_b:
        st.markdown("**New drone types get absorbed**")
        st.markdown(
            "When a unit encounters an unknown airframe, it queues the detection for "
            "test-range labelling. Once the new type is collected and labelled, "
            "the pipeline retrains and pushes a model that recognises it fleet-wide — "
            "not just to the unit that first saw it."
        )
    with col_c:
        st.markdown("**Hardware variance gets averaged out**")
        st.markdown(
            "Pooling data across many units with different manufacturing tolerances "
            "forces the model to learn features that generalise across hardware — "
            "the learned representation becomes less dependent on any single unit's "
            "specific antenna or ADC characteristics."
        )

    st.divider()

    # ── Live event feed ───────────────────────────────────────────────────────
    st.markdown("### Simulated activity feed")
    st.caption("A sample of what the monitoring backend would log during normal fleet operation.")

    if "fleet_seed" not in st.session_state:
        st.session_state["fleet_seed"] = 0

    if st.button("↺  Refresh events", key="fleet_refresh"):
        st.session_state["fleet_seed"] += 1

    events = _event_log(st.session_state["fleet_seed"])

    rows = ""
    for icon, source, msg in events:
        source_color = (
            "#f39c12" if source == "Hub"
            else "#2ecc71" if source == "Agent"
            else "#c084fc"
        )
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;font-size:16px'>{icon}</td>"
            f"<td style='padding:4px 8px;color:{source_color};font-weight:600;"
            f"white-space:nowrap;font-size:12px'>{source}</td>"
            f"<td style='padding:4px 8px;color:#ddd;font-size:12px'>{msg}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='background:rgba(0,0,0,0.3);border-radius:8px;padding:8px'>"
        f"<table style='width:100%;border-collapse:collapse'>{rows}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── The compounding argument ───────────────────────────────────────────────
    st.success(
        "**The central argument:** a XENTA fleet running this pipeline would not stay at the "
        "detection performance of the day it was deployed. Every new environment, every new "
        "drone type, every PSI alert that triggers a test-range session makes the next model "
        "better — for the whole fleet, not just the unit that saw the edge case. "
        "Classical signal processing has no equivalent mechanism: its operating parameters "
        "are fixed at algorithm design time."
    )
