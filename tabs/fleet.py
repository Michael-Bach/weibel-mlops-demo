"""
tabs/fleet.py — Fleet learning loop visualisation.

Illustrates the proposed continuous learning cycle: XENTA units spread across
deployment environments, streaming detection data to a central training hub,
receiving updated ONNX models, and collectively classifying an ever-growing
catalogue of drone types.
"""

import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Drone knowledge base ─────────────────────────────────────────────────────

_CATALOGUE = [
    {"type": "DJI Mavic 3",    "first_seen": "2025-11-14", "seqs": 1200, "source": "syn + range", "confidence": 0.96, "status": "deployed"},
    {"type": "DJI Spark",      "first_seen": "2025-12-03", "seqs":  980, "source": "syn + range", "confidence": 0.94, "status": "deployed"},
    {"type": "DJI Matrice 4",  "first_seen": "2026-04-11", "seqs":  640, "source": "syn + range", "confidence": 0.92, "status": "deployed"},
    {"type": "Autel EVO II",   "first_seen": "2026-01-18", "seqs":  312, "source": "syn + range", "confidence": 0.88, "status": "deployed"},
    {"type": "FPV-250",        "first_seen": "2026-03-02", "seqs":   80, "source": "range only",  "confidence": 0.81, "status": "deployed",
     "note": "⚠ limited coverage — syn expansion recommended"},
    {"type": "Unknown-A",      "first_seen": "2026-05-21", "seqs":    0, "source": "—",           "confidence": None, "status": "queued",
     "note": "Detected by C3-0042 — awaiting test-range session"},
]

_STATUS_BADGE = {
    "deployed":  ("<span style='background:#2ecc71;color:#000;padding:2px 8px;"
                  "border-radius:4px;font-size:11px;font-weight:600'>deployed</span>"),
    "queued":    ("<span style='background:#e74c3c;color:#fff;padding:2px 8px;"
                  "border-radius:4px;font-size:11px;font-weight:600'>queued</span>"),
    "scheduled": ("<span style='background:#f39c12;color:#000;padding:2px 8px;"
                  "border-radius:4px;font-size:11px;font-weight:600'>range scheduled</span>"),
    "expanding": ("<span style='background:#3498db;color:#fff;padding:2px 8px;"
                  "border-radius:4px;font-size:11px;font-weight:600'>syn expanding</span>"),
    "deploying": ("<span style='background:#9b59b6;color:#fff;padding:2px 8px;"
                  "border-radius:4px;font-size:11px;font-weight:600'>deploying</span>"),
}

def _conf_color(c):
    if c is None:   return "#888"
    if c >= 0.90:   return "#2ecc71"
    if c >= 0.80:   return "#f39c12"
    return "#e74c3c"

def _coverage_bar(seqs: int, max_seqs: int = 1200) -> str:
    pct = min(100, int(seqs / max_seqs * 100))
    color = "#2ecc71" if pct >= 70 else "#f39c12" if pct >= 20 else "#e74c3c"
    return (
        f"<div style='background:#333;border-radius:3px;height:8px;width:100px;display:inline-block'>"
        f"<div style='background:{color};height:8px;border-radius:3px;width:{pct}px'></div></div>"
        f" <span style='color:#aaa;font-size:11px'>{seqs}</span>"
    )

def _knowledge_base_section():
    """Render the drone knowledge base, augmented by any agent actions this session."""
    st.markdown("### Fleet knowledge base — known drone catalogue")
    st.caption(
        "The collective knowledge of the fleet: every drone type the model has been trained to recognise, "
        "how much training coverage it has, and its current fleet-wide detection confidence. "
        "Run the **🌊 Coastal drift** or **🚁 New drone** scenario in the Agent tab to see the agent "
        "update this catalogue in real time."
    )

    # Read any actions taken by the agent this session
    actions = st.session_state.get("agent_actions", [])
    scheduled_types  = set()
    expanding_types  = set()
    deploying_units  = []
    for a in actions:
        if a["tool"] == "schedule_test_range_session":
            for d in a["inputs"].get("drone_types", []):
                scheduled_types.add(d.lower())
        if a["tool"] == "request_synthetic_expansion":
            expanding_types.add(a["inputs"].get("drone_type", "").lower())
        if a["tool"] == "deploy_model":
            deploying_units.extend(a["inputs"].get("target_units", []))

    # Build augmented catalogue
    rows = ""
    for entry in _CATALOGUE:
        drone_lc = entry["type"].lower()
        status = entry["status"]
        # Override status if agent has acted on this type
        if any(t in drone_lc or drone_lc in t for t in expanding_types):
            status = "expanding"
        elif any(t in drone_lc or drone_lc in t for t in scheduled_types):
            status = "scheduled"
        # Also flag if it's the "Unknown-A" and either action fired
        if entry["status"] == "queued" and (scheduled_types or expanding_types):
            status = "expanding" if expanding_types else "scheduled"

        conf = entry["confidence"]
        conf_str = (
            f"<span style='color:{_conf_color(conf)};font-weight:600'>{conf:.2f}</span>"
            if conf is not None else "<span style='color:#888'>—</span>"
        )
        note = entry.get("note", "")
        badge = _STATUS_BADGE.get(status, _STATUS_BADGE["queued"])

        rows += (
            f"<tr style='border-bottom:1px solid #333'>"
            f"<td style='padding:7px 10px;color:#eee;font-size:13px'>{entry['type']}</td>"
            f"<td style='padding:7px 10px;color:#aaa;font-size:12px'>{entry['first_seen']}</td>"
            f"<td style='padding:7px 10px'>{_coverage_bar(entry['seqs'])}</td>"
            f"<td style='padding:7px 10px;color:#aaa;font-size:12px'>{entry['source']}</td>"
            f"<td style='padding:7px 10px;text-align:center'>{conf_str}</td>"
            f"<td style='padding:7px 10px'>{badge}</td>"
            f"<td style='padding:7px 10px;color:#888;font-size:11px;font-style:italic'>{note}</td>"
            f"</tr>"
        )

    header = (
        "<tr style='border-bottom:2px solid #555'>"
        + "".join(
            f"<th style='padding:7px 10px;color:#aaa;font-size:12px;text-align:left'>{h}</th>"
            for h in ["Drone type", "First seen", "Training seqs", "Source", "Confidence", "Status", "Notes"]
        )
        + "</tr>"
    )
    st.markdown(
        f"<div style='background:rgba(0,0,0,0.3);border-radius:8px;padding:4px;overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse'>{header}{rows}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Show agent action log if any actions have been taken
    if actions:
        st.markdown("**Agent actions this session**")
        for a in actions:
            icon = {"schedule_test_range_session": "🗓", "request_synthetic_expansion": "🔬",
                    "deploy_model": "🚀"}.get(a["tool"], "⚙")
            st.markdown(
                f"<div style='background:rgba(243,156,18,0.1);border-left:3px solid #f39c12;"
                f"padding:6px 12px;margin:4px 0;border-radius:0 4px 4px 0;font-size:12px;color:#ddd'>"
                f"{icon} <b style='color:#f39c12'>{a['tool']}</b> · {a['timestamp']} · "
                f"{json.dumps(a['result'].get('session_id') or a['result'].get('job_id') or a['result'].get('deployment_id',''))}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No agent actions yet — run a scenario in the 🧠 Agent tab to see this update.")

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


def _triad_fig() -> go.Figure:
    """Triangle diagram: three data sources converging on the training hub."""

    # Vertex positions (equilateral triangle, hub at centroid)
    SX,  SY  =  0.0,  2.6   # Synthetic — top
    TX,  TY  = -2.3, -1.3   # Test range — bottom-left
    FX,  FY  =  2.3, -1.3   # Fleet live  — bottom-right
    HX,  HY  =  0.0,  0.0   # Hub — centre

    C_SYN  = "#3498db"   # blue
    C_RNG  = "#f39c12"   # gold
    C_FLT  = "#9b59b6"   # purple
    C_HUB  = "#ecf0f1"   # light

    fig = go.Figure()

    # Triangle outline (faint background)
    fig.add_trace(go.Scatter(
        x=[SX, TX, FX, SX], y=[SY, TY, FY, SY],
        mode="lines",
        line=dict(color="rgba(200,200,200,0.10)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Arrows: each vertex → hub
    for (ax, ay, color) in [(SX, SY, C_SYN), (TX, TY, C_RNG), (FX, FY, C_FLT)]:
        # Draw line from vertex to hub, stopping short so the node marker shows
        t = 0.72  # stop at 72% of the way (leaves room for hub marker)
        ex = ax + t * (HX - ax)
        ey = ay + t * (HY - ay)
        fig.add_annotation(
            x=ex, y=ey, ax=ax, ay=ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=2.5,
            arrowcolor=color, arrowsize=1.3,
        )

    # Arrow: hub → fleet (downward, deployment)
    fig.add_annotation(
        x=HX, y=HY - 1.8, ax=HX, ay=HY - 0.55,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowwidth=2.5,
        arrowcolor=C_HUB, arrowsize=1.3,
    )
    fig.add_annotation(
        x=HX, y=HY - 2.1,
        text="<b>XENTA Fleet</b><br><span style='font-size:10px'>ONNX v8 · OTA push</span>",
        showarrow=False, font=dict(color=C_HUB, size=11), align="center",
    )

    # Vertex nodes
    for (vx, vy, color, label, sublabel) in [
        (SX, SY, C_SYN,
         "🔵  Synthetic Runs",
         "Physics simulator · continuous<br>Parameterisable · reproducible<br>Unlimited scale"),
        (TX, TY, C_RNG,
         "🟡  Test Range",
         "Instrumentation radar TSPI<br>Sub-cm ground truth labels<br>New drone type capture"),
        (FX, FY, C_FLT,
         "🟣  Fleet Live Data",
         "All environments simultaneously<br>PSI drift detection<br>Experience replay buffer"),
    ]:
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy], mode="markers",
            marker=dict(size=52, color=color, opacity=0.92,
                        line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip",
        ))
        # Label above/below node
        lpos = "top center" if vy > 0 else "bottom center"
        fig.add_annotation(
            x=vx, y=vy,
            text=f"<b>{label}</b>",
            showarrow=False, font=dict(color="white", size=11), align="center",
        )
        # Sub-label offset outward from hub
        ox = (vx - HX) * 0.55
        oy = (vy - HY) * 0.55
        fig.add_annotation(
            x=vx + ox, y=vy + oy,
            text=f"<span style='font-size:9px;color:#ccc'>{sublabel}</span>",
            showarrow=False, align="center",
        )

    # Hub node
    fig.add_trace(go.Scatter(
        x=[HX], y=[HY], mode="markers",
        marker=dict(size=60, color="#2c3e50", opacity=0.97,
                    line=dict(color=C_HUB, width=3)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_annotation(
        x=HX, y=HY,
        text="<b>Training Hub</b><br><span style='font-size:9px'>MLflow · DVC · Accuracy gate</span>",
        showarrow=False, font=dict(color="white", size=10), align="center",
    )

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-3.8, 3.8]),
        yaxis=dict(visible=False, range=[-3.2, 3.8], scaleanchor="x"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown("## Proposed: Continuous Fleet Learning")
    st.caption("*Three data sources feed one training hub — every new deployment makes the whole fleet smarter.*")
    st.markdown(
        "The proposed training pipeline draws from three complementary data sources simultaneously. "
        "No single source is sufficient on its own: synthetic data covers the full parameter space "
        "but misses real hardware; test-range data provides precision labels but is expensive to collect; "
        "fleet live data reflects the true operational distribution but lacks ground-truth labels. "
        "Together they form a self-reinforcing triad — each source compensating for the others' gaps."
    )
    st.info(
        "**Proposed design** — the diagram and sections below describe the intended architecture. "
        "The training run and model artifacts in the Pipeline tab are real; the fleet and monitoring "
        "loop are the proposed additions this architecture would enable."
    )

    st.divider()

    # ── Triad diagram ─────────────────────────────────────────────────────────
    st.markdown("### Three data sources — one training hub")
    st.plotly_chart(_triad_fig(), use_container_width=True)

    st.divider()

    # ── Three source deep-dives ───────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🔵 Synthetic training runs")
        st.markdown(
            "The physics simulator runs continuously, producing PPI sequences across the full "
            "operational envelope — SNR from −20 to +40 dB, all aspect angles, random velocities, "
            "every clutter type in the parameter library. "
            "This is the backbone of the training set: unlimited scale, fully reproducible via DVC seed, "
            "and the only source that can guarantee coverage of rare but important conditions "
            "(e.g. head-on approach at −15 dB SNR in sea-state 4 clutter)."
        )
        st.markdown(
            "**Synthetic multiplication of new captures:** when a new drone type is confirmed "
            "at the test range, the simulator extracts its physical signature — rotor frequency, "
            "RCS, micro-Doppler bandwidth — and generates thousands of synthetic variants. "
            "A handful of real captures becomes a full training distribution."
        )
        st.code(
            "synthetic_batch = generator.sample(\n"
            "    drone_params  = extracted_signature,\n"
            "    n_sequences   = 2000,       # 40–100× real captures\n"
            "    snr_range_db  = (-20, 40),\n"
            "    aspect_angles = 'uniform',\n"
            "    clutter_types = ['rayleigh',\n"
            "                     'sea-state-3',\n"
            "                     'urban'],\n"
            "    seed          = run_id,     # DVC reproducible\n"
            ")",
            language="python",
        )

    with col_b:
        st.markdown("#### 🟡 Instrumentation test range")
        st.markdown(
            "Weibel's own MFCW instrumentation radars — the same systems deployed at missile "
            "test ranges globally — act as the precision labelling engine. "
            "A cooperative target (drone) flies through the range while the instrumentation radar "
            "records its TSPI (Time Space Position Information) at sub-centimetre accuracy. "
            "The co-located XENTA unit records the same scene through its own hardware. "
            "Label fusion pairs the two: XENTA signal as feature, TSPI position as label."
        )
        st.markdown(
            "**This is where instrument error gets absorbed.** "
            "Because the label comes from the instrumentation radar — not the XENTA — "
            "the model must learn to predict the true position from a distorted measurement. "
            "Hardware imperfections (phase noise, ADC non-linearity, gain drift) are baked "
            "into the input but not the label. The model corrects for them automatically."
        )
        st.markdown(
            "**New drone types:** when a unit flags an unknown airframe, the test range "
            "schedules a dedicated session. The resulting labelled sequences seed the "
            "synthetic multiplication pipeline and are added to the experience buffer."
        )

    with col_c:
        st.markdown("#### 🟣 Fleet live data")
        st.markdown(
            "Every deployed XENTA unit contributes to the training distribution continuously. "
            "Confirmed detections (high-confidence classifications) are tagged with their "
            "operating environment and uploaded to the hub. "
            "This is the source the other two cannot replicate: the actual distribution of "
            "real signals the model will face in production — coastal sea clutter, urban multipath, "
            "border-terrain reflections, hardware aging, weather effects."
        )
        st.markdown(
            "**PSI drift detection** monitors when the incoming distribution diverges from "
            "the training reference. A fleet-wide simultaneous drift signals a new drone type; "
            "a localised drift on units sharing an environment signals a clutter change. "
            "Both trigger the appropriate response: test-range session or synthetic expansion."
        )
        st.markdown(
            "**Experience replay:** confirmed detections are stored in a buffer and replayed "
            "alongside fresh synthetic data throughout training. The model retains memory of "
            "rare real events even as the synthetic dataset grows around them."
        )

    st.divider()

    st.divider()

    # ── Drone knowledge base ──────────────────────────────────────────────────
    _knowledge_base_section()

    st.divider()

    # ── Variation table ───────────────────────────────────────────────────────
    st.markdown("### What the synthetic source covers that the others cannot")
    st.markdown("""
| Condition | Synthetic | Test Range | Fleet Live |
|---|:---:|:---:|:---:|
| Full SNR sweep (−20 to +40 dB) | ✓ unlimited | ✓ limited sessions | ✗ uncontrolled |
| All aspect angles | ✓ | partial | ✗ |
| Known clutter statistics | ✓ parameterised | ✓ controlled | ✗ unknown |
| Real hardware imperfections | ✗ | ✓ per-unit | ✓ in-situ |
| New drone types | ✗ | ✓ gold standard | ✓ discovered |
| Operational clutter distribution | ✗ approximated | ✗ controlled env | ✓ real |
| Unlimited scale | ✓ | ✗ expensive | ✗ rate-limited |
| Reproducible (DVC seed) | ✓ | partial (TSPI log) | ✗ |
    """)

    st.divider()

    # ── Fleet ring diagram ────────────────────────────────────────────────────
    st.markdown("### Fleet live layer in detail")
    st.caption("Each node is a deployed XENTA unit. Colours show current PSI status.")

    col_diag, col_key = st.columns([3, 1])
    with col_diag:
        st.plotly_chart(_fleet_fig(), use_container_width=True)
    with col_key:
        st.markdown("**Node colour**")
        for color, label in [
            ("#2ecc71", "Stable  (PSI < 0.10)"),
            ("#f39c12", "Warn    (PSI 0.10–0.20)"),
            ("#e74c3c", "Alert   (PSI > 0.20)"),
        ]:
            st.markdown(
                f"<span style='color:{color};font-size:18px'>●</span> {label}",
                unsafe_allow_html=True,
            )
        st.markdown("**Flow lines**")
        st.markdown(
            "<span style='color:#f39c12'>▶ ···</span> Data upload  \n"
            "<span style='color:#c084fc'>▶ ───</span> Model push  \n"
            "<span style='color:#888'>──────</span> Idle",
            unsafe_allow_html=True,
        )
        st.markdown("**Detections**")
        st.markdown(
            "<span style='color:#00ccff'>▲</span> Known — classified  \n"
            "<span style='color:#ffdd44'>▲</span> Unknown — queued",
            unsafe_allow_html=True,
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
