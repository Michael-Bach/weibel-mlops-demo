"""
tabs/agent.py — Agentic MLOps tab.

A Claude-powered advisor that interprets PSI drift alerts, queries MLflow
experiment history, and recommends a retraining action. Demonstrates agentic
AI (tool use + multi-step reasoning) integrated with the pipeline in the
previous tab.
"""

import json
import os
import traceback
from datetime import datetime

import streamlit as st

try:
    import anthropic as _anthropic_mod
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# ── Simulated data backing the tools ─────────────────────────────────────────

_SCENARIOS = {
    "stable": {
        "label": "✅  Stable fleet",
        "description": (
            "Routine scheduled check. All units have been operating in known environments "
            "for 7 days. No anomalies reported by the monitoring backend."
        ),
        "fleet": [
            {"unit_id": "XENTA-C3-0041", "model_version": "v7", "psi": 0.04, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0042", "model_version": "v7", "psi": 0.07, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0051", "model_version": "v7", "psi": 0.06, "status": "ok",    "env": "coastal"},
            {"unit_id": "XENTA-C1-0012", "model_version": "v7", "psi": 0.03, "status": "ok",    "env": "urban"},
        ],
    },
    "coastal_drift": {
        "label": "🌊  Coastal drift alert",
        "description": (
            "Three coastal XENTA-C3 units entered PSI alert state after 48 h of high sea state. "
            "Inland and urban units remain nominal. The fleet monitoring backend has filed an "
            "automatic retraining review ticket."
        ),
        "fleet": [
            {"unit_id": "XENTA-C3-0041", "model_version": "v7", "psi": 0.05, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0042", "model_version": "v7", "psi": 0.34, "status": "alert", "env": "coastal-high-sea"},
            {"unit_id": "XENTA-C3-0051", "model_version": "v7", "psi": 0.28, "status": "alert", "env": "coastal-high-sea"},
            {"unit_id": "XENTA-C3-0089", "model_version": "v7", "psi": 0.22, "status": "alert", "env": "coastal-high-sea"},
            {"unit_id": "XENTA-C1-0012", "model_version": "v7", "psi": 0.08, "status": "ok",    "env": "urban"},
        ],
    },
    "new_drone": {
        "label": "🚁  New drone type",
        "description": (
            "Fleet-wide moderate PSI drift appearing simultaneously across all environments. "
            "PSI is driven by the spectral centroid feature rather than signal energy — "
            "consistent with a new drone airframe whose rotor micro-Doppler signature "
            "falls outside the training distribution."
        ),
        "fleet": [
            {"unit_id": "XENTA-C3-0041", "model_version": "v7", "psi": 0.17, "status": "warn", "env": "inland"},
            {"unit_id": "XENTA-C3-0042", "model_version": "v7", "psi": 0.19, "status": "warn", "env": "coastal"},
            {"unit_id": "XENTA-C3-0051", "model_version": "v7", "psi": 0.14, "status": "warn", "env": "coastal"},
            {"unit_id": "XENTA-C1-0012", "model_version": "v7", "psi": 0.16, "status": "warn", "env": "urban"},
            {"unit_id": "XENTA-C3-0089", "model_version": "v7", "psi": 0.13, "status": "warn", "env": "border"},
        ],
    },
    "hw_fault": {
        "label": "⚠️  Suspected hardware fault",
        "description": (
            "XENTA-C3-0042 shows PSI = 0.67 — far above the alert threshold and well above "
            "what any environmental shift would produce. All other units are nominal. "
            "The unit was recently moved to a new site and physically re-mounted."
        ),
        "fleet": [
            {"unit_id": "XENTA-C3-0041", "model_version": "v7", "psi": 0.05, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0042", "model_version": "v7", "psi": 0.67, "status": "alert", "env": "coastal"},
            {"unit_id": "XENTA-C3-0051", "model_version": "v7", "psi": 0.06, "status": "ok",    "env": "coastal"},
            {"unit_id": "XENTA-C1-0012", "model_version": "v7", "psi": 0.04, "status": "ok",    "env": "urban"},
        ],
    },
}

_MLF_RUNS = [
    {
        "run_id": "7f03b68e8ac5", "run_name": "adventurous-loon-618",
        "model": "ConvGRU", "best_val_f1": 0.356, "n_params": 5694,
        "trained_on": "synthetic + range_v1 (800+312 seqs)", "registry": "Production",
    },
    {
        "run_id": "3f126326dd21", "run_name": "bold-hawk-201",
        "model": "ConvGRU", "best_val_f1": 0.341, "n_params": 5694,
        "trained_on": "synthetic only (800 seqs)", "registry": "Archived",
    },
    {
        "run_id": "e633e618604d", "run_name": "calm-jay-077",
        "model": "CNN", "best_val_f1": 0.307, "n_params": 19073,
        "trained_on": "synthetic only (800 seqs)", "registry": "Archived",
    },
]

# ── Tool definitions (Anthropic tool_use schema) ──────────────────────────────

_TOOLS = [
    {
        "name": "get_fleet_status",
        "description": (
            "Returns the current deployment status of all XENTA fleet units: "
            "unit ID, current model version, latest PSI score, alert status, "
            "and operating environment. Always call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_psi_report",
        "description": (
            "Returns the detailed PSI drift report for a specific XENTA unit: "
            "psi_energy, psi_spectral_centroid, psi_overall, status, and distribution stats. "
            "Call this for every unit in warn or alert state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "XENTA unit serial number, e.g. 'XENTA-C3-0042'",
                }
            },
            "required": ["unit_id"],
        },
    },
    {
        "name": "query_mlflow_runs",
        "description": (
            "Returns recent MLflow training runs: model type, best val F1, training data used, "
            "and registry stage. Check this before recommending retraining — "
            "a better model may already exist in Staging."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_runs": {
                    "type": "integer",
                    "description": "Number of recent runs to return (default 5)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_model_registry",
        "description": (
            "Returns the MLflow model registry state: current Production version, "
            "any Staging candidate, promotion timestamp, accuracy gate thresholds, "
            "and what data it was trained on."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

_SYSTEM = """You are an MLOps advisor for the Weibel XENTA counter-UAS radar fleet.

Your task:
1. Call get_fleet_status to understand the fleet-wide picture.
2. For any unit in warn or alert state, call get_psi_report to get the breakdown.
3. Call query_mlflow_runs and get_model_registry to understand the current model before recommending retraining.
4. Give a clear, specific, justified recommendation — one of:
   - No action (explain why the drift is within tolerance)
   - Increase monitoring cadence (for warn-level drift)
   - Schedule test-range data collection (specify which units, environment, drone types)
   - Trigger retraining (specify what new data is needed and why)
   - Investigate hardware fault (when PSI is extreme and isolated to one unit — retraining will not help)
   - Rollback model (when a regression is suspected)

Key domain knowledge:
- PSI < 0.10: stable. PSI 0.10–0.20: warn. PSI > 0.20: alert. PSI > 0.50: likely hardware, not data.
- Fleet-wide simultaneous drift across different environments → new drone type (centroid shift).
- Localised drift on units sharing an environment → environmental/clutter change (energy shift).
- Single-unit extreme drift while others are normal → hardware fault, not a retraining problem.
- Before recommending retraining, confirm no Staging model already covers the new distribution.

Be concise. Cite specific PSI numbers, unit IDs, and model versions in your recommendation."""


# ── Tool execution ─────────────────────────────────────────────────────────────

def _run_tool(name: str, inputs: dict, scenario_key: str) -> str:
    fleet = _SCENARIOS[scenario_key]["fleet"]

    if name == "get_fleet_status":
        counts = {"alert": 0, "warn": 0, "ok": 0}
        for u in fleet:
            counts[u["status"]] += 1
        return json.dumps({
            "units": fleet,
            "summary": counts,
            "active_model": "radar-classifier v7 (ConvGRU, 5694 params)",
            "checked_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, indent=2)

    if name == "get_psi_report":
        uid = inputs.get("unit_id", "")
        unit = next((u for u in fleet if u["unit_id"] == uid), None)
        if unit is None:
            return json.dumps({"error": f"Unit '{uid}' not in fleet"})
        psi = unit["psi"]
        return json.dumps({
            "unit_id": uid,
            "environment": unit["env"],
            "psi_energy":            round(psi * 0.82 + 0.008, 3),
            "psi_spectral_centroid": round(psi * 0.60 + 0.005, 3),
            "psi_overall":           psi,
            "status":                unit["status"],
            "thresholds":            {"warn": 0.10, "alert": 0.20},
            "sample_counts":         {"reference": 2000, "current": 490},
            "distribution_stats": {
                "ref_energy_mean":    1.000,
                "cur_energy_mean":    round(1.0 + psi * 0.55, 3),
                "ref_centroid_mean":  0.300,
                "cur_centroid_mean":  round(0.30 + psi * 0.22, 3),
            },
        }, indent=2)

    if name == "query_mlflow_runs":
        n = inputs.get("n_runs", 5)
        return json.dumps({"experiment": "recurrent-radar-detector", "runs": _MLF_RUNS[:n]}, indent=2)

    if name == "get_model_registry":
        return json.dumps({
            "model_name": "radar-classifier",
            "Production": {
                "version": "v7", "run_id": "7f03b68e8ac5",
                "best_val_f1": 0.356,
                "trained_on": "synthetic + range_v1 (800 synthetic + 312 real XENTA seqs, inland + light-coastal clutter)",
                "promoted_at": "2026-05-20T09:14:00Z",
            },
            "Staging": None,
            "accuracy_gate": {"staging": 0.90, "production": 0.95},
        }, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Agent loop ─────────────────────────────────────────────────────────────────

def _run_agent(scenario_key: str, api_key: str) -> list:
    """Run the agent to completion and return the full event trace."""
    import anthropic  # re-import in case top-level failed
    client = anthropic.Anthropic(api_key=api_key)

    scen = _SCENARIOS[scenario_key]
    messages = [{
        "role": "user",
        "content": f"Scenario: {scen['description']}\n\nPlease assess the fleet and give your recommendation.",
    }]

    trace = [("user", messages[0]["content"])]

    for _ in range(8):  # max turns
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            system=_SYSTEM,
            tools=_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace.append(("text", block.text.strip()))

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _run_tool(block.name, block.input, scenario_key)
                trace.append(("tool_call", block.name, block.input))
                trace.append(("tool_result", block.name, result))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return trace


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _card(border_color: str, label: str, body: str, mono: bool = False) -> str:
    font = "font-family:monospace;" if mono else ""
    return (
        f"<div style='border-left:3px solid {border_color};background:rgba(0,0,0,0.25);"
        f"padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0'>"
        f"<div style='color:{border_color};font-weight:600;font-size:12px;margin-bottom:4px'>{label}</div>"
        f"<div style='color:#ddd;font-size:13px;{font}white-space:pre-wrap'>{body}</div>"
        f"</div>"
    )


def _render_trace(trace: list):
    for event in trace:
        kind = event[0]
        if kind == "user":
            st.markdown(_card("#555", "📨 Scenario prompt", event[1]), unsafe_allow_html=True)
        elif kind == "text":
            st.markdown(_card("#2ecc71", "🤖 Agent", event[1]), unsafe_allow_html=True)
        elif kind == "tool_call":
            args = json.dumps(event[2], separators=(",", ":")) if event[2] else "()"
            st.markdown(_card("#c084fc", f"🔧 Tool call → {event[1]}", args, mono=True),
                        unsafe_allow_html=True)
        elif kind == "tool_result":
            preview = event[2]
            if len(preview) > 400:
                preview = preview[:400] + "\n  …"
            st.markdown(_card("#27ae60", f"📦 Result ← {event[1]}", preview, mono=True),
                        unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render():
    try:
        _render_impl()
    except Exception:
        st.error("Agentic MLOps tab encountered an error — details below:")
        st.code(traceback.format_exc())


def _render_impl():
    st.markdown("## Concept: Agentic MLOps Advisor")
    st.markdown(
        "The pipeline described in the previous tab produces alerts — PSI spikes, accuracy regressions, "
        "fleet drift reports. Someone or something still has to interpret them and decide what to do. "
        "This tab explores what it would look like to hand that reasoning step to an LLM-based agent: "
        "give it access to the same tools an on-call engineer would use, and ask it to "
        "produce a justified, specific recommendation rather than just a number."
    )
    st.info(
        "**Exploratory prototype** — the tools the agent calls are simulated backends backed by "
        "the same PSI and MLflow logic from the pipeline tab. The agent itself is real: "
        "Claude Haiku with tool use, running live against the Anthropic API. "
        "The four scenarios are designed so each requires a *different* response — "
        "distinguishing a data problem from an environmental shift from a hardware fault."
    )

    # ── Dependency check ──────────────────────────────────────────────────────
    if not _ANTHROPIC_OK:
        st.error(
            "The `anthropic` package is not installed in this environment. "
            "It was added to `requirements.txt` — **reboot the Streamlit Cloud app** "
            "to install it: Manage app → Reboot app."
        )
        return

    # ── API key ───────────────────────────────────────────────────────────────
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
    except Exception:
        api_key = ""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.warning(
            "No `ANTHROPIC_API_KEY` found. "
            "Add it to `.streamlit/secrets.toml` (local) or Streamlit Cloud → Settings → Secrets (deployed)."
        )
        st.code("# .streamlit/secrets.toml\nANTHROPIC_API_KEY = \"sk-ant-...\"", language="toml")
        return

    st.divider()

    # ── Scenario selector ─────────────────────────────────────────────────────
    st.markdown("### 1 — Choose a drift scenario")
    scenario_keys = list(_SCENARIOS.keys())
    scenario_labels = [_SCENARIOS[k]["label"] for k in scenario_keys]

    prev = st.session_state.get("agent_scenario", "coastal_drift")
    prev_idx = scenario_keys.index(prev) if prev in scenario_keys else 1

    chosen_label = st.radio(
        "Scenario",
        scenario_labels,
        index=prev_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected = scenario_keys[scenario_labels.index(chosen_label)]
    if selected != prev:
        st.session_state["agent_scenario"] = selected
        st.session_state.pop("agent_trace", None)
        st.rerun()

    st.caption(f"**{_SCENARIOS[selected]['description']}**")

    # ── Fleet snapshot ────────────────────────────────────────────────────────
    st.markdown("### 2 — Simulated fleet snapshot")
    fleet = _SCENARIOS[selected]["fleet"]
    status_icon = {"ok": "🟢", "warn": "🟡", "alert": "🔴"}
    # 2-column grid works on both desktop and mobile
    for i in range(0, len(fleet), 2):
        row_units = fleet[i:i + 2]
        cols = st.columns(len(row_units))
        for col, unit in zip(cols, row_units):
            col.metric(
                label=unit["unit_id"],
                value=f"PSI {unit['psi']:.2f}",
                delta=f"{status_icon[unit['status']]} {unit['status'].upper()}",
                delta_color="off",
            )
            col.caption(unit["env"])

    st.divider()

    # ── Agent section ─────────────────────────────────────────────────────────
    st.markdown("### 3 — Agent reasoning trace"
    "\n*Watch the agent call tools, observe results, and build its recommendation step by step.*")

    run_btn = st.button("▶  Run Agent", type="primary", key="agent_run_btn")

    if run_btn:
        st.session_state.pop("agent_trace", None)
        with st.spinner("Agent reasoning…"):
            trace = _run_agent(selected, api_key)
        st.session_state["agent_trace"] = trace
        st.rerun()

    if "agent_trace" in st.session_state:
        _render_trace(st.session_state["agent_trace"])
    else:
        st.info("Select a scenario above and click **▶ Run Agent** to see the agent in action.")

    st.divider()

    # ── Explainer ─────────────────────────────────────────────────────────────
    st.markdown("### Why I think this is worth building")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**The agentic patterns it would demonstrate**")
        st.markdown(
            "- **Tool use over static context**: the agent queries live state rather than "
            "summarising a fixed dashboard — it sees what an engineer would see\n"
            "- **Multi-step investigative reasoning**: fleet-wide check first → drill into "
            "specific units → inspect training history → form recommendation. "
            "That sequence matters; doing it in the wrong order wastes tool calls\n"
            "- **Domain-aware triage**: the system prompt encodes the key distinction — "
            "PSI 0.67 on a single unit after a physical remount is a hardware problem, "
            "not a retraining problem. An LLM with the right context can make that call\n"
            "- **Auditable reasoning**: every recommendation in the trace is grounded in "
            "specific numbers and unit IDs, not vague advice — making human review fast"
        )
    with col_r:
        st.markdown("**How it would slot into the proposed pipeline**")
        st.code(
            "# PSI monitor fires on cron schedule\n"
            "report = drift_detect.run_fleet_check()\n"
            "\n"
            "if report.has_alerts:\n"
            "    rec = mlops_agent.advise(\n"
            "        fleet_report=report,\n"
            "        tools=[\n"
            "            get_fleet_status,\n"
            "            get_psi_report,\n"
            "            query_mlflow_runs,\n"
            "            get_model_registry,\n"
            "        ],\n"
            "    )\n"
            "    # Draft posted to ops channel for human approval\n"
            "    ops.notify(rec)\n"
            "    # Human approves → retraining or rollback triggered\n"
            "    # Agent recommendation stored in audit log",
            language="python",
        )
        st.caption(
            "The agent would sit between the automated PSI alert and the human decision — "
            "reducing the cognitive load of on-call review without removing the human from the loop."
        )
