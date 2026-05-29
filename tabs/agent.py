"""
tabs/agent.py — Agentic MLOps tab.

Claude-powered advisor with:
  - Four preset drift scenarios + custom scenario builder
  - Tool inspector (see what each tool returns before running)
  - Live agent trace with tool call / result cards
  - Follow-up conversation after the initial recommendation
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

# ── Preset scenarios ──────────────────────────────────────────────────────────

_PRESETS = {
    "stable": {
        "label": "✅ Stable",
        "description": (
            "Routine scheduled check. All units operating in known environments for 7 days. "
            "No anomalies reported."
        ),
        "fleet": [
            {"unit_id": "XENTA-C3-0041", "model_version": "v7", "psi": 0.04, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0042", "model_version": "v7", "psi": 0.07, "status": "ok",    "env": "inland"},
            {"unit_id": "XENTA-C3-0051", "model_version": "v7", "psi": 0.06, "status": "ok",    "env": "coastal"},
            {"unit_id": "XENTA-C1-0012", "model_version": "v7", "psi": 0.03, "status": "ok",    "env": "urban"},
        ],
    },
    "coastal_drift": {
        "label": "🌊 Coastal drift",
        "description": (
            "Three coastal units entered PSI alert state after 48 h of high sea state. "
            "Inland and urban units remain nominal."
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
        "label": "🚁 New drone",
        "description": (
            "Fleet-wide moderate PSI drift across all environments simultaneously. "
            "Spectral centroid driven — consistent with an unknown drone airframe."
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
        "label": "⚠️ HW fault",
        "description": (
            "XENTA-C3-0042 shows PSI = 0.67 — extreme, isolated, after a physical remount. "
            "All other units are nominal."
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
    {"run_id": "7f03b68e8ac5", "run_name": "adventurous-loon-618",
     "model": "ConvGRU", "best_val_f1": 0.356, "n_params": 5694,
     "trained_on": "synthetic + range_v1 (800+312 seqs)", "registry": "Production"},
    {"run_id": "3f126326dd21", "run_name": "bold-hawk-201",
     "model": "ConvGRU", "best_val_f1": 0.341, "n_params": 5694,
     "trained_on": "synthetic only (800 seqs)", "registry": "Archived"},
    {"run_id": "e633e618604d", "run_name": "calm-jay-077",
     "model": "CNN", "best_val_f1": 0.307, "n_params": 19073,
     "trained_on": "synthetic only (800 seqs)", "registry": "Archived"},
]

# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "get_fleet_status",
        "description": (
            "Returns the current deployment status of all XENTA fleet units: "
            "unit ID, model version, PSI score, alert status, and operating environment. "
            "Always call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_psi_report",
        "description": (
            "Returns the detailed PSI drift report for a specific XENTA unit: "
            "psi_energy, psi_spectral_centroid, psi_overall, status, and distribution stats. "
            "Call for every unit in warn or alert state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "string",
                            "description": "XENTA unit serial number, e.g. 'XENTA-C3-0042'"}
            },
            "required": ["unit_id"],
        },
    },
    {
        "name": "query_mlflow_runs",
        "description": (
            "Returns recent MLflow training runs: model type, best val F1, training data, "
            "and registry stage. Check before recommending retraining."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_runs": {"type": "integer", "description": "Runs to return (default 5)"}
            },
            "required": [],
        },
    },
    {
        "name": "get_model_registry",
        "description": (
            "Returns the MLflow model registry: current Production version, "
            "any Staging candidate, accuracy gate thresholds, and training data provenance."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

_DEFAULT_SYSTEM = """You are an MLOps advisor for the Weibel XENTA counter-UAS radar fleet.

Your task:
1. Call get_fleet_status to understand the fleet-wide picture.
2. For any unit in warn or alert state, call get_psi_report to get the breakdown.
3. Call query_mlflow_runs and get_model_registry to understand the current model.
4. Give a clear, specific, justified recommendation — one of:
   - No action (drift within tolerance)
   - Increase monitoring cadence
   - Schedule test-range data collection (specify units, environment, drone types)
   - Trigger retraining (specify what new data is needed)
   - Investigate hardware fault (extreme isolated PSI — retraining will not help)
   - Rollback model (suspected regression)

Key domain knowledge:
- PSI < 0.10: stable. PSI 0.10–0.20: warn. PSI > 0.20: alert. PSI > 0.50: likely hardware.
- Fleet-wide simultaneous drift across environments → new drone type (centroid shift).
- Localised drift on units sharing an environment → environmental/clutter change (energy shift).
- Single-unit extreme drift while others are normal → hardware fault, not a data problem.
- Check registry before recommending retraining — a Staging model may already exist.

Be concise. Cite specific PSI numbers, unit IDs, and model versions."""


# ── Tool execution ────────────────────────────────────────────────────────────

def _run_tool(name: str, inputs: dict, fleet: list) -> str:
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
            return json.dumps({"error": f"Unit '{uid}' not found"})
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
                "ref_energy_mean":   1.000,
                "cur_energy_mean":   round(1.0 + psi * 0.55, 3),
                "ref_centroid_mean": 0.300,
                "cur_centroid_mean": round(0.30 + psi * 0.22, 3),
            },
        }, indent=2)

    if name == "query_mlflow_runs":
        n = inputs.get("n_runs", 5)
        return json.dumps({"experiment": "recurrent-radar-detector",
                           "runs": _MLF_RUNS[:n]}, indent=2)

    if name == "get_model_registry":
        return json.dumps({
            "model_name": "radar-classifier",
            "Production": {
                "version": "v7", "run_id": "7f03b68e8ac5",
                "best_val_f1": 0.356,
                "trained_on": "synthetic + range_v1 (800 synthetic + 312 real XENTA seqs, "
                              "inland + light-coastal clutter)",
                "promoted_at": "2026-05-20T09:14:00Z",
            },
            "Staging": None,
            "accuracy_gate": {"staging": 0.90, "production": 0.95},
        }, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Agent loop ────────────────────────────────────────────────────────────────

def _agent_loop(messages: list, fleet: list, api_key: str, system: str) -> tuple[list, list]:
    """Run one agent loop pass. Returns (new_trace_events, updated_messages)."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    trace = []

    for _ in range(8):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1400,
            system=system,
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
                result = _run_tool(block.name, block.input, fleet)
                trace.append(("tool_call", block.name, block.input))
                trace.append(("tool_result", block.name, result))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return trace, messages


def _run_agent(fleet: list, description: str, api_key: str, system: str) -> tuple[list, list]:
    prompt = f"Scenario: {description}\n\nPlease assess the fleet and give your recommendation."
    messages = [{"role": "user", "content": prompt}]
    trace = [("user", prompt)]
    new_events, messages = _agent_loop(messages, fleet, api_key, system)
    return trace + new_events, messages


def _run_followup(messages: list, question: str, fleet: list,
                  api_key: str, system: str) -> tuple[list, list]:
    messages = messages + [{"role": "user", "content": question}]
    new_events, messages = _agent_loop(messages, fleet, api_key, system)
    return new_events, messages


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _card(border_color: str, label: str, body: str, mono: bool = False) -> str:
    font = "font-family:monospace;" if mono else ""
    return (
        f"<div style='border-left:3px solid {border_color};background:rgba(0,0,0,0.22);"
        f"padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0'>"
        f"<div style='color:{border_color};font-weight:600;font-size:12px;margin-bottom:4px'>"
        f"{label}</div>"
        f"<div style='color:#ddd;font-size:13px;{font}white-space:pre-wrap'>{body}</div>"
        f"</div>"
    )


def _render_events(events: list):
    for ev in events:
        kind = ev[0]
        if kind == "user":
            st.markdown(_card("#555", "📨 Prompt", ev[1]), unsafe_allow_html=True)
        elif kind == "text":
            st.markdown(_card("#2ecc71", "🤖 Agent", ev[1]), unsafe_allow_html=True)
        elif kind == "tool_call":
            args = json.dumps(ev[2], separators=(",", ":")) if ev[2] else "()"
            st.markdown(_card("#c084fc", f"🔧 Tool call → {ev[1]}", args, mono=True),
                        unsafe_allow_html=True)
        elif kind == "tool_result":
            preview = ev[2]
            if len(preview) > 450:
                preview = preview[:450] + "\n  …"
            st.markdown(_card("#27ae60", f"📦 Result ← {ev[1]}", preview, mono=True),
                        unsafe_allow_html=True)
        elif kind == "followup_q":
            st.markdown(_card("#3498db", f"💬 Follow-up", ev[1]), unsafe_allow_html=True)


def _psi_status(psi: float) -> str:
    if psi >= 0.20:
        return "alert"
    if psi >= 0.10:
        return "warn"
    return "ok"


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    try:
        _render_impl()
    except Exception:
        st.error("Agent tab error:")
        st.code(traceback.format_exc())


def _render_impl():
    st.markdown("## Concept: Agentic MLOps Advisor")
    st.markdown(
        "The pipeline produces alerts — PSI spikes, drift reports, accuracy regressions. "
        "This explores handing the interpretation step to an LLM agent: give it the same "
        "tools an on-call engineer would use and ask it to produce a justified recommendation. "
        "Build your own scenario, inspect what the agent sees, ask follow-up questions."
    )
    st.info(
        "**Exploratory prototype** — tool backends are simulated. "
        "The agent is real: Claude Haiku with tool use, live against the Anthropic API."
    )

    if not _ANTHROPIC_OK:
        st.error("The `anthropic` package is not installed — reboot the Streamlit Cloud app.")
        return

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
    except Exception:
        api_key = ""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.warning("No `ANTHROPIC_API_KEY` found.")
        st.code("# .streamlit/secrets.toml\nANTHROPIC_API_KEY = \"sk-ant-...\"", language="toml")
        return

    st.divider()

    # ── 1. Scenario selector ──────────────────────────────────────────────────
    st.markdown("### 1 — Choose a scenario")

    preset_keys   = list(_PRESETS.keys())
    preset_labels = [_PRESETS[k]["label"] for k in preset_keys]
    all_labels    = preset_labels + ["🛠 Custom"]

    prev_sel = st.session_state.get("agent_scenario", "coastal_drift")
    if prev_sel == "custom":
        prev_idx = len(all_labels) - 1
    else:
        prev_idx = preset_keys.index(prev_sel) if prev_sel in preset_keys else 1

    chosen = st.radio("Scenario", all_labels, index=prev_idx,
                      horizontal=True, label_visibility="collapsed")

    selected = "custom" if chosen == "🛠 Custom" else preset_keys[all_labels.index(chosen)]

    if selected != prev_sel:
        st.session_state["agent_scenario"] = selected
        st.session_state.pop("agent_trace", None)
        st.session_state.pop("agent_messages", None)
        st.session_state.pop("agent_followups", None)
        st.rerun()

    # ── Custom scenario builder ───────────────────────────────────────────────
    if selected == "custom":
        with st.expander("🛠 Build your scenario", expanded=True):
            st.caption("Set PSI and environment for each unit, then run the agent.")
            n_units = st.slider("Number of units", 2, 5, 3, key="custom_n")
            envs = ["inland", "coastal", "coastal-high-sea", "urban", "border"]
            custom_fleet = []
            cols = st.columns(n_units)
            for i, col in enumerate(cols):
                with col:
                    uid = f"XENTA-C3-{100 + i:04d}"
                    psi = st.slider(f"{uid}\nPSI", 0.01, 0.70, [0.05, 0.18, 0.35, 0.08, 0.52][i % 5],
                                    step=0.01, key=f"custom_psi_{i}")
                    env = st.selectbox("Environment", envs,
                                       index=i % len(envs), key=f"custom_env_{i}")
                    custom_fleet.append({
                        "unit_id": uid, "model_version": "v7",
                        "psi": psi, "status": _psi_status(psi), "env": env,
                    })
            st.session_state["agent_custom_fleet"] = custom_fleet
            desc = (f"Custom scenario with {n_units} units. "
                    + ", ".join(f"{u['unit_id']} PSI={u['psi']:.2f} ({u['env']})"
                                for u in custom_fleet))
            st.session_state["agent_custom_desc"] = desc
    else:
        st.caption(f"**{_PRESETS[selected]['description']}**")

    # Resolve active fleet + description
    if selected == "custom":
        fleet = st.session_state.get("agent_custom_fleet", [])
        description = st.session_state.get("agent_custom_desc", "Custom scenario.")
    else:
        fleet = _PRESETS[selected]["fleet"]
        description = _PRESETS[selected]["description"]

    # ── 2. Fleet snapshot ─────────────────────────────────────────────────────
    st.markdown("### 2 — Fleet snapshot")
    status_icon = {"ok": "🟢", "warn": "🟡", "alert": "🔴"}
    for i in range(0, len(fleet), 2):
        row = fleet[i:i + 2]
        cols = st.columns(len(row))
        for col, unit in zip(cols, row):
            col.metric(
                label=unit["unit_id"],
                value=f"PSI {unit['psi']:.2f}",
                delta=f"{status_icon[unit['status']]} {unit['status'].upper()}",
                delta_color="off",
            )
            col.caption(unit["env"])

    # ── Tool inspector ────────────────────────────────────────────────────────
    with st.expander("🔍 Inspect tool responses — see what the agent will query"):
        tool_tabs = st.tabs(["get_fleet_status", "get_psi_report", "query_mlflow_runs", "get_model_registry"])
        with tool_tabs[0]:
            st.json(json.loads(_run_tool("get_fleet_status", {}, fleet)))
        with tool_tabs[1]:
            alert_units = [u for u in fleet if u["status"] in ("alert", "warn")]
            if alert_units:
                uid_choice = st.selectbox("Unit", [u["unit_id"] for u in alert_units],
                                          key="inspector_uid")
                st.json(json.loads(_run_tool("get_psi_report", {"unit_id": uid_choice}, fleet)))
            else:
                st.info("No warn/alert units in this scenario.")
        with tool_tabs[2]:
            st.json(json.loads(_run_tool("query_mlflow_runs", {}, fleet)))
        with tool_tabs[3]:
            st.json(json.loads(_run_tool("get_model_registry", {}, fleet)))

    # ── Agent instructions ────────────────────────────────────────────────────
    with st.expander("📝 Edit agent instructions (system prompt)"):
        system = st.text_area(
            "System prompt",
            value=st.session_state.get("agent_system", _DEFAULT_SYSTEM),
            height=260,
            key="agent_system_input",
            label_visibility="collapsed",
        )
        col_save, col_reset, _ = st.columns([1, 1, 4])
        if col_save.button("Save", key="sys_save"):
            st.session_state["agent_system"] = system
            st.session_state.pop("agent_trace", None)
            st.session_state.pop("agent_messages", None)
            st.session_state.pop("agent_followups", None)
            st.success("Saved — next run will use the updated instructions.")
        if col_reset.button("Reset", key="sys_reset"):
            st.session_state["agent_system"] = _DEFAULT_SYSTEM
            st.session_state.pop("agent_trace", None)
            st.session_state.pop("agent_messages", None)
            st.session_state.pop("agent_followups", None)
            st.rerun()
    system = st.session_state.get("agent_system", _DEFAULT_SYSTEM)

    st.divider()

    # ── 3. Agent trace ────────────────────────────────────────────────────────
    st.markdown("### 3 — Agent reasoning trace")
    st.caption("Watch the agent call tools, observe results, and build its recommendation.")

    run_col, clear_col, _ = st.columns([1, 1, 5])
    run_btn   = run_col.button("▶  Run Agent", type="primary", key="agent_run_btn")
    clear_btn = clear_col.button("✕  Clear", key="agent_clear_btn")

    if clear_btn:
        for k in ("agent_trace", "agent_messages", "agent_followups"):
            st.session_state.pop(k, None)
        st.rerun()

    if run_btn:
        for k in ("agent_trace", "agent_messages", "agent_followups"):
            st.session_state.pop(k, None)
        with st.spinner("Agent reasoning…"):
            trace, messages = _run_agent(fleet, description, api_key, system)
        st.session_state["agent_trace"]    = trace
        st.session_state["agent_messages"] = messages
        st.session_state["agent_followups"] = []
        st.rerun()

    if "agent_trace" in st.session_state:
        _render_events(st.session_state["agent_trace"])

        # Render any follow-up exchanges
        for q, evs in st.session_state.get("agent_followups", []):
            st.markdown(_card("#3498db", "💬 Follow-up", q), unsafe_allow_html=True)
            _render_events(evs)

        st.divider()

        # ── Follow-up conversation ────────────────────────────────────────────
        st.markdown("**Ask a follow-up — the agent keeps its tools and context**")
        followup = st.chat_input(
            "e.g. 'What if we can't access the test range for 3 weeks?' "
            "or 'How confident are you in that recommendation?'"
        )
        if followup:
            with st.spinner("Agent thinking…"):
                evs, updated_msgs = _run_followup(
                    st.session_state["agent_messages"],
                    followup, fleet, api_key, system,
                )
            st.session_state["agent_messages"] = updated_msgs
            st.session_state.setdefault("agent_followups", []).append((followup, evs))
            st.rerun()
    else:
        st.info("Select a scenario and click **▶ Run Agent** to start.")

    st.divider()

    # ── Explainer ─────────────────────────────────────────────────────────────
    st.markdown("### Why I think this is worth building")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Agentic patterns shown**")
        st.markdown(
            "- **Tool use over static context** — queries live state, not a fixed dashboard\n"
            "- **Multi-step investigation** — fleet-wide first, then drill, then training history\n"
            "- **Domain-aware triage** — distinguishes HW fault (PSI 0.67, isolated) "
            "from environmental drift (shared environment) from new drone (fleet-wide centroid)\n"
            "- **Conversational refinement** — follow-ups let the operator push back, "
            "add constraints, or ask for alternatives without starting over"
        )
    with col_r:
        st.markdown("**How it would slot into the pipeline**")
        st.code(
            "# PSI monitor fires on schedule\n"
            "if report.has_alerts:\n"
            "    rec = mlops_agent.advise(\n"
            "        fleet_report=report,\n"
            "        tools=[get_fleet_status,\n"
            "               get_psi_report,\n"
            "               query_mlflow_runs,\n"
            "               get_model_registry],\n"
            "    )\n"
            "    ops.notify(rec)          # human reviews\n"
            "    audit_log.append(rec)    # traceable",
            language="python",
        )
        st.caption(
            "The agent sits between automated alert and human decision — "
            "reducing on-call cognitive load without removing the human from the loop."
        )
