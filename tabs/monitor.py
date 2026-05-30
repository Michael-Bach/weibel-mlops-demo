"""
tabs/monitor.py — MLflow training monitor tab.

Reads the mlruns/ file store directly (no mlflow import) to avoid the
protobuf / Python 3.14 incompatibility on Streamlit Cloud.

MLflow file-store layout:
  mlruns/{exp_id}/meta.yaml          — experiment name
  mlruns/{exp_id}/{run_id}/meta.yaml — run status, start_time
  mlruns/{exp_id}/{run_id}/metrics/{key}  — lines: "ts value step"
  mlruns/{exp_id}/{run_id}/params/{key}   — single value
  mlruns/{exp_id}/{run_id}/tags/{key}     — single value
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT    = Path(__file__).parent.parent
MLRUNS  = ROOT / "mlruns"

_MODELS = {
    "CNN  (batch detector)": {
        "metrics_file": "ppi_metrics.json",
        "experiment":   "ppi-radar-detector",
        "train_cmd":    "PYTHONPATH=. python src/train_ppi.py",
        "color":        "#ffe66d",
    },
    "ConvGRU  (streaming detector)": {
        "metrics_file": "recurrent_metrics.json",
        "experiment":   "recurrent-radar-detector",
        "train_cmd":    "PYTHONPATH=. python src/train_recurrent.py",
        "color":        "#c084fc",
    },
}

_STATUS_ICON = {"FINISHED": "🟢", "RUNNING": "🟡", "FAILED": "🔴"}


# ── file-store helpers ────────────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return ""


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _find_experiment_id(name: str) -> str | None:
    if not MLRUNS.exists():
        return None
    for exp_dir in MLRUNS.iterdir():
        meta = _read_yaml(exp_dir / "meta.yaml")
        if meta.get("name") == name:
            return exp_dir.name
    return None


def _load_run(run_dir: Path) -> dict | None:
    meta = _read_yaml(run_dir / "meta.yaml")
    if not meta:
        return None

    def _kv(subdir: str) -> dict:
        d = {}
        p = run_dir / subdir
        if p.exists():
            for f in p.iterdir():
                d[f.name] = _read_file(f)
        return d

    params = _kv("params")
    tags   = _kv("tags")

    # latest value for each metric (last non-empty line)
    metrics: dict[str, float] = {}
    m_dir = run_dir / "metrics"
    if m_dir.exists():
        for mf in m_dir.iterdir():
            lines = [l for l in mf.read_text().splitlines() if l.strip()]
            if lines:
                try:
                    metrics[mf.name] = float(lines[-1].split()[1])
                except Exception:
                    pass

    return {
        "run_id":     meta.get("run_id", run_dir.name),
        "status":     meta.get("status", "UNKNOWN"),
        "start_time": meta.get("start_time", 0),
        "metrics":    metrics,
        "params":     params,
        "tags":       tags,
        "_run_dir":   run_dir,
    }


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_runs(experiment_name: str, n: int = 20) -> list[dict]:
    exp_id = _find_experiment_id(experiment_name)
    if exp_id is None:
        return []
    exp_dir = MLRUNS / exp_id
    runs = []
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "meta.yaml":
            continue
        r = _load_run(run_dir)
        if r:
            runs.append(r)
    runs.sort(key=lambda r: r["start_time"], reverse=True)
    return runs[:n]


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_history(run_dir_str: str, key: str) -> list[tuple[int, float]]:
    mf = Path(run_dir_str) / "metrics" / key
    if not mf.exists():
        return []
    result = []
    for line in mf.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                result.append((int(parts[2]), float(parts[1])))
            except Exception:
                pass
    return result


def _deployed_metrics(filename: str) -> dict:
    p = ROOT / "artifacts" / filename
    return json.loads(p.read_text()) if p.exists() else {}


def _f1_color(val: float | None) -> str:
    if val is None:
        return "grey"
    if val >= 0.60:
        return "#2ecc71"
    if val >= 0.40:
        return "#f39c12"
    return "#e74c3c"


# ── render ────────────────────────────────────────────────────────────────────

def render():
    st.markdown("### 📈 Model Training Monitor")
    st.caption(
        "Live view of MLflow training runs (reads `mlruns/` directly). "
        "Val F1 is at threshold 0.3 on the held-out validation set."
    )

    col_btn, col_note = st.columns([1, 5])
    with col_btn:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_note:
        st.caption("Cache TTL 30 s — or click Refresh for immediate update.")

    for model_label, cfg in _MODELS.items():
        st.markdown("---")
        st.markdown(f"#### {model_label}")

        deployed  = _deployed_metrics(cfg["metrics_file"])
        runs      = _fetch_runs(cfg["experiment"])
        d_f1      = deployed.get("val_f1")
        d_nparams = deployed.get("n_params")

        c_dep, c_run, c_hint = st.columns([1, 1, 2])

        with c_dep:
            st.markdown("**Deployed model**")
            color = _f1_color(d_f1)
            if d_f1 is not None:
                st.markdown(
                    f"<span style='font-size:2rem;color:{color};font-weight:700'>"
                    f"{d_f1:.3f}</span>&nbsp;val F1",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("—")
            if d_nparams:
                st.caption(f"{d_nparams:,} parameters")

        if not runs:
            with c_run:
                st.markdown("**Latest run**")
                st.info("No MLflow runs found.")
            with c_hint:
                st.markdown("**To train**")
                st.code(cfg["train_cmd"], language="bash")
            continue

        latest   = runs[0]
        lm       = latest["metrics"]
        lp       = latest["params"]
        ci_gate  = latest["tags"].get("ci_gate", "PASSED")
        status   = latest["status"]
        best_f1  = lm.get("best_val_f1")
        delta_f1 = (best_f1 - d_f1) if (best_f1 is not None and d_f1 is not None) else None

        with c_run:
            st.markdown("**Latest run**")
            icon = _STATUS_ICON.get(status, "⚪")
            ci_label = "✅ CI PASSED" if ci_gate != "FAILED" else "❌ CI FAILED"
            st.markdown(f"{icon} **{status}** &nbsp; {ci_label}", unsafe_allow_html=True)
            r_color = _f1_color(best_f1)
            if best_f1 is not None:
                st.markdown(
                    f"<span style='font-size:2rem;color:{r_color};font-weight:700'>"
                    f"{best_f1:.3f}</span>&nbsp;best val F1",
                    unsafe_allow_html=True,
                )
            if delta_f1 is not None:
                arrow  = "▲" if delta_f1 > 0 else "▼"
                dcolor = "#2ecc71" if delta_f1 > 0 else "#e74c3c"
                st.markdown(
                    f"<span style='color:{dcolor}'>{arrow} {abs(delta_f1):.3f} vs deployed</span>",
                    unsafe_allow_html=True,
                )
            st.caption(
                f"lr={lp.get('lr','—')}  "
                f"epochs={lp.get('epochs','—')}  "
                f"n_params={lp.get('n_params','—')}"
            )
            st.caption(f"run `{latest['run_id'][:8]}`")

        with c_hint:
            st.markdown("**To retrain**")
            st.code(cfg["train_cmd"], language="bash")
            if d_f1 is not None and d_f1 < 0.45:
                st.warning(
                    f"Val F1 = {d_f1:.3f} — model is likely missing targets in live detection. "
                    "Consider retraining with more epochs or a wider SNR range.",
                    icon="⚠️",
                )

        # ── training curve ──────────────────────────────────────────────────
        run_dir_str = str(latest["_run_dir"])
        loss_hist   = _fetch_history(run_dir_str, "train_loss")
        f1_hist     = _fetch_history(run_dir_str, "val_f1")
        seq_hist    = _fetch_history(run_dir_str, "seq_len")

        if loss_hist or f1_hist:
            fig   = go.Figure()
            _GRID = "rgba(128,128,128,0.15)"

            if loss_hist:
                steps, vals = zip(*loss_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps), y=list(vals), name="Train loss",
                    line=dict(color="#e74c3c", width=2),
                ))

            if seq_hist:
                steps_s, vals_s = zip(*seq_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps_s), y=list(vals_s), name="Seq len (curriculum)",
                    yaxis="y3",
                    line=dict(color="#888", width=1, dash="dot"),
                ))

            if f1_hist:
                steps2, vals2 = zip(*f1_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps2), y=list(vals2), name="Val F1",
                    yaxis="y2",
                    line=dict(color=cfg["color"], width=2.5),
                ))
                if d_f1 is not None:
                    fig.add_hline(
                        y=d_f1, line_dash="dash", line_color="#888",
                        annotation_text=f"deployed {d_f1:.3f}",
                        annotation_position="bottom right",
                        yref="y2",
                    )

            fig.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Epoch", gridcolor=_GRID),
                yaxis=dict(title="Train loss", color="#e74c3c", gridcolor=_GRID),
                yaxis2=dict(
                    title="Val F1", color=cfg["color"],
                    overlaying="y", side="right", range=[0, 1.05],
                ),
                yaxis3=dict(
                    overlaying="y", side="right",
                    showticklabels=False, showgrid=False,
                    anchor="free", position=1.0,
                ),
                legend=dict(
                    bgcolor="rgba(30,30,30,0.7)", bordercolor="#444",
                    borderwidth=1, font=dict(size=11),
                ),
                margin=dict(t=15, b=35, l=55, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── run history table ───────────────────────────────────────────────
        if len(runs) > 1:
            with st.expander(f"All {len(runs)} recorded runs"):
                rows = []
                for r in runs:
                    rows.append({
                        "Run ID":      r["run_id"][:8],
                        "Status":      _STATUS_ICON.get(r["status"], "⚪") + " " + r["status"],
                        "Best val F1": round(r["metrics"].get("best_val_f1", float("nan")), 3),
                        "Final loss":  round(r["metrics"].get("train_loss",  float("nan")), 4),
                        "lr":          r["params"].get("lr", "—"),
                        "epochs":      r["params"].get("epochs", "—"),
                        "n_params":    r["params"].get("n_params", "—"),
                        "CI gate":     r["tags"].get("ci_gate", "PASSED"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info(
        "**MLflow tracking:** runs are stored under `mlruns/`. "
        "Metrics update epoch-by-epoch while training — hit Refresh to see the latest step."
    )
