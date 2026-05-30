"""
tabs/monitor.py — MLflow training monitor tab.
"""
import json
import warnings
from pathlib import Path

import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent

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


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_runs(experiment_name: str, n: int = 20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            client = mlflow.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return []
            return list(client.search_runs(
                exp.experiment_id,
                order_by=["start_time DESC"],
                max_results=n,
            ))
        except Exception:
            return []


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_history(run_id: str, key: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            client = mlflow.MlflowClient()
            return [(h.step, h.value) for h in client.get_metric_history(run_id, key)]
        except Exception:
            return []


def _deployed_metrics(filename: str) -> dict:
    p = ROOT / "artifacts" / filename
    return json.loads(p.read_text()) if p.exists() else {}


def _f1_color(val_f1: float | None) -> str:
    if val_f1 is None:
        return "grey"
    if val_f1 >= 0.60:
        return "#2ecc71"
    if val_f1 >= 0.40:
        return "#f39c12"
    return "#e74c3c"


def render():
    st.markdown("### 📈 Model Training Monitor")
    st.caption(
        "Live view of MLflow training runs. "
        "Val F1 is computed at threshold 0.3 on the held-out validation set."
    )

    col_refresh, col_note = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_note:
        st.caption("Auto-refreshes every 30 s.  Click Refresh to pull latest run immediately.")

    for model_label, cfg in _MODELS.items():
        st.markdown("---")
        st.markdown(f"#### {model_label}")

        deployed   = _deployed_metrics(cfg["metrics_file"])
        runs       = _fetch_runs(cfg["experiment"])
        d_f1       = deployed.get("val_f1")
        d_nparams  = deployed.get("n_params")

        # ── top row: deployed | latest run | retrain hint ──────────────────
        c_dep, c_run, c_hint = st.columns([1, 1, 2])

        with c_dep:
            st.markdown("**Deployed model**")
            color = _f1_color(d_f1)
            st.markdown(
                f"<span style='font-size:2rem;color:{color};font-weight:700'>"
                f"{d_f1:.3f}</span> val F1" if d_f1 is not None else "—",
                unsafe_allow_html=True,
            )
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

        latest    = runs[0]
        lm        = latest.data.metrics
        lp        = latest.data.params
        ci_gate   = latest.data.tags.get("ci_gate", "PASSED")
        status    = latest.info.status
        best_f1   = lm.get("best_val_f1")
        delta_f1  = (best_f1 - d_f1) if (best_f1 is not None and d_f1 is not None) else None

        with c_run:
            st.markdown("**Latest run**")
            icon = _STATUS_ICON.get(status, "⚪")
            st.markdown(f"{icon} **{status}**  {'✅ CI PASSED' if ci_gate != 'FAILED' else '❌ CI FAILED'}")
            r_color = _f1_color(best_f1)
            st.markdown(
                f"<span style='font-size:2rem;color:{r_color};font-weight:700'>"
                f"{best_f1:.3f}</span> best val F1" if best_f1 is not None else "—",
                unsafe_allow_html=True,
            )
            if delta_f1 is not None:
                arrow = "▲" if delta_f1 > 0 else "▼"
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
            st.caption(f"run `{latest.info.run_id[:8]}`")

        with c_hint:
            st.markdown("**To retrain**")
            st.code(cfg["train_cmd"], language="bash")
            if d_f1 is not None and d_f1 < 0.45:
                st.warning(
                    f"Val F1 = {d_f1:.3f} — model is missing targets in live detection. "
                    "Consider retraining with more epochs or higher SNR range.",
                    icon="⚠️",
                )

        # ── training curve ──────────────────────────────────────────────────
        loss_hist = _fetch_history(latest.info.run_id, "train_loss")
        f1_hist   = _fetch_history(latest.info.run_id, "val_f1")
        seq_hist  = _fetch_history(latest.info.run_id, "seq_len")

        if loss_hist or f1_hist:
            fig = go.Figure()
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
                    line=dict(color="#aaa", width=1, dash="dot"),
                ))

            if f1_hist:
                steps2, vals2 = zip(*f1_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps2), y=list(vals2), name="Val F1",
                    yaxis="y2",
                    line=dict(color=cfg["color"], width=2.5),
                ))
                # deployed F1 reference line
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
                yaxis=dict(
                    title="Train loss", color="#e74c3c",
                    gridcolor=_GRID,
                ),
                yaxis2=dict(
                    title="Val F1", color=cfg["color"],
                    overlaying="y", side="right",
                    range=[0, 1.05],
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
                        "Run ID":       r.info.run_id[:8],
                        "Status":       _STATUS_ICON.get(r.info.status, "⚪") + " " + r.info.status,
                        "Best val F1":  round(r.data.metrics.get("best_val_f1", float("nan")), 3),
                        "Final loss":   round(r.data.metrics.get("train_loss",  float("nan")), 4),
                        "lr":           r.data.params.get("lr", "—"),
                        "epochs":       r.data.params.get("epochs", "—"),
                        "n_params":     r.data.params.get("n_params", "—"),
                        "CI gate":      r.data.tags.get("ci_gate", "PASSED"),
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    st.info(
        "**MLflow tracking:** runs are stored locally under `mlruns/`. "
        "Metrics update live while training is in progress — refresh this tab to see the latest epoch."
    )
