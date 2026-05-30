"""
tabs/monitor.py — MLflow training monitor tab.

Reads the mlruns/ file store directly (no mlflow import) to avoid the
protobuf / Python 3.14 incompatibility on Streamlit Cloud.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT   = Path(__file__).parent.parent
MLRUNS = ROOT / "mlruns"

_MODELS = {
    "CNN": {
        "label":        "CNN  (batch detector)",
        "metrics_file": "ppi_metrics.json",
        "experiment":   "ppi-radar-detector",
        "train_cmd":    "PYTHONPATH=. python src/train_ppi.py",
        "colors":       ["#ffe66d", "#f39c12", "#e67e22", "#d35400"],
    },
    "ConvGRU": {
        "label":        "ConvGRU  (streaming detector)",
        "metrics_file": "recurrent_metrics.json",
        "experiment":   "recurrent-radar-detector",
        "train_cmd":    "PYTHONPATH=. python src/train_recurrent.py",
        "colors":       ["#c084fc", "#8b5cf6", "#7c3aed", "#5b21b6"],
    },
}

_STATUS_ICON = {"FINISHED": "🟢", "RUNNING": "🟡", "FAILED": "🔴"}
_MEDALS      = ["🥇", "🥈", "🥉"]


# ── file-store helpers ────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _find_experiment_dir(name: str) -> Path | None:
    if not MLRUNS.exists():
        return None
    for d in MLRUNS.iterdir():
        if not d.is_dir():
            continue
        meta = _read_yaml(d / "meta.yaml")
        if meta.get("name") == name:
            return d
    return None


def _load_run(run_dir: Path) -> dict | None:
    meta = _read_yaml(run_dir / "meta.yaml")
    if not meta:
        return None

    def _kv(sub: str) -> dict:
        p = run_dir / sub
        return {f.name: f.read_text().strip() for f in p.iterdir()} if p.exists() else {}

    params = _kv("params")
    tags   = _kv("tags")

    metrics: dict[str, float] = {}
    m_dir = run_dir / "metrics"
    if m_dir.exists():
        for mf in m_dir.iterdir():
            lines = [ln for ln in mf.read_text().splitlines() if ln.strip()]
            if lines:
                try:
                    metrics[mf.name] = float(lines[-1].split()[1])
                except Exception:
                    pass

    # MLflow may write status as an integer (1=RUNNING,3=FINISHED,4=FAILED)
    _STATUS_MAP = {1: "RUNNING", 2: "SCHEDULED", 3: "FINISHED", 4: "FAILED", 5: "KILLED"}
    raw_status  = meta.get("status", "UNKNOWN")
    status      = _STATUS_MAP.get(raw_status, str(raw_status))

    return {
        "run_id":     str(meta.get("run_id", run_dir.name)),
        "status":     status,
        "start_time": meta.get("start_time", 0),
        "metrics":    metrics,
        "params":     params,
        "tags":       tags,
        "_run_dir":   run_dir,
    }


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_runs(experiment_name: str) -> list[dict]:
    exp_dir = _find_experiment_dir(experiment_name)
    if exp_dir is None:
        return []
    runs = []
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        r = _load_run(run_dir)
        if r:
            runs.append(r)
    runs.sort(key=lambda r: r["start_time"], reverse=True)
    return runs


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


def _deployed(filename: str) -> dict:
    p = ROOT / "artifacts" / filename
    return json.loads(p.read_text()) if p.exists() else {}


def _f1_color(v: float | None) -> str:
    if v is None:
        return "#888"
    if v >= 0.60:
        return "#2ecc71"
    if v >= 0.40:
        return "#f39c12"
    return "#e74c3c"


def _run_label(r: dict) -> str:
    lp = r["params"]
    return f"lr={lp.get('lr','?')} ep={lp.get('epochs','?')}"


# ── render ────────────────────────────────────────────────────────────────────

def render():
    st.markdown("### 📈 Model Training Monitor")
    st.caption(
        "The Pipeline tab described how training runs feed the deployment loop. "
        "This tab is where you observe that loop: compare experiments, spot regressions, "
        "and verify a new run beats the deployed model before promoting it to the fleet."
    )

    st.info(
        "**Reading the F1 scores:** val F1 is measured at threshold 0.3 on *soft Gaussian "
        "heatmap labels* — a continuous ground-truth blob around the target cell, not a clean "
        "binary mask. Hard-threshold F1 on soft labels is inherently noisy and lower than you'd "
        "see with binary annotations: a prediction offset by one bin is penalised even if it "
        "correctly located the target. The ROC AUC (see the ROC tab) is the more reliable "
        "performance indicator; val F1 here measures calibration quality, not detection ceiling."
    )

    col_btn, col_note = st.columns([1, 6])
    with col_btn:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_note:
        st.caption("Reads `mlruns/` directly — no mlflow server needed.  Cache TTL 30 s.")

    # ── cross-model hero row ──────────────────────────────────────────────────
    all_deployed = {k: _deployed(v["metrics_file"]) for k, v in _MODELS.items()}
    all_runs     = {k: _fetch_runs(v["experiment"])  for k, v in _MODELS.items()}

    st.markdown("#### Deployed models at a glance")
    hero_cols = st.columns(len(_MODELS))
    for col, (key, cfg) in zip(hero_cols, _MODELS.items()):
        d    = all_deployed[key]
        runs = all_runs[key]
        d_f1 = d.get("val_f1")
        best_run_f1 = max(
            (r["metrics"].get("best_val_f1", 0) for r in runs), default=None
        ) if runs else None

        with col:
            color = _f1_color(d_f1)
            st.markdown(
                f"<div style='border:1px solid #333;border-radius:8px;padding:16px 20px;'>"
                f"<div style='font-size:0.85rem;color:#aaa'>{cfg['label']}</div>"
                f"<div style='font-size:2.6rem;font-weight:700;color:{color};line-height:1.1'>"
                f"{d_f1:.3f}</div>"
                f"<div style='font-size:0.8rem;color:#888'>deployed val F1</div>"
                + (
                    f"<div style='font-size:0.8rem;margin-top:6px;color:"
                    f"{'#2ecc71' if best_run_f1 and best_run_f1 > (d_f1 or 0) else '#888'}'>"
                    f"best logged: {best_run_f1:.3f}</div>"
                    if best_run_f1 is not None else ""
                )
                + f"<div style='font-size:0.75rem;color:#666'>{d.get('n_params',0):,} params  "
                f"|  {len(runs)} run{'s' if len(runs)!=1 else ''}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── per-experiment panels ─────────────────────────────────────────────────
    for key, cfg in _MODELS.items():
        st.markdown("---")
        st.markdown(f"#### {cfg['label']}")

        d_f1  = all_deployed[key].get("val_f1")
        runs  = all_runs[key]
        colors = cfg["colors"]

        if not runs:
            st.info(f"No runs found for `{cfg['experiment']}`.")
            st.code(cfg["train_cmd"], language="bash")
            continue

        # Sort by best_val_f1 for ranking
        ranked = sorted(runs, key=lambda r: r["metrics"].get("best_val_f1", 0), reverse=True)
        best_run = ranked[0]

        # ── metrics strip ────────────────────────────────────────────────────
        n_cols = min(len(runs), 4)
        m_cols = st.columns(n_cols + 1)

        with m_cols[0]:
            color = _f1_color(d_f1)
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:0.75rem;color:#aaa'>⭐ deployed</div>"
                f"<div style='font-size:1.8rem;font-weight:700;color:{color}'>{d_f1:.3f}</div>"
                f"<div style='font-size:0.7rem;color:#666'>val F1</div></div>",
                unsafe_allow_html=True,
            )

        for i, r in enumerate(ranked[:n_cols]):
            f1   = r["metrics"].get("best_val_f1")
            col  = colors[i % len(colors)]
            icon   = _MEDALS[i] if i < 3 else f"#{i+1}"
            lp     = r["params"]
            f1_str = "—" if f1 is None else "{:.3f}".format(f1)
            with m_cols[i + 1]:
                st.markdown(
                    f"<div style='text-align:center;border-left:3px solid {col};padding-left:8px'>"
                    f"<div style='font-size:0.75rem;color:#aaa'>{icon} {_run_label(r)}</div>"
                    f"<div style='font-size:1.8rem;font-weight:700;color:{_f1_color(f1)}'>"
                    f"{f1_str}</div>"
                    f"<div style='font-size:0.7rem;color:#666'>"
                    f"n_train={lp.get('n_train','?')}</div></div>",
                    unsafe_allow_html=True,
                )

        # ── multi-run chart ──────────────────────────────────────────────────
        st.markdown("**Training curves — all runs**")
        fig   = go.Figure()
        _GRID = "rgba(128,128,128,0.12)"

        for i, r in enumerate(ranked):
            col   = colors[i % len(colors)]
            label = _MEDALS[i] if i < 3 else f"#{i+1}"
            width = 2.5 if i == 0 else 1.5
            rdstr = str(r["_run_dir"])

            loss_hist = _fetch_history(rdstr, "train_loss")
            f1_hist   = _fetch_history(rdstr, "val_f1")

            if loss_hist:
                steps, vals = zip(*loss_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps), y=list(vals),
                    name=f"{label} loss ({_run_label(r)})",
                    line=dict(color=col, width=width, dash="dot"),
                    opacity=0.55,
                    showlegend=(i == 0),
                    legendgroup=f"run{i}",
                ))

            if f1_hist:
                steps2, vals2 = zip(*f1_hist)
                fig.add_trace(go.Scatter(
                    x=list(steps2), y=list(vals2),
                    name=f"{label} {_run_label(r)}",
                    yaxis="y2",
                    line=dict(color=col, width=width),
                    legendgroup=f"run{i}",
                ))

        if d_f1 is not None:
            fig.add_hline(
                y=d_f1, line_dash="dash", line_color="#888", line_width=1,
                annotation_text=f"deployed {d_f1:.3f}",
                annotation_position="bottom right",
                yref="y2",
            )

        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch", gridcolor=_GRID),
            yaxis=dict(title="Train loss (dotted)", gridcolor=_GRID, color="#888"),
            yaxis2=dict(
                title="Val F1", overlaying="y", side="right",
                range=[0, 1.05], color="white",
            ),
            legend=dict(
                bgcolor="rgba(20,20,20,0.8)", bordercolor="#444",
                borderwidth=1, font=dict(size=10),
            ),
            margin=dict(t=10, b=35, l=55, r=65),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── curriculum strip (GRU only) ──────────────────────────────────────
        for i, r in enumerate(ranked[:1]):
            seq = _fetch_history(str(r["_run_dir"]), "seq_len")
            if seq:
                st.caption(f"Curriculum (best run): sequence length {' → '.join(str(int(v)) for _, v in seq[::max(1, len(seq)//6)])}")

        # ── leaderboard ──────────────────────────────────────────────────────
        st.markdown("**Run leaderboard**")
        rows = []
        for i, r in enumerate(ranked):
            lm, lp, lt = r["metrics"], r["params"], r["tags"]
            f1 = lm.get("best_val_f1")
            rows.append({
                "Rank":        (_MEDALS[i] if i < 3 else f"#{i+1}"),
                "Run":         r["run_id"][:8],
                "Status":      _STATUS_ICON.get(r["status"], "⚪") + " " + r["status"],
                "Best val F1": f"{f1:.4f}" if f1 is not None else "—",
                "Δ deployed":  f"{f1 - d_f1:+.4f}" if (f1 and d_f1) else "—",
                "lr":          lp.get("lr", "—"),
                "epochs":      lp.get("epochs", "—"),
                "n_train":     lp.get("n_train", "—"),
                "CI gate":     "✅" if lt.get("ci_gate") != "FAILED" else "❌",
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color: rgba(46,204,113,0.15)" if row["Rank"] == "🥇" else ""
                ] * len(row),
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── cross-model comparison ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### CNN vs ConvGRU — all runs combined")

    fig2   = go.Figure()
    _GRID2 = "rgba(128,128,128,0.12)"
    for key, cfg in _MODELS.items():
        runs = all_runs[key]
        ranked = sorted(runs, key=lambda r: r["metrics"].get("best_val_f1", 0), reverse=True)
        for i, r in enumerate(ranked):
            f1_hist = _fetch_history(str(r["_run_dir"]), "val_f1")
            if not f1_hist:
                continue
            steps, vals = zip(*f1_hist)
            col   = cfg["colors"][i % len(cfg["colors"])]
            label = f"{key} {_MEDALS[i] if i < 3 else f'#{i+1}'} {_run_label(r)}"
            fig2.add_trace(go.Scatter(
                x=list(steps), y=list(vals), name=label,
                line=dict(color=col, width=2.5 if i == 0 else 1.5),
                opacity=1.0 if i == 0 else 0.7,
            ))

    fig2.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Epoch", gridcolor=_GRID2),
        yaxis=dict(title="Val F1", gridcolor=_GRID2, range=[0, 1.0]),
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)", bordercolor="#444",
            borderwidth=1, font=dict(size=10),
        ),
        margin=dict(t=10, b=35, l=55, r=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── ONNX model registry ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Deployed ONNX artefacts")
    onnx_files = {
        "CNN":      ROOT / "artifacts" / "ppi_model.onnx",
        "ConvGRU":  ROOT / "artifacts" / "recurrent_model.onnx",
    }
    reg_cols = st.columns(len(onnx_files))
    for col, (name, path) in zip(reg_cols, onnx_files.items()):
        with col:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                mtime   = pd.Timestamp(path.stat().st_mtime, unit="s")
                st.markdown(
                    f"**{name}** `{path.name}`  \n"
                    f"Size: **{size_kb:.1f} KB**  \n"
                    f"Updated: {mtime.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                st.warning(f"{name}: ONNX not found")

    st.caption(
        "Metrics update epoch-by-epoch during training — hit **Refresh** to pull the latest step."
    )
