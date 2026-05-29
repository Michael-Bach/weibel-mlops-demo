"""
tabs/roc.py — render() for tab_roc.
"""

import plotly.graph_objects as go
import streamlit as st

from radar.detection import _roc_data, _roc, _auc


def _auc_label(val: float) -> str:
    if val >= 0.95:
        return "Excellent"
    if val >= 0.85:
        return "Good"
    if val >= 0.70:
        return "Fair"
    return "Poor"


def render():
    st.caption("*Drag the SNR slider to see every detector's detection-vs-false-alarm curve — closer to the top-left corner is better.*")
    st.markdown("### ROC curves: how reliably does each detector separate signal from noise?")
    st.markdown(
        "A ROC curve answers: *if I set the detector sensitivity to a given level, "
        "how many real targets do I catch — and how many false alarms do I generate?* "
        "The curve sweeps every possible threshold and plots **Detection Rate** (targets found) "
        "against **False Alarm Rate** (noise mistakenly flagged). "
        "**Closer to the top-left corner = better.** A detector on the diagonal is no better than "
        "a coin flip. The **AUC** (Area Under Curve) collapses the whole curve to one number: "
        "1.0 = perfect, 0.5 = random guessing, anything above ~0.75 is operationally useful."
    )
    st.markdown(
        "**How scores are computed** across 100 trials (50 with target, 50 clutter-only), "
        "all evaluated at the oracle target position for a fair cell-level comparison: "
        "**CNN** — peak probability map output in the target's neighbourhood. "
        "**ConvGRU** — peak hidden-state confidence at the target's neighbourhood after 10 sweeps. "
        "**CA-CFAR** — fraction of sweeps where CFAR fires near the target's actual position."
    )
    st.info(
        "**Why the Kalman Filter is not on the ROC curve:** "
        "The KF is a *track confirmer*, not a frame-level detector. "
        "It doesn't produce a continuous score that can be swept for a ROC — "
        "its output is binary (confirmed track / no confirmed track). "
        "At 5% Pfa across 11 520 cells there are ~540 false alarms per sweep, "
        "and the tracker (which processes only the top 10 peaks) is overwhelmed before it can latch onto the real target. "
        "The **Algorithm Comparison** tab shows KF performance correctly: once a confirmed track lands "
        "within 6 bins of the true target, it counts as a detection — "
        "that's the right metric for a tracker."
    )
    col_r, col_p = st.columns([1, 2])
    with col_r:
        st.markdown("**SNR for this trial set:**")
        roc_snr = st.slider("SNR (dB)", -20.0, 40.0, 0.0, 1.0, key="roc_snr")
        st.caption(
            "Drag the slider to see how each algorithm's ROC changes with target strength. "
            "At low SNR all three curves collapse toward the diagonal. "
            "At high SNR the ML curve should reach the top-left corner first."
        )

    ml_s, cf_s, gru_s, lrt_s, tbd_s, tf_s, lbs = _roc_data(roc_snr, _v=11)
    ml_f,  ml_t  = _roc(ml_s,  lbs)
    cf_f,  cf_t  = _roc(cf_s,  lbs)
    gru_f, gru_t = _roc(gru_s, lbs)
    lrt_f, lrt_t = _roc(lrt_s, lbs)
    tbd_f, tbd_t = _roc(tbd_s, lbs)
    tf_f,  tf_t  = _roc(tf_s,  lbs)

    _ROC_GRID = "rgba(100,100,100,0.15)"
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(color="#aaa", dash="dot"), name="Chance (AUC 0.500)"))
    fig_r.add_trace(go.Scatter(x=cf_f, y=cf_t, mode="lines",
                               name=f"CA-CFAR  AUC {_auc(cf_f, cf_t):.3f}",
                               line=dict(color="#e67e22", width=2.0, dash="dash")))
    fig_r.add_trace(go.Scatter(x=ml_f, y=ml_t, mode="lines",
                               name=f"CNN  AUC {_auc(ml_f, ml_t):.3f}",
                               line=dict(color="#2980b9", width=2.5)))
    fig_r.add_trace(go.Scatter(x=gru_f, y=gru_t, mode="lines",
                               name=f"ConvGRU  AUC {_auc(gru_f, gru_t):.3f}",
                               line=dict(color="#8e44ad", width=2.5)))
    fig_r.add_trace(go.Scatter(x=lrt_f, y=lrt_t, mode="lines",
                               name=f"LRT (non-coh.)  AUC {_auc(lrt_f, lrt_t):.3f}",
                               line=dict(color="#27ae60", width=2.0, dash="dashdot")))
    fig_r.add_trace(go.Scatter(x=tbd_f, y=tbd_t, mode="lines",
                               name=f"DP-TBD  AUC {_auc(tbd_f, tbd_t):.3f}",
                               line=dict(color="#e74c3c", width=2.0, dash="dashdot")))
    fig_r.add_trace(go.Scatter(x=tf_f, y=tf_t, mode="lines",
                               name=f"Transformer  AUC {_auc(tf_f, tf_t):.3f}",
                               line=dict(color="#f39c12", width=2.5)))
    fig_r.update_layout(
        height=440,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="False Alarm Rate  (fraction of clutter cells incorrectly flagged)",
                   range=[-0.02, 1.02], gridcolor=_ROC_GRID, zeroline=False),
        yaxis=dict(title="Detection Rate  (fraction of real targets found)",
                   range=[-0.02, 1.04], gridcolor=_ROC_GRID, zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    with col_p:
        st.plotly_chart(fig_r, use_container_width=True)

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("CNN AUC",           f"{_auc(ml_f,  ml_t):.3f}",  _auc_label(_auc(ml_f,  ml_t)))
    r2.metric("Transformer AUC",   f"{_auc(tf_f,  tf_t):.3f}",  _auc_label(_auc(tf_f,  tf_t)))
    r3.metric("ConvGRU AUC",       f"{_auc(gru_f, gru_t):.3f}", _auc_label(_auc(gru_f, gru_t)))
    r4.metric("CA-CFAR AUC",       f"{_auc(cf_f,  cf_t):.3f}",  _auc_label(_auc(cf_f,  cf_t)))
    r5.metric("LRT (non-coh.) AUC",f"{_auc(lrt_f, lrt_t):.3f}", _auc_label(_auc(lrt_f, lrt_t)))
    r6.metric("DP-TBD AUC",        f"{_auc(tbd_f, tbd_t):.3f}", _auc_label(_auc(tbd_f, tbd_t)))
    st.caption(
        "**Reading the ROC:** left side = conservative (low false alarms, some targets missed). "
        "Right side = aggressive (catch more targets, more false alarms). "
        "The CFAR curve is often jagged because its score (fraction of sweeps with a hit) "
        "takes only a handful of discrete values. "
        "The ConvGRU score is the final accumulated hidden-state value at the target cell — "
        "it builds confidence across sweeps, so its AUC rises sharply once SNR exceeds ~4 dB."
    )

    st.info(
        "**Next tab →** Strengths & Trade-offs: when to use each algorithm and why "
        "combining them gives the best operational outcome."
    )
