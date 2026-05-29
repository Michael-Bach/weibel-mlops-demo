"""
tabs/classical.py — render() for tab_classical.
"""

import streamlit as st


def render():
    st.markdown("## Classical radar detection: CA-CFAR + Kalman tracker")
    st.caption("*Understand the baseline — and exactly where CFAR breaks before the ML case is made.*")
    st.markdown(
        "Before introducing machine learning it is useful to understand the standard pipeline "
        "it replaces — and why that pipeline is difficult to improve without changing the "
        "underlying architecture."
    )

    st.divider()

    # ── Signal chain ──────────────────────────────────────────────────────────
    st.markdown("### 1 — The signal chain")
    st.markdown(
        "A rotating surveillance radar illuminates the scene in a sequence of narrow azimuth "
        "beams. For each beam it transmits a short pulse and records the echo amplitude at "
        "each range gate as the pulse return window slides forward in time. The result after "
        "one full 360° rotation is a **Plan Position Indicator (PPI) sweep** — a 2-D grid of "
        "amplitude samples indexed by (azimuth, range)."
    )
    st.code(
        "One rotation = one PPI sweep  (n_az × n_range  amplitude samples)\n"
        "Ten rotations = the sequence used for detection in this demo\n"
        "\n"
        "  Azimuth resolution:  2°  (180 bins per sweep)\n"
        "  Range resolution:    7.5 m  (64 bins, ≈ 480 m max range)\n"
        "  Rotation period:     2 s  (30 RPM)",
        language="text",
    )

    st.divider()

    # ── CA-CFAR ───────────────────────────────────────────────────────────────
    st.markdown("### 2 — Cell-Averaging CFAR (CA-CFAR)")
    st.markdown(
        "**CFAR** stands for *Constant False Alarm Rate*. The idea is that a target detection "
        "threshold should automatically adapt to the local noise level — otherwise a fixed "
        "threshold would produce too many false alarms in strong clutter and miss targets in "
        "quiet regions."
    )

    col_cfar_txt, col_cfar_eq = st.columns([3, 2])
    with col_cfar_txt:
        st.markdown(
            "For each **cell under test (CUT)**, CA-CFAR computes a noise estimate from the "
            "surrounding **reference cells**, excluding a **guard band** that protects against "
            "the target's own sidelobe energy leaking into the reference window:"
        )
        st.markdown(
            "- Reference cells: the N cells surrounding the guard ring\n"
            "- Guard cells: a small exclusion zone around the CUT\n"
            "- Noise estimate: Z = mean amplitude across all reference cells\n"
            "- Detection threshold: **T = α · Z**\n"
            "- Detection if CUT > T, otherwise no detection"
        )
        st.markdown(
            "The constant **α** is derived analytically to achieve a target false alarm "
            "probability *Pfa* — hence *Constant False Alarm Rate*. "
            "In this demo: guard = 2 cells, reference = 5 cells each side, α = 2.5."
        )
    with col_cfar_eq:
        st.code(
            "# 2-D CA-CFAR (per sweep)\n"
            "for each cell (az, r):\n"
            "    ref = cells in ring(az±7, r±7)\n"
            "         excluding guard ring\n"
            "    Z   = mean(ref)\n"
            "    T   = α * Z\n"
            "    det[az,r] = (cell > T)",
            language="python",
        )

    st.markdown(
        "**Where it works:** CFAR is mathematically optimal when background noise is "
        "homogeneous (all reference cells drawn from the same distribution). For an isolated "
        "target in Rayleigh-distributed clutter with sufficient SNR it achieves very close "
        "to the theoretical Pd curve."
    )
    st.warning(
        "**Where it fails:** The homogeneity assumption breaks down in real coastal and "
        "littoral environments. A reference window straddling a sea-land boundary, a rain "
        "cell, or a vessel wake mixes two different noise distributions. The resulting noise "
        "estimate is wrong — producing either excessive false alarms (underestimate) or "
        "missed detections (overestimate). This failure mode is *fundamental to the "
        "algorithm* and cannot be fixed by tuning α."
    )

    st.divider()

    # ── Kalman filter tracker ─────────────────────────────────────────────────
    st.markdown("### 3 — Kalman filter tracker")
    st.markdown(
        "A single CFAR sweep produces hundreds of detections per rotation — most of them "
        "noise. The **Kalman filter tracker** converts the noisy binary detection map into "
        "confirmed tracks by requiring *spatial and temporal consistency*."
    )

    col_kf1, col_kf2 = st.columns(2)
    with col_kf1:
        st.markdown("**State vector per track:**")
        st.code(
            "x = [range_bin,     # estimated radial position\n"
            "     range_rate,    # bins/sweep\n"
            "     azimuth_bin,   # estimated angular position\n"
            "     azimuth_rate]  # bins/sweep",
            language="python",
        )
        st.markdown("**Predict–update cycle (per sweep):**")
        st.code(
            "# Predict\n"
            "x_pred = F @ x_prev          # kinematics\n"
            "P_pred = F @ P_prev @ F.T + Q # process noise\n"
            "\n"
            "# Associate\n"
            "for each new detection d:\n"
            "    dist = euclidean(d, x_pred)\n"
            "    if dist < gate:            # 8-bin gate\n"
            "        update(x, P, d)        # standard KF\n"
            "\n"
            "# Confirm\n"
            "if track.hits >= 4:  # M-of-N rule\n"
            "    confirm(track)             # report to operator",
            language="python",
        )
    with col_kf2:
        st.markdown("**Why ≥ 4 hits?**")
        st.markdown(
            "A random noise spike that triggers CFAR in one sweep is unlikely to appear in "
            "the same location next sweep, and the sweep after that. Requiring *at least 4 "
            "consistent associations within an 8-bin gate* reduces the false track rate by "
            "several orders of magnitude compared to reporting every CFAR hit directly."
        )
        st.markdown("**Classical pipeline summary:**")
        st.code(
            "raw PPI sweep\n"
            "  → CFAR  (binary detection map)\n"
            "  → peak extraction  (centroid of each blob)\n"
            "  → KF predict  (forward-project existing tracks)\n"
            "  → nearest-neighbour association\n"
            "  → KF update\n"
            "  → confirm if hits ≥ 4\n"
            "  → confirmed track (position + velocity)",
            language="text",
        )

    st.divider()

    # ── LRT — optional deep-dive ──────────────────────────────────────────────
    with st.expander(
        "Advanced: Likelihood Ratio Test (LRT) — multi-sweep energy accumulation",
        expanded=False,
    ):
        st.caption(
            "A classical technique that improves on single-sweep CFAR by summing signal energy "
            "across multiple rotations. Skip if you are following the ML story — return here to "
            "understand why ML has an architecture advantage over all classical methods."
        )
        st.markdown("### Likelihood Ratio Test (LRT) / Non-coherent Integrator")
        st.markdown(
            "The **Neyman-Pearson Lemma** states that the most powerful detector at a given false-alarm "
            "rate is the one that computes the likelihood ratio between the two hypotheses and "
            "thresholds it. For Rayleigh-distributed clutter the optimal single-sweep detector "
            "reduces to a simple amplitude threshold — which is what CFAR approximates. "
            "Across *N* sweeps, when the noise variance is known but the target amplitude is not "
            "(the GLRT / composite-hypothesis case), the optimal statistic at low SNR is the "
            "**square-law (energy) combiner**:"
        )
        col_lrt_txt, col_lrt_code = st.columns([3, 2])
        with col_lrt_txt:
            st.markdown(r"""
$$\Lambda(az,r) = \sum_{k=1}^{N} \left(\frac{x_k(az,r)}{\hat{\sigma}(r)}\right)^2$$

where $\hat{\sigma}(r)$ is the range-dependent noise floor (10th-percentile amplitude).

| Property | Value |
|---|---|
| **Complexity** | $O(N \cdot N_{az} \cdot N_r)$ — a single pass over the cube |
| **Assumes** | Rayleigh clutter, *known* noise floor, *stationary* target |
| **Optimal when** | Target stays in one cell across all N sweeps |
| **Degrades when** | Target moves across cells (energy is spread over the trajectory) |
""")
        with col_lrt_code:
            st.code(
                "# radar/detection.py\n"
                "def _lrt_score(ppi):          # ppi: (N, n_az, n_r)\n"
                "    nf = percentile(ppi, 10,\n"
                "             axis=(1,2),\n"
                "             keepdims=True).clip(1e-6)\n"
                "    normed = ppi / nf\n"
                "    return (normed**2).sum(0) # (n_az, n_r)",
                language="python",
            )
        st.info(
            "**Where LRT beats CFAR:** CFAR applies a threshold independently to each sweep and "
            "discards the result. The LRT accumulates energy over all N sweeps at each fixed cell — "
            "giving an SNR gain of roughly 10 log₁₀(N) dB for a stationary target. "
            "For a target that barely clears the noise floor on any individual sweep, this "
            "integration gain can be the difference between detection and a miss."
        )
        st.warning(
            "**Where LRT falls short:** it assumes the target sits in a single (az, r) cell for all "
            "N sweeps. A moving target spreads its energy across several cells as it traverses the PPI, "
            "so the score at any single cell only sees a fraction of the available energy. "
            "DP-TBD (below) solves exactly this problem."
        )

    st.divider()

    # ── DP-TBD — optional deep-dive ───────────────────────────────────────────
    with st.expander(
        "Advanced: Track-Before-Detect (DP-TBD) — following the target path",
        expanded=False,
    ):
        st.caption(
            "The most capable purely classical approach: accumulates evidence along the target's "
            "trajectory rather than at a fixed cell. Understanding it makes clear where and why "
            "the ML models outperform the entire classical family."
        )
        st.markdown("### Dynamic-Programming Track-Before-Detect (DP-TBD)")
        st.markdown(
            "**Track-Before-Detect** reverses the classical pipeline: instead of thresholding each "
            "sweep and then tracking the resulting detections, TBD accumulates *raw amplitude* across "
            "sweeps along every plausible target trajectory, and only thresholds the final accumulated "
            "score. This avoids discarding evidence from sweeps where the target was below the "
            "single-sweep threshold."
        )
        col_tbd1, col_tbd2 = st.columns(2)
        with col_tbd1:
            st.markdown("**The DP recursion:**")
            st.markdown(r"""
Initialise $S_0(az,r) = \tilde{x}_0(az,r)$ (normalised amplitude, sweep 0).

For each subsequent sweep $k$:

$$S_k(az,r) = \tilde{x}_k(az,r) + \max_{|\Delta az|\le v_{az},\,|\Delta r|\le v_r} S_{k-1}(az+\Delta az,\, r+\Delta r)$$

After $N$ sweeps, declare a detection at any cell where $S_N(az,r) > \eta$.

| Parameter | Value in this demo |
|---|---|
| Max radial motion $v_r$ | ±3 range bins / sweep |
| Max angular motion $v_{az}$ | ±2 azimuth bins / sweep |
| Complexity | $O(N \cdot (2v_{az}+1)(2v_r+1) \cdot N_{az} \cdot N_r)$ |
""")
        with col_tbd2:
            st.code(
                "# radar/detection.py\n"
                "def _dp_tbd_score(ppi, max_vr=3, max_vaz=2):\n"
                "    nf = percentile(ppi, 10, axis=(1,2),\n"
                "                   keepdims=True).clip(1e-6)\n"
                "    normed = ppi / nf           # (N, n_az, n_r)\n"
                "    S = normed[0].copy()\n"
                "    size = (2*max_vaz+1, 2*max_vr+1)\n"
                "    for k in range(1, N):\n"
                "        # max over velocity neighbourhood\n"
                "        best = maximum_filter(S, size=size,\n"
                "                   mode=('wrap','nearest'))\n"
                "        S = normed[k] + best\n"
                "    return S                    # (n_az, n_r)",
                language="python",
            )
        st.success(
            "**Why DP-TBD wins at very low SNR:** by following the target's actual trajectory through "
            "the cube, DP-TBD accumulates the full N-sweep integration gain *even for a moving target*. "
            "The max-filter propagation step ensures that if a target has moved by up to ±3 bins "
            "between sweeps, the score still traces it. "
            "The tradeoff is velocity ambiguity: the neighbourhood size must bound the fastest "
            "target you expect, and a larger neighbourhood increases both sensitivity and false-alarm rate."
        )

    st.divider()

    # ── Limitations summary ───────────────────────────────────────────────────
    st.markdown("### Operational limitations")
    col_lim1, col_lim2, col_lim3 = st.columns(3)
    with col_lim1:
        st.markdown("**Low SNR performance**")
        st.markdown(
            "CFAR applies a threshold independently to each sweep. When the target echo "
            "barely clears the noise floor, CFAR misses it most sweeps and the KF never "
            "accumulates enough hits to confirm. The ML pipeline integrates evidence across "
            "sweeps through a learned representation — achieving full detection "
            "several decibels lower."
        )
    with col_lim2:
        st.markdown("**Heterogeneous clutter**")
        st.markdown(
            "Reference window contamination at clutter boundaries floods the display with "
            "false alarms. The KF suppresses most of them but confirmed false tracks still "
            "reach the operator. CFAR has no mechanism to learn or adapt to a specific "
            "clutter environment — the noise model is fixed at algorithm design time."
        )
    with col_lim3:
        st.markdown("**Fixed operating point**")
        st.markdown(
            "α is a scalar constant. Changing the operating point (Pfa vs Pd trade-off) "
            "requires re-deriving and redeploying the constant. An ML detector outputs a "
            "continuous confidence score that can be thresholded post-hoc at any operating "
            "point — the ROC curve is swept by adjusting the threshold, not the model."
        )

    st.info(
        "**Next tab →** The bridge from CFAR to ML: how the problem is reframed as a "
        "classification task, what training data is needed, and which architecture is used."
    )
