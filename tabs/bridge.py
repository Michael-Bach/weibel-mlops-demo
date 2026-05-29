"""
tabs/bridge.py — render() for tab_bridge.
"""

import streamlit as st


def render():
    st.markdown("## From CFAR to ML: reframing radar detection as a learning problem")
    st.markdown(
        "CFAR is a hand-crafted statistical decision rule. The ML alternative replaces the "
        "hand-crafted rule with a *learned* one — trained on examples of what targets and "
        "clutter look like, rather than on a closed-form noise model."
    )

    st.divider()

    # ── Problem reframing ─────────────────────────────────────────────────────
    st.markdown("### 1 — Reframing the problem")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Classical framing**")
        st.markdown(
            "For each cell (az, r) in each sweep:\n"
            "- Is this amplitude value above a locally-derived threshold?\n"
            "- Binary decision per cell per sweep\n"
            "- Threshold derived analytically from a noise model\n"
            "- No learning, no memory between sweeps beyond the KF"
        )
    with col_r2:
        st.markdown("**ML framing**")
        st.markdown(
            "Given a sequence of sweeps:\n"
            "- Produce a probability map P(target | sweeps 1..k) for every cell\n"
            "- Continuous output that can be thresholded at any operating point\n"
            "- The decision boundary is *learned from data*, not derived from a noise model\n"
            "- Temporal evidence integration is built into the architecture"
        )

    st.divider()

    # ── What the model sees ───────────────────────────────────────────────────
    st.markdown("### 2 — What the model sees")
    st.markdown(
        "A moving target leaves a distinctive signature in the PPI sequence: "
        "it appears at a slightly different (az, r) position each sweep, producing a "
        "streak in the *temporal maximum* image and elevated variance in the *temporal std* "
        "image. Static clutter (ground return, buildings) stays fixed and averages out "
        "in the std channel."
    )
    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        st.markdown("**CNN batch approach**")
        st.code(
            "# Temporal feature map (3 channels)\n"
            "noise_fl = percentile(ppi, 10, axis=0)\n"
            "norm     = ppi / noise_fl        # (10, 180, 64)\n"
            "\n"
            "ch_max  = norm.max(axis=0)       # bright where target was\n"
            "ch_mean = norm.mean(axis=0)      # local noise estimate\n"
            "ch_std  = norm.std(axis=0)       # high where target moved\n"
            "\n"
            "X = stack([ch_max, ch_mean, ch_std])  # (3, 180, 64)\n"
            "# Feed to CNN once after all 10 sweeps",
            language="python",
        )
        st.markdown(
            "✓ Simple, fast, easy to train  \n"
            "✗ Requires the full 10-sweep window before inference  \n"
            "✗ Temporal statistics conflate multiple simultaneous targets"
        )
    with col_inp2:
        st.markdown("**ConvGRU streaming approach**")
        st.code(
            "# Process one sweep at a time\n"
            "h = zeros(1, 180, 64)   # hidden state = P(target)\n"
            "\n"
            "for sweep in sequence:\n"
            "    norm = sweep / ema_noise_floor\n"
            "    h    = ConvGRU(norm, h)\n"
            "    # h is now the detection probability map\n"
            "    # Threshold h to get peaks → feed KF\n"
            "\n"
            "# Hidden state h IS the output:\n"
            "#   h[az, r] ≈ P(target at (az,r) | sweeps seen so far)",
            language="python",
        )
        st.markdown(
            "✓ Streaming: one sweep at a time, no waiting  \n"
            "✓ Multi-target: each cell in h is independent  \n"
            "✓ Natural integration: h accumulates evidence over time  \n"
            "✓ Matches real radar operation"
        )

    st.divider()

    # ── Training data ─────────────────────────────────────────────────────────
    st.markdown("### 3 — Training data: synthetic vs real")
    st.markdown(
        "A radar ML model needs labelled examples: PPI sequences with known target "
        "positions (positive class) and sequences without targets (negative class). "
        "There are two sources."
    )

    col_syn, col_real = st.columns(2)
    with col_syn:
        st.markdown("**Synthetic data** *(used in this demo)*")
        st.markdown(
            "Generate amplitude cubes from a physical model:\n\n"
            "- **Target**: Gaussian beam profile × range-gate pulse × SNR amplitude, "
            "moving across the PPI at a random velocity\n"
            "- **Clutter**: Rayleigh-distributed with range-dependent R⁻² amplitude floor "
            "and log-normal speckle\n"
            "- **SNR** drawn uniformly from −20 to +40 dB\n\n"
            "Reproducible (seeded), free, arbitrarily large, no classification issues.\n\n"
            "**Risk:** if the real clutter statistics differ significantly from the Rayleigh "
            "model, the model will over-fit to the synthetic distribution and degrade on real data. "
            "Drift detection (PSI metric) catches this in deployment."
        )
        st.code(
            "# This demo — full pipeline\n"
            "python src/data/generate.py   # seed-controlled\n"
            "dvc repro                     # cached DAG\n"
            "# All parameters in params_ppi.yaml",
            language="bash",
        )
    with col_real:
        st.markdown("**Real data** *(production requirement)*")
        st.markdown(
            "Record actual radar returns and label them:\n\n"
            "- Manual labelling by radar operators, or\n"
            "- Cooperative targets (AIS cross-referencing), or\n"
            "- Fusion with a second, independent sensor\n\n"
            "Captures actual clutter statistics (sea state, terrain, rain) that the "
            "synthetic model can only approximate.\n\n"
            "**Workflow:** train on synthetic → validate on real → identify mismatch → "
            "collect labelled real examples → add to training set → retrain via pipeline "
            "→ promote if accuracy gate passes."
        )
        st.info(
            "In classified or air-gapped environments, real data stays on-prem. "
            "This is exactly why the demo includes a self-hosted Gitea pipeline and "
            "MLflow with a local SQLite backend — no data needs to leave the network."
        )

    st.divider()

    # ── Architecture summary ──────────────────────────────────────────────────
    st.markdown("### 4 — Architecture summary")
    st.code(
        "ConvGRU (streaming detector)\n"
        "────────────────────────────────────────────────\n"
        "Input each sweep:   sweep_norm  (1, 180, 64)  normalised amplitude\n"
        "                    h_in        (1, 180, 64)  previous hidden state\n"
        "\n"
        "Encoder:  4 × Conv2d(1→16→32→32→16, k=3) + BN + ReLU\n"
        "GRU gate: ConvGRU cell — produces h_out ∈ [0,1]^(1,180,64)\n"
        "\n"
        "Output:   h_out  =  P(target | sweeps 1..k)   per cell\n"
        "          (identical to h_in for the next sweep)\n"
        "\n"
        "Export:   ONNX opset 17 — onnxruntime on embedded Linux or FPGA via Vitis AI\n"
        "Latency:  < 1 ms per sweep on CPU  (p50, batch_size=1)",
        language="text",
    )

    st.divider()

    # ── Pipeline value ────────────────────────────────────────────────────────
    st.markdown("### 5 — Why the MLOps pipeline matters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**Reproducibility**")
        st.markdown(
            "DVC locks the exact dataset version, code commit, and hyperparameters "
            "that produced each model artifact. Any deployed model can be reproduced "
            "bit-identically from `dvc repro` on the tagged commit."
        )
    with col_p2:
        st.markdown("**Adaptation**")
        st.markdown(
            "Operating clutter statistics change (new area, seasonal sea state, hardware "
            "aging). PSI drift detection fires a Prometheus alert when the incoming "
            "distribution diverges from the training reference. The retraining loop is "
            "triggered automatically — CFAR has no equivalent mechanism."
        )
    with col_p3:
        st.markdown("**Safety gates**")
        st.markdown(
            "No model reaches production without passing the accuracy gate in CI, human "
            "review in the MLflow registry, and ONNX latency validation on the target "
            "hardware. Rollback is one `register_model.py` command — restore the previous "
            "Production version from the archive."
        )

    st.info(
        "**Next tab →** The Math: the CFAR threshold formula, KF equations, "
        "ConvGRU forward pass, and ONNX export details."
    )
