"""
tabs/math_tab.py — render() for tab_math.
"""

import streamlit as st


def render():
    st.markdown("## Signal & Detection Math")
    st.markdown(
        "Everything in this demo derives from four equations. "
        "Each section below shows the formula, what each symbol means, "
        "and how it maps to a line of code."
    )

    # ── 1. Radar signal model ──────────────────────────────────────────────────
    with st.expander("1 · Radar signal model", expanded=True):
        st.markdown(r"""
### Received power at range $r$

$$
P_r(r) = \underbrace{A}_{\text{target}} \cdot r^{-2} + \underbrace{\eta(r)}_{\text{clutter}}
$$

| Symbol | Meaning |
|--------|---------|
| $A$ | Target reflectivity, set by the **SNR (dB)** slider |
| $r^{-2}$ | Free-space path loss (two-way) — signal falls off as range squared |
| $\eta(r)$ | Range-dependent Rayleigh clutter: $\eta \sim r^{-2} \cdot \text{Rayleigh}(\sigma=1)$ |

The clutter envelope follows a **Rayleigh distribution** — a standard model for
ground/sea clutter where many small random scatterers add in quadrature.
The $r^{-2}$ factor means close-range cells are much brighter than far-range cells,
which is why we need range normalisation before any detection step.
""")
        st.code(
            "# src/data/ppi_generator.py\n"
            "noise_floor = np.arange(n_range) / n_range   # r / R_max\n"
            "clutter = rng.rayleigh(scale=noise_floor)     # Rayleigh(σ = r/R_max)\n"
            "# Target injected at configured (range_bin, azimuth_deg)\n"
            "snr_linear = 10 ** (snr_db / 10)\n"
            "sweep[az_b, r_b] += snr_linear * noise_floor[r_b]",
            language="python",
        )

    # ── 2. Temporal features ────────────────────────────────────────────────────
    with st.expander("2 · Temporal feature extraction", expanded=True):
        st.markdown(r"""
### Three-channel feature map fed to the CNN

For a sequence of $T$ sweeps $\{x_t\}_{t=1}^{T}$, each cell $(az, r)$ produces:

$$
f_{\max}(az,r) = \frac{\max_t\, x_t(az,r)}{\hat{\sigma}(r)}, \quad
f_{\mu}(az,r)  = \frac{\mu_t\, x_t(az,r)}{\hat{\sigma}(r)}, \quad
f_{\sigma}(az,r) = \frac{\sigma_t\, x_t(az,r)}{\hat{\sigma}(r)}
$$

where $\hat{\sigma}(r)$ is the **10th-percentile amplitude** at range bin $r$
(estimated from all azimuth cells in the sequence) — an empirical range-dependent clutter floor.

| Channel | Why it helps |
|---------|-------------|
| $f_{\max}$ | Captures peak illumination — a target brightens its cell on each look |
| $f_{\mu}$ | Provides the clutter baseline — stationary clutter has a stable mean |
| $f_{\sigma}$ | Motion fingerprint — a moving target has **high variance** across sweeps; clutter has low variance |

Dividing by $\hat{\sigma}(r)$ cancels range attenuation so all three channels
are range-invariant, making the CNN's job easier.
""")
        st.code(
            "# src/data/ppi_generator.py — temporal_features()\n"
            "noise_floor = np.percentile(seq, 10, axis=(0, 1))  # shape (n_range,)\n"
            "noise_floor = np.maximum(noise_floor, 1e-6)\n"
            "normed = seq / noise_floor[np.newaxis, np.newaxis, :]  # broadcast over (T, az, r)\n"
            "feat = np.stack([normed.max(0), normed.mean(0), normed.std(0)])",
            language="python",
        )

    # ── 3. CA-CFAR ─────────────────────────────────────────────────────────────
    with st.expander("3 · Constant False-Alarm Rate (CA-CFAR)", expanded=True):
        st.markdown(r"""
### Cell-Averaging CFAR threshold

A cell under test (CUT) at $(az, r)$ is declared a detection if:

$$
x(az, r) > \alpha \cdot \hat{\mu}_{\text{ref}}
$$

where

$$
\hat{\mu}_{\text{ref}} = \frac{1}{N_{\text{ref}}} \sum_{(i,j)\,\in\,\mathcal{R}} x(i, j)
$$

$\mathcal{R}$ is the **reference window** — a rectangular annulus centred on the CUT
with guard cells excluded to prevent target self-masking:

$$
N_{\text{ref}} = (2 r_a + 2 g_a + 1)(2 r_r + 2 g_r + 1) - (2 g_a + 1)(2 g_r + 1)
$$

| Parameter | Value | Role |
|-----------|-------|------|
| $g_a, g_r$ | 2 cells | Guard cells — exclude the target's own sidelobes |
| $r_a, r_r$ | 5 cells | Reference cells each side |
| $\alpha$ | 2.5 | Threshold factor — set empirically for ~1% FA rate |

**Pre-whitening**: before applying CFAR, each sweep is divided by the
$r^{-2}$ clutter-floor model so the threshold is range-invariant.
The reference window sum is computed in $O(N_{az} \times N_r)$
using a **2-D prefix sum**, making the algorithm fast regardless of window size.
""")
        st.code(
            "# src/baseline/ppi_cfar_kf.py\n"
            "# Pre-whiten: divide by R^-2 floor\n"
            "sweep = sweep / _rng_norm(sweep.shape[1])[np.newaxis, :]\n"
            "# 2-D prefix sum with azimuth wrap-around padding\n"
            "ps = padded.cumsum(axis=0).cumsum(axis=1)\n"
            "outer_sum = box(-outer_az, outer_az, -outer_r, outer_r)\n"
            "inner_sum = box(-ga, ga, -gr, gr)\n"
            "ref_sum   = outer_sum - inner_sum          # reference window\n"
            "noise_est = ref_sum / ref_cells\n"
            "det[:, ri] = cell_val > self.thr * noise_est",
            language="python",
        )

    # ── 4. Kalman filter tracker ────────────────────────────────────────────────
    with st.expander("4 · Kalman filter tracker", expanded=True):
        st.markdown(r"""
### State-space model in (range, azimuth) bins

State vector and constant-velocity dynamics:

$$
\mathbf{x} = \begin{bmatrix} r \\ \dot{r} \\ az \\ \dot{az} \end{bmatrix}, \qquad
\mathbf{F} = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

Observation matrix (we only measure position, not velocity):

$$
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$

**Predict step** (between sweeps):

$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\hat{\mathbf{x}}_{k-1}, \qquad
\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^\top + \mathbf{Q}
$$

**Update step** (on CFAR peak association):

$$
\mathbf{K} = \mathbf{P}_{k|k-1}\mathbf{H}^\top\mathbf{S}^{-1}, \qquad
\mathbf{S} = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^\top + \mathbf{R}
$$
$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}_{k|k-1})
$$

**Mahalanobis gating** — a CFAR detection $\mathbf{z}$ is associated to a track only if:

$$
(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})^\top \mathbf{S}^{-1} (\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}) < \gamma
$$

with gate $\gamma = 16$ (roughly a 4-bin radius in normalised innovation space).
A track is **confirmed** after $\geq 4$ successful associations (hits).

| Matrix | Values | Meaning |
|--------|--------|---------|
| $\mathbf{Q}$ | diag(1, 0.5, 1, 0.5) | Process noise — allows for manoeuvring |
| $\mathbf{R}$ | diag(4, 4) | Measurement noise — ±2 bin uncertainty in CFAR centroid |
""")
        st.code(
            "# src/baseline/ppi_cfar_kf.py — _KalmanTrack2D\n"
            "def predict(self):\n"
            "    self.x = self.F @ self.x\n"
            "    self.P = self.F @ self.P @ self.F.T + self.Q\n\n"
            "def update(self, z):\n"
            "    inn = z - self.H @ self.x          # innovation\n"
            "    S   = self.H @ self.P @ self.H.T + self.R\n"
            "    K   = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain\n"
            "    self.x = self.x + K @ inn\n"
            "    self.P = (np.eye(4) - K @ self.H) @ self.P",
            language="python",
        )

    # ── 5. CNN architecture ─────────────────────────────────────────────────────
    with st.expander("5 · CNN architecture", expanded=True):
        st.markdown(r"""
### Fully-convolutional network (FCN)

Input: **3 × 180 × 64** feature map (channels = max, mean, std).

$$
\text{Input} \xrightarrow{3\to16} \text{BN+ReLU} \xrightarrow{16\to32} \text{BN+ReLU}
\xrightarrow{32\to32} \text{BN+ReLU} \xrightarrow{32\to16} \text{BN+ReLU}
\xrightarrow{16\to1} \text{logit map}
$$

All convolutions are **3×3, stride 1, padding 1** — the spatial resolution
is preserved throughout (no pooling), so the output is a **1×180×64** logit map
converted to probabilities by sigmoid: $p = \sigma(\text{logit})$.

| Layer | Filters | Parameters |
|-------|---------|-----------|
| Conv1 | 16 | 3×3×3×16 = 432 |
| Conv2 | 32 | 3×3×16×32 = 4 608 |
| Conv3 | 32 | 3×3×32×32 = 9 216 |
| Conv4 | 16 | 3×3×32×16 = 4 608 |
| Conv5 | 1  | 3×3×16×1 = 144 |
| **Total** | | **~19 000** |

The network is tiny by deep-learning standards but sufficient because
the three temporal features already do the heavy lifting — the CNN
learns a compact spatial filter to suppress residual clutter.

Loss: binary cross-entropy on the per-cell logits against the
ground-truth label map (1 at target cell ± 1 bin, 0 elsewhere).
""")
        st.code(
            "# src/models/ppi_detector.py\n"
            "class PPIDetectorCNN(nn.Module):\n"
            "    def __init__(self, filters=(16, 32, 32, 16)):\n"
            "        layers = []\n"
            "        in_ch = 3\n"
            "        for out_ch in filters:\n"
            "            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),\n"
            "                       nn.BatchNorm2d(out_ch), nn.ReLU()]\n"
            "            in_ch = out_ch\n"
            "        layers.append(nn.Conv2d(in_ch, 1, 3, padding=1))\n"
            "        self.net = nn.Sequential(*layers)\n\n"
            "    def forward(self, x):          # x: (B, 3, n_az, n_range)\n"
            "        return self.net(x).squeeze(1)   # (B, n_az, n_range) logits",
            language="python",
        )

    # ── 6. ConvGRU streaming detector ───────────────────────────────────────────
    with st.expander("6 · Convolutional GRU — hidden state as confidence heatmap", expanded=True):
        st.markdown(r"""
### Why a GRU instead of the batch CNN?

The batch CNN needs **all N sweeps at once** to compute its three temporal features
(max, mean, std) — it cannot produce output until the full window has arrived.

The ConvGRU processes **one sweep at a time** and maintains its entire belief state
in a single spatial map $h$. Output is available after every sweep — latency is one
rotation period, not N.

---

### The hidden state IS the confidence heatmap

$$
h_t \in [0, 1]^{N_{az} \times N_{range}}
$$

This is the key design choice: $h$ is a **single-channel probability map** over
the full PPI grid, not an abstract multi-channel feature tensor. After every sweep
you can read $h$ directly as the detector's current belief — no further
transformation needed. This is the **purple overlay** on the PPI display.

Compare with the batch CNN, which produces a probability map only after all
sweeps have been processed.

---

### Update equations

At each sweep $t$, given the EMA-normalised amplitude $x_t$:

**Step 1 — encode** the sweep into a rich multi-channel feature map:
$$
f_t = \text{Encoder}(x_t), \quad f_t \in \mathbb{R}^{C \times N_{az} \times N_{range}}
$$
The encoder is two Conv2d + ReLU layers ($1 \to 16 \to 32$ channels).

**Step 2 — gate** using the concatenation $[f_t \,\|\, h_{t-1}]$:
$$
r_t = \sigma\!\bigl(\text{Conv}_r([f_t \,\|\, h_{t-1}])\bigr) \quad \text{(reset gate)}
$$
$$
z_t = \sigma\!\bigl(\text{Conv}_z([f_t \,\|\, h_{t-1}])\bigr) \quad \text{(update gate)}
$$

**Step 3 — candidate** using the gated old state:
$$
\tilde{h}_t = \sigma\!\bigl(\text{Conv}_n([f_t \,\|\, r_t \odot h_{t-1}])\bigr) \in [0,1]
$$
Using $\sigma$ (not $\tanh$) keeps the candidate in $[0,1]$, matching the probability
space of $h$.

**Step 4 — blend**:
$$
h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
$$
A convex combination of two values in $[0,1]$ stays in $[0,1]$ — the heatmap
remains a valid probability map at every sweep.

| Gate | Behaviour when → 0 | Behaviour when → 1 |
|------|---------------------|---------------------|
| Reset $r$ | Ignore old confidence when computing candidate | Let old state influence candidate |
| Update $z$ | Keep old confidence unchanged (copy-through) | Replace with new candidate evidence |

---

### Architecture and parameter count

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input $x_t$ | $(1,\,1,\,180,\,64)$ | EMA-normalised amplitude — one sweep |
| Encoder output | $(1,\,32,\,180,\,64)$ | Feature map from the two Conv layers |
| **$h$** | $(1,\,1,\,180,\,64)$ | **Confidence heatmap — the hidden state** |
| $r,\, z,\, \tilde{h}$ | $(1,\,1,\,180,\,64)$ each | Gates and candidate — single channel |

| Component | Parameters |
|-----------|------------|
| Encoder Conv2d(1→16) | 160 |
| Encoder Conv2d(16→32) | 4 640 |
| Conv_r, Conv_z, Conv_n each (33→1) | 3 × 298 = 894 |
| **Total** | **~5 700** |

---

### Streaming inference (O(1) per sweep)

```python
h = np.zeros((1, 1, 180, 64), dtype=np.float32)   # start with zero confidence

for sweep in live_radar_stream:
    sweep_norm = preprocess(sweep)                  # EMA normalise: (1, 1, 180, 64)
    prob_map, h = onnx_session.run(
        None, {"sweep_norm": sweep_norm, "h_in": h}
    )
    # prob_map IS h — the heatmap you see in the purple overlay
    detections = (prob_map[0, 0] > 0.30)           # threshold → binary map
```

Each call costs the same compute regardless of how long the antenna has been
running. The heatmap builds up confidence at the target cell as more sweeps arrive
and decays where no evidence is seen — visible in real time on the PPI display.
""")
        st.code(
            "# src/model/conv_gru.py — ConvGRUDetector.forward()\n"
            "def forward(self, sweep_norm, h):\n"
            "    feat = self.encoder(sweep_norm)                  # (B, 32, H, W)\n"
            "    xh   = torch.cat([feat, h], dim=1)               # (B, 33, H, W)\n"
            "    r    = torch.sigmoid(self.r_gate(xh))            # reset gate ∈ [0,1]\n"
            "    z    = torch.sigmoid(self.z_gate(xh))            # update gate ∈ [0,1]\n"
            "    n    = torch.sigmoid(self.cand(                  # candidate ∈ [0,1]\n"
            "               torch.cat([feat, r * h], dim=1)))\n"
            "    h_next = (1.0 - z) * h + z * n                  # stays in [0,1]\n"
            "    return h_next, h_next                            # prob_map ≡ h_out",
            language="python",
        )

    # ── 7. LRT — non-coherent integrator ───────────────────────────────────────
    with st.expander("7 · Likelihood Ratio Test — non-coherent square-law integrator", expanded=False):
        st.markdown(r"""
### Neyman-Pearson optimal detector

The **Neyman-Pearson Lemma** states that among all tests at a fixed false-alarm rate $P_{fa}$,
the one maximising detection probability $P_d$ is the **likelihood ratio test**:

$$
\Lambda(\mathbf{x}) = \frac{p(\mathbf{x} \mid H_1)}{p(\mathbf{x} \mid H_0)} \gtrless \eta
$$

#### Single-sweep case — Rayleigh clutter

Under $H_0$ (clutter only), each amplitude sample $x$ follows a Rayleigh distribution:

$$
p(x \mid H_0) = \frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}, \quad x \geq 0
$$

Under $H_1$ (target + clutter), the signal-plus-noise envelope follows a Rice distribution.
Taking the log-LRT and discarding constants, the optimal single-sweep detector reduces to:

$$
x(az,r) \gtrless \eta' \quad \text{(amplitude threshold — the same structure as CFAR)}
$$

#### Multi-sweep non-coherent integration (GLRT at low SNR)

Across $N$ independent sweeps and unknown target amplitude $A$, the Generalised LRT
(GLRT — maximise over $A$) at low SNR yields the **square-law combiner**:

$$
\boxed{
\Lambda(az,r) = \sum_{k=1}^{N} \left(\frac{x_k(az,r)}{\hat{\sigma}(r)}\right)^2 \gtrless \eta
}
$$

$\hat{\sigma}(r)$ is the range-dependent noise floor (10th-percentile amplitude across azimuths).
Under $H_0$ this statistic follows a **chi-squared distribution** with $2N$ degrees of freedom,
giving analytical control over $P_{fa}$.

#### Integration gain

For a stationary target, summing $N$ independent noise-limited observations gives a
coherent SNR improvement of:

$$
\text{SNR}_{\text{integrated}} \approx N \cdot \text{SNR}_{\text{single}} \quad \text{(3 dB per doubling of } N\text{)}
$$

In this demo $N = 10$, so the theoretical gain is $10\,\log_{10}(10) \approx +10\,\text{dB}$.

| Assumption | Consequence if violated |
|---|---|
| Rayleigh clutter | Optimal threshold changes; performance degrades |
| Stationary target | Energy spread across cells → integration loss |
| Known noise floor | Bias in $\Lambda$ → $P_{fa}$ deviates from design |
""")
        st.code(
            "# radar/detection.py — _lrt_score()\n"
            "noise_floor = np.percentile(ppi, 10,\n"
            "                 axis=(1, 2), keepdims=True).clip(1e-6)\n"
            "normed = ppi / noise_floor        # (N, n_az, n_range)\n"
            "score  = (normed ** 2).sum(0)     # (n_az, n_range)  chi-sq(2N) under H0",
            language="python",
        )

    # ── 8. DP-TBD ───────────────────────────────────────────────────────────────
    with st.expander("8 · Dynamic-programming Track-Before-Detect (DP-TBD)", expanded=False):
        st.markdown(r"""
### Motivation

The LRT accumulates energy at a **fixed** cell. A moving target spreads across cells,
incurring an *integration loss* proportional to the number of cells traversed.
**Track-Before-Detect (TBD)** avoids this by accumulating raw amplitude along
*all plausible trajectories simultaneously* — delaying the threshold decision until
all sweeps have been processed.

### DP recursion (Barniv 1985 / Viterbi-style)

Let $\tilde{x}_k(az,r)$ be the normalised amplitude at cell $(az,r)$ in sweep $k$.
Define the **cumulative trajectory score** recursively:

$$
S_0(az,r) = \tilde{x}_0(az,r)
$$

$$
\boxed{
S_k(az,r) = \tilde{x}_k(az,r) +
\max_{\substack{|\Delta az| \leq v_{az} \\ |\Delta r| \leq v_r}}
S_{k-1}(az + \Delta az,\; r + \Delta r)
}
$$

After $N$ sweeps, **declare a detection** at any cell $(az^*, r^*)$ where $S_N > \eta$.
The trajectory can be recovered by back-tracking the $\arg\max$ at each step.

### The max-filter interpretation

The inner $\max$ over the velocity neighbourhood is a **2-D maximum filter** with
kernel size $(2v_{az}+1) \times (2v_r+1)$, applied to the previous score map.
`scipy.ndimage.maximum_filter` computes this in $O(N_{az} \cdot N_r)$ per sweep
using a sliding rectangular window — far faster than explicit looping.

$$
S_k = \tilde{x}_k + \text{MaxFilter}_{(2v_{az}+1,\,2v_r+1)}(S_{k-1})
$$

### Complexity and parameters

| Parameter | This demo | Effect of increasing |
|---|---|---|
| Max radial velocity $v_r$ | 3 bins/sweep | Larger neighbourhood → more SNR gain, more false alarms |
| Max angular velocity $v_{az}$ | 2 bins/sweep | Same trade-off |
| Sweeps $N$ | 10 | Linear score increase for real targets |
| Complexity | $O(N \cdot N_{az} \cdot N_r)$ | Dominated by max-filter, independent of $v$ |

### Comparison with LRT

| Property | LRT | DP-TBD |
|---|---|---|
| Moving targets | Integration loss | Full $N$-sweep gain |
| Stationary targets | Optimal | Equivalent |
| Threshold $\eta$ | Analytically set (chi-sq $P_{fa}$) | Empirical (depends on trajectory density) |
| Complexity | $O(N \cdot N_{az} \cdot N_r)$ | Same (max-filter) |
| Memory | One score map | One score map (back-track path needs $N$ maps) |
""")
        st.code(
            "# radar/detection.py — _dp_tbd_score()\n"
            "from scipy.ndimage import maximum_filter\n"
            "\n"
            "noise_floor = np.percentile(ppi, 10,\n"
            "                 axis=(1,2), keepdims=True).clip(1e-6)\n"
            "normed = (ppi / noise_floor).astype(np.float32)  # (N, n_az, n_r)\n"
            "S      = normed[0].copy()\n"
            "size   = (2*max_vaz + 1, 2*max_vr + 1)\n"
            "for k in range(1, N):\n"
            "    best = maximum_filter(S, size=size,\n"
            "                mode=('wrap', 'nearest'))  # azimuth wraps, range clips\n"
            "    S    = normed[k] + best\n"
            "# S[az, r] = accumulated score of the best trajectory ending at (az,r)",
            language="python",
        )
