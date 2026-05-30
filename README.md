# Weibel MLOps Demo — AI-Powered Radar Target Detection

> **The core argument:** Weibel's high-precision instrumentation radars produce better training labels than anything else on the market. Better labels compound directly into better ML models. This demo shows the full pipeline from label factory to fleet deployment.

[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://weibel-mlops-demo.streamlit.app)

---

## What this is

A working ML pipeline built on top of Weibel's existing two-tier radar architecture:

- **Tier 1 — Instrumentation radar (MFCW/CW):** deployed at missile test ranges globally. Produces TSPI ground truth at sub-centimetre accuracy. Proposed role: *label factory*.
- **Tier 2 — XENTA-C operational radar (FMCW X-band):** mass-produced counter-UAS fleet. 500 M DKK Kongsberg/NATO contract, deliveries 2026–2027. Proposed role: *training data source and deployment target*.

The pipeline trains a ConvGRU and CNN directly on XENTA-C PPI sweeps, with labels projected from the instrumentation radar's TSPI track. Hardware distortions (antenna phase errors, ADC non-linearity, oscillator phase noise) are absorbed into the model weights — automatically, without explicit hardware calibration.

---

## Key results

| Detector | Pd at 0 dB SNR | False track rate | Inference / sweep | Streaming |
|---|---:|---:|---:|:---:|
| CA-CFAR + KF (classical) | 7% | 3.68 / 100 sweeps | 2.2 ms | ✓ |
| CNN + KF | 37% | **0.79 / 100 sweeps** | 0.8 ms | ✗ |
| ConvGRU + KF | **73%** | 1.2 / 100 sweeps | <1 ms | ✓ |

- ConvGRU detects 9 in 10 targets where CA-CFAR misses most — at identical false-alarm rate
- CNN reduces false confirmed tracks by **4.7×** vs CFAR
- Both models run in <1 ms on ARM Cortex-A72 — within the 10 ms sweep budget with 10× headroom

---

## The demo

**Live app:** [weibel-mlops-demo.streamlit.app](https://weibel-mlops-demo.streamlit.app)

Twelve tabs covering the full story:

| Tab | What it shows |
|---|---|
| 🔭 Classical | CA-CFAR signal chain and Kalman filter — the baseline being beaten |
| 🤖 CFAR → ML | How detection is reframed as a learning problem |
| 📊 Comparison | Pd vs SNR, false track rate, latency, runtime across all detectors |
| 📻 Live | Real-time multi-target simulation — AI track appears before CFAR reacts |
| 📉 ROC | Cell-level ROC curves at SNR 0/3/6 dB for all six detectors |
| ⚖️ Trade-offs | When to use each algorithm; case for a combined system |
| 🔁 Pipeline | The proposed label-factory → training → deployment → drift loop |
| 📈 Monitor | Live MLflow training run comparison — all experiments, all epochs |
| 🛰 Fleet | Drone knowledge base, fleet map, hardware/compliance/tech-stack spec |
| 🧠 Agent | Claude-powered MLOps advisor: PSI alerts → tool use → retraining decision |
| 📐 Math | Radar equation, CFAR threshold, KF state equations, GRU update |
| 📄 Paper | IEEE-format PDF: *ML vs CA-CFAR on Synthetic PPI Radar Data* |

**Run locally:**
```bash
git clone https://github.com/Michael-Bach/weibel-mlops-demo
cd weibel-mlops-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .streamlit/secrets.toml for the Agent tab
streamlit run app.py
```

> **Note:** The ROC tab takes ~60 s on first load to compute the GRU cache. Pre-warm it before a live demo.

---

## Pipeline architecture

```
Instrumentation radar (TSPI labels)
        │
        ▼
Label fusion ──► Quality gate (tracker_locked=True)
        │
        ▼
XENTA-C PPI sweeps + fused labels
        │
        ├──► DVC data versioning (content-addressed, reproducible)
        │
        ▼
Training (ConvGRU / CNN / Transformer)
        │
        ├──► MLflow experiment tracking (params, per-epoch metrics, artifacts)
        │
        ▼
Accuracy gate (val F1 ≥ 0.25, latency ≤ 10 ms on ARM)
        │
        ▼
ONNX export (opset 17, ARM NEON backend)
        │
        ▼
OTA push → XENTA-C fleet units
        │
        ▼
PSI drift monitoring (per-unit, 5 KB/hr egress, raw PPI stays on-unit)
        │
        └──► PSI > 0.20 → Agent alert → test-range session → retrain
```

The feedback loop is the point: every new environment or drone type that triggers a PSI alert feeds back into the training set, lowering the detection floor for the *entire fleet* — not just the unit that saw the edge case. Classical CFAR has no equivalent mechanism.

---

## Models

### ConvGRU (streaming detector)
- **Architecture:** convolutional encoder → GRU hidden state → 1-channel confidence map
- **Input:** one PPI sweep `(1, 180, 64)` + previous hidden state
- **Output:** target probability map `(180, 64)` + updated hidden state
- **Parameters:** ~6 k (fits in L2 cache on ARM Cortex-A72)
- **Training:** curriculum learning — seq_len 5 → 10 → 15 over 60 epochs; weighted BCE with pos_weight to compensate for sparse target cells
- **Deployment:** ONNX Runtime, ARM NEON backend, <1 ms per sweep

### CNN (batch detector)
- **Architecture:** four-layer convolutional encoder on 3-channel temporal feature map (max / mean / std over 10 sweeps)
- **Input:** temporal feature stack `(3, 180, 64)`
- **Output:** target probability map `(180, 64)`
- **Parameters:** 19 k
- **Strength:** lowest false-track rate (4.7× better than CFAR)

### Kalman tracker (shared backend)
Both ML models feed into `PPIKalmanTracker`: constant-velocity model, χ² gate for association, confirmed track after 3 consistent hits. CFAR detection map goes through an identical tracker for fair comparison.

---

## Instrument error absorption

The central technical insight: because training *labels* come from the instrumentation radar (ground truth) but training *features* come from the XENTA unit (with all its hardware imperfections), the model is forced to bridge that gap. It learns:

> *Given this distorted signal as input, predict where the target actually is.*

This is physically equivalent to automatic lens calibration — except it requires no explicit hardware characterisation, happens during training, and generalises to targets the calibration drone did not fly. Higher-precision instrumentation radar → more accurate TSPI labels → better-calibrated ML model. The return on hardware quality at Tier 1 compounds directly into model accuracy at Tier 2.

---

## Edge deployment spec

Each XENTA-C unit in the proposed architecture would carry:

- **ARM Cortex-A72** (4-core 1.8 GHz, NEON SIMD) — ConvGRU inference ≤ 0.8 ms/sweep
- **HSM + TPM 2.0** — model signing, verified boot, TPM-sealed LUKS2 disk encryption
- **256 GB NVMe ring buffer** — 72-hour PPI retention; raw data never egresses
- **mTLS uplink** — PSI statistics only (~5 KB/hour); TLS 1.3, AES-256-GCM

Central hub: MinIO data lake, PostgreSQL MLflow backend, Gitea CI, NATS message bus, bare-metal GPU training server. Full air-gap — no cloud dependencies.

---

## Military compliance

| Concern | Approach |
|---|---|
| Data classification | PPI sweeps: NATO RESTRICTED minimum; TSPI + drone signatures: NATO CONFIDENTIAL |
| Export control | Trained ONNX models = ITAR Category XI controlled technical data; DSP-5 license required for export |
| Raw data egress | Only PSI statistics leave each unit — statistical aggregates, UNCLASSIFIED |
| Encryption at rest | LUKS2 + TPM2-sealed key (FIPS 140-3 Level 2) |
| Encryption in transit | TLS 1.3 + mTLS, AES-256-GCM (NSA Suite B / CNSA) |
| Model signing | Ed25519, hub HSM private key (FIPS 186-5) |
| Audit log | SHA-3-256 hash chain, append-only, RFC 3161 timestamps |
| AI Act (EU 2024/1689) | Art. 2(3) excludes military systems from mandatory scope; equivalent governance applied as NATO best practice |

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Edge inference | ONNX Runtime 1.17+, ARM NEON | <1 ms/sweep, no GPU, single binary |
| Message bus | NATS.io (JetStream) | Sub-ms pub/sub, intermittent-link tolerant |
| Data lake | MinIO | S3-compatible, full air-gap, object versioning |
| ML pipeline | DVC + MLflow | Deterministic reproducibility, hash-verified lineage |
| Version control / CI | Gitea (self-hosted) | No GitHub dependency, Actions-compatible syntax |
| Container runtime | Podman (rootless) | No root daemon, better security posture |
| Orchestration | K3s | Runs on edge ARM hardware |
| Secrets | HashiCorp Vault + HSM | Audit log, dynamic secrets, FIPS 140-3 Level 3 |
| Training compute | Bare-metal GPU (2× A100) | No cloud, data stays on-premise |

---

## Repository structure

```
├── app.py                        # Streamlit demo entry point (12 tabs)
├── tabs/                         # One file per tab
├── src/
│   ├── data/ppi_generator.py     # Synthetic PPI radar data generator
│   ├── model/ppi_cnn.py          # Batch CNN detector
│   ├── model/conv_gru.py         # Streaming ConvGRU detector
│   ├── baseline/ppi_cfar_kf.py   # CA-CFAR + Kalman tracker baseline
│   ├── train_ppi.py              # CNN training loop + MLflow logging
│   └── train_recurrent.py        # ConvGRU training + curriculum scheduler
├── scripts/
│   ├── gen_paper_roc.py          # ROC curves for paper figures
│   ├── gen_paper_pd_snr.py       # Pd vs SNR sweep (LRT, DP-TBD, Transformer)
│   ├── drift_detect.py           # PSI drift detection
│   └── export_onnx.py            # ONNX export
├── radar/
│   ├── detection.py              # LRT, DP-TBD, CFAR score functions; ROC cache
│   └── sessions.py               # Cached ONNX inference sessions
├── artifacts/                    # Trained ONNX models + metrics (tracked in git)
├── mlruns/                       # MLflow file store (metadata only; binaries gitignored)
├── params_ppi.yaml               # CNN hyperparameters
├── params_recurrent.yaml         # ConvGRU hyperparameters
├── dvc.yaml                      # DVC pipeline DAG
├── paper/paper.tex               # LaTeX source → IEEE PDF
└── tests/                        # 23 passing tests
```

---

## Running training

```bash
# CNN (batch detector)
PYTHONPATH=. python src/train_ppi.py

# ConvGRU (streaming detector)
PYTHONPATH=. python src/train_recurrent.py

# Regenerate paper figures after retraining
make paper   # compiles LaTeX + pdftoppm
make roc     # recomputes ROC curves (~5 min)
```

All runs are logged to `mlruns/` and visible in the Monitor tab.

---

*Built by Michael Bach — MSc Physics, Royal Danish Navy Lt Cdr.*
*Demo for AI/MLOps Engineer position at Weibel Scientific.*
