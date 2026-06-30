# Signal Classification MLOps Pipeline

End-to-end pipeline for binary signal classification on 2D scan data — synthetic data generation through ONNX edge export, with human-gated model promotion and full experiment lineage.

[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://weibel-mlops-demo.streamlit.app)

> **Provenance note:** This pipeline was originally built as a technical assessment for a radar AI engineering role. It is included here as a sample of MLOps pipeline design — not as a claim of experience in any specific sensing domain. The signal model and infrastructure are the transferable parts; the radar framing is context, not expertise.

---

## What this is

A working ML pipeline that trains CNN and ConvGRU classifiers to distinguish a signal-of-interest from background interference on synthetic 2D scan frames. The infrastructure is the point: the models are small and the data is synthetic, but the pipeline from data generation to edge deployment is production-shaped.

**The classification problem:** given a sequence of 2D signal frames (bearing × range, Rayleigh background + Gaussian signal-of-interest), detect the presence and location of a moving target at SNR levels down to −20 dB. Binary label per cell: signal-of-interest vs. background/noise.

---

## Key results

| Detector | Pd at 0 dB SNR | False track rate | Inference / frame | Streaming |
|---|---:|---:|---:|:---:|
| Threshold baseline (CA-CFAR + KF) | 7% | 3.68 / 100 frames | 2.2 ms | ✓ |
| CNN + KF | 37% | **0.79 / 100 frames** | 0.8 ms | ✗ |
| ConvGRU + KF | **73%** | 1.2 / 100 frames | <1 ms | ✓ |

- ConvGRU detects 9 in 10 targets where the threshold baseline misses most — at identical false-alarm rate
- CNN reduces false confirmed tracks by **4.7×** vs threshold baseline
- Both models run in <1 ms on ARM Cortex-A72 — within a 10 ms frame budget with 10× headroom

---

## Engineering highlights

These are the decisions that matter for an MLOps portfolio:

**Human promotion gate.** A trained checkpoint is never exported to ONNX automatically. The pipeline pauses after training, logs metrics to MLflow, and requires an explicit promotion step. This prevents a noisy training run from silently replacing a production model.

**CI quality gate.** `sys.exit(1)` if val F1 < `baseline_f1` (default 0.25). A regression in model quality fails the build — not just a warning, a hard stop. The threshold is tracked in `params_*.yaml` and versioned alongside the model.

**On-prem / air-gapped CI.** CI runs on self-hosted Gitea with Actions-compatible syntax. No GitHub dependency. Relevant for enterprise or regulated environments where pushing training data or model weights to a cloud CI runner is not acceptable.

**Dynamic-batch ONNX export.** Models export with a dynamic batch axis so the same ONNX graph runs on ONNX Runtime, TensorRT, OpenVINO, and FPGA toolchains without re-export. The ConvGRU hidden state is threaded through the graph as a static input/output, enabling streaming inference without unrolling.

**DVC + MLflow lineage.** Every training run is reproducible: DVC tracks data versions by content hash; MLflow logs params, per-epoch metrics, and artifact paths. Given a run ID, the exact data, code, and hyperparameters that produced a checkpoint are recoverable.

---

## Pipeline architecture

```
Synthetic data generator (Rayleigh background + Gaussian signal)
        │
        ▼
DVC data versioning (content-addressed, reproducible)
        │
        ▼
Training (ConvGRU / CNN) with MLflow experiment tracking
        │                     (params, per-epoch metrics, artifacts)
        ▼
CI quality gate ── val F1 ≥ baseline_f1 → pass; else sys.exit(1)
        │
        ▼
Human promotion step (explicit sign-off before export)
        │
        ▼
ONNX export (opset 17, dynamic batch, streaming-compatible)
        │
        ▼
Edge deployment (ONNX Runtime / TensorRT / OpenVINO / FPGA)
        │
        ▼
PSI drift monitoring (statistics only — raw data stays on-device)
        │
        └──► PSI > 0.20 → alert → data collection → retrain
```

The feedback loop matters: edge devices ship only population stability index (PSI) statistics, not raw frames. Drift triggers a targeted data collection run, which feeds back into the training set. The threshold baseline has no equivalent mechanism.

---

## Models

### ConvGRU (streaming classifier)
- **Architecture:** convolutional encoder → GRU hidden state → 1-channel confidence map
- **Input:** one scan frame `(1, 180, 64)` + previous hidden state
- **Output:** signal-of-interest probability map `(180, 64)` + updated hidden state
- **Parameters:** ~6 k (fits in L2 cache on ARM Cortex-A72)
- **Training:** curriculum learning — seq_len 5 → 10 → 15 over 60 epochs; weighted BCE with pos_weight to handle sparse positive cells (~31:1 background/foreground ratio)
- **Deployment:** ONNX Runtime, ARM NEON backend, <1 ms per frame

### CNN (batch classifier)
- **Architecture:** four-layer convolutional encoder on 3-channel temporal feature map (max / mean / std over 10 frames)
- **Input:** temporal feature stack `(3, 180, 64)`
- **Output:** signal-of-interest probability map `(180, 64)`
- **Parameters:** 19 k
- **Strength:** lowest false-track rate (4.7× better than threshold baseline)

### Kalman tracker (shared backend)
Both ML models feed into a constant-velocity Kalman tracker with χ² gating for detection association, confirmed track after 3 consistent hits. The threshold baseline uses an identical tracker for a fair comparison.

---

## Synthetic signal model

The data generator (`src/data/ppi_generator.py`) produces 2D scan frames on a bearing × range grid:

- **Background:** Rayleigh-distributed amplitude with a range-dependent floor (R⁻² attenuation) and log-normal speckle. Produces realistic heterogeneous interference including clutter patches and range-dependent noise floor.
- **Signal-of-interest:** Gaussian directivity envelope × Gaussian range-gate envelope, scaled to a target SNR drawn uniformly from `snr_range` (default −20 to +40 dB). Position and velocity are randomized per sample.
- **Labels:** soft Gaussian maps (σ = 5 bins) centered on the signal position, union across sweeps for the CNN.

No domain-specific parameters (frequencies, wavelengths, PRFs) are embedded. Grid dimensions, beam width, and pulse width are configurable via `params_*.yaml` and are purely synthetic.

---

## Edge deployment spec

Proposed target hardware per the original design:

- **ARM Cortex-A72** (4-core 1.8 GHz, NEON SIMD) — ConvGRU inference ≤ 0.8 ms/frame
- **HSM + TPM 2.0** — model signing, verified boot, TPM-sealed LUKS2 disk encryption
- **256 GB NVMe ring buffer** — 72-hour frame retention; raw data never egresses
- **mTLS uplink** — PSI statistics only (~5 KB/hour); TLS 1.3, AES-256-GCM

Central hub: MinIO data lake, PostgreSQL MLflow backend, Gitea CI, NATS message bus, bare-metal GPU training server. Full air-gap — no cloud dependencies.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Edge inference | ONNX Runtime 1.17+, ARM NEON | <1 ms/frame, no GPU, single binary |
| Message bus | NATS.io (JetStream) | Sub-ms pub/sub, intermittent-link tolerant |
| Data lake | MinIO | S3-compatible, full air-gap, object versioning |
| ML pipeline | DVC + MLflow | Deterministic reproducibility, hash-verified lineage |
| Version control / CI | Gitea (self-hosted) | No cloud dependency, Actions-compatible syntax |
| Container runtime | Podman (rootless) | No root daemon, better security posture |
| Orchestration | K3s | Runs on edge ARM hardware |
| Secrets | HashiCorp Vault + HSM | Audit log, dynamic secrets, FIPS 140-3 Level 3 |
| Training compute | Bare-metal GPU (2× A100) | No cloud, data stays on-premise |

---

## Security posture

| Concern | Approach |
|---|---|
| Data egress | Only PSI statistics leave each device — statistical aggregates, no raw frames |
| Encryption at rest | LUKS2 + TPM2-sealed key (FIPS 140-3 Level 2) |
| Encryption in transit | TLS 1.3 + mTLS, AES-256-GCM |
| Model signing | Ed25519, hub HSM private key (FIPS 186-5) |
| Audit log | SHA-3-256 hash chain, append-only, RFC 3161 timestamps |

---

## Repository structure

```
├── app.py                          # Streamlit demo entry point
├── tabs/                           # One file per demo tab
├── src/
│   ├── data/ppi_generator.py       # Synthetic 2D scan data generator
│   ├── data/sequence_dataset.py    # Variable-length sequence dataset for ConvGRU
│   ├── model/ppi_cnn.py            # Batch CNN classifier
│   ├── model/conv_gru.py           # Streaming ConvGRU classifier
│   ├── train_ppi.py                # CNN training loop + MLflow logging
│   ├── train_recurrent.py          # ConvGRU training + curriculum scheduler
│   └── train_transformer.py        # Transformer variant
├── scripts/
│   ├── gen_paper_pd_snr.py         # Pd vs SNR sweep figures
│   └── drift_detect.py             # PSI drift detection
├── radar/
│   ├── detection.py                # Threshold detector, ROC cache, score functions
│   └── sessions.py                 # Cached ONNX inference sessions
├── artifacts/                      # Trained ONNX models + metrics (tracked in git)
├── mlruns/                         # MLflow file store (metadata only; binaries gitignored)
├── params_ppi.yaml                 # CNN hyperparameters
├── params_recurrent.yaml           # ConvGRU hyperparameters
├── params_recurrent_tuned.yaml     # Tuned ConvGRU run
├── dvc.yaml                        # DVC pipeline DAG
├── paper/paper.tex                 # LaTeX source — IEEE-format writeup
└── tests/                          # Test suite
```

---

## Running training

```bash
# CNN (batch classifier)
PYTHONPATH=. python src/train_ppi.py

# ConvGRU (streaming classifier)
PYTHONPATH=. python src/train_recurrent.py

# Regenerate figures after retraining
make paper   # compiles LaTeX + pdftoppm
make roc     # recomputes ROC curves (~5 min)
```

All runs are logged to `mlruns/` and visible in the Monitor tab of the demo app.

**Run the demo locally:**
```bash
git clone https://github.com/Michael-Bach/weibel-mlops-demo
cd weibel-mlops-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .streamlit/secrets.toml for the Agent tab
streamlit run app.py
```

---

## Roadmap — next-iteration extensions

These would turn this into a demonstration of audio/acoustic ML competence specifically, rather than a domain-neutral signal pipeline. Not implemented; listed as concrete next steps.

**1. Swap synthetic generator for real spectrogram data**
Replace `ppi_generator.py` with a spectrogram loader (e.g. ESC-50, UrbanSound8K, or custom recordings). The CNN and ConvGRU architectures are input-shape-agnostic — only the data loader and label schema change. This would validate that the pipeline handles real-world label noise and class imbalance, not just a controlled synthetic distribution.

**2. Add model-size and latency budget check to the CI gate**
Extend the current F1 gate to also assert: (a) ONNX model size ≤ N MB, (b) median inference latency ≤ T ms on a reference device or `onnxruntime` CPU benchmark. Currently the latency claim is measured manually; the CI gate only checks F1. Making latency a hard CI constraint closes that gap.

**3. INT8 quantization before ONNX export**
Add a `onnxruntime.quantization.quantize_dynamic` step between the promotion gate and export. Re-run the F1 and latency benchmarks on the quantized graph. This is the practical path to FPGA and microcontroller deployment where FP32 is not viable.

**4. Domain adaptation fine-tuning step**
Add a fine-tuning stage that takes a small set of real labeled examples and adapts the synthetic-pretrained model. Measure how many real examples are needed to recover synthetic-trained F1 — this is the practical question for any synthetic-to-real transfer pipeline.

**5. Acoustic-appropriate tracking backend**
Replace the Kalman constant-velocity tracker with a particle filter or JPDA tracker suited to acoustic scenarios where velocity is not well-modeled as constant (e.g., footsteps, machinery transients). The current tracker is well-matched to constant-velocity kinematic targets; acoustic events often are not.
