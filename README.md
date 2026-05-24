# Radar Signal Classifier — MLOps Demo

Binary classification of radar returns (target vs. clutter) built as an end-to-end MLOps
pipeline. The signal model is intentionally simple — the pipeline architecture is the point.

---

## 1. Architecture Overview

```mermaid
flowchart TD
    G[1. generate.py\nSynthetic radar data] --> T[2. ruff + pytest\nLint and test]
    T --> D[3. docker build\nDockerfile.train]
    D --> K[4. kubectl apply\nK8s Job runs training\nmultiple runs / param sweeps]
    K --> WB[5. MLflow dashboard\nInspect runs\nSelect best model]
    WB --> X[6. export_onnx.py\nExport selected model\nto ONNX format]
```

The pipeline has a deliberate human decision point at step 5. Multiple training runs land in
MLflow — different hyperparameters, different seeds — and the best model is selected before
export. ONNX export is not automatic; it is a promotion decision.

The final artifact is `model.onnx`. In production it targets the signal preprocessing hardware
inside the radar system.

---

## 2. Integration Contract

The deployed artifact is `artifacts/model.onnx` — a self-contained inference graph with no PyTorch dependency.

| | Name | Shape | dtype |
|---|---|---|---|
| **Input** | `signal` | `[batch, 128]` | `float32` |
| **Output** | `logits` | `[batch, 2]` | `float32` |

Input is 128 raw time-domain ADC samples from a single PRI (pulse repetition interval). The FFT is baked into the graph — the signal processor feeds samples directly, no upstream frequency transform required.

Output is a logit pair `[clutter_score, target_score]`. Predicted class: `argmax(logits)` (0 = clutter, 1 = target). Confidence: `softmax(logits)[predicted_class]`.

**Latency:** Target hardware TBD. Host CPU baseline: **p50 = 0.035 ms, p99 = 0.055 ms** (batch_size=1, 50 warmup + 1000 timed runs, `python src/inference/validate_onnx.py`).

<details>
<summary>ONNX graph inspection — <code>python scripts/inspect_onnx.py</code></summary>

```
=== Inputs ===
  signal: FLOAT['batch_size', 128]

=== Outputs ===
  logits: FLOAT['batch_size', 2]

=== Graph ===
  Nodes : 14
  Learned params : 16,838
  Total (incl. buffers): 16,966

=== Op types ===
  Mul      : 1   ← Hanning window
  DFT      : 1   ← torch.fft.rfft (baked FFT)
  Gather   : 2   ← extract real and imaginary parts
  Pow      : 2   ← re², im²
  Add      : 1   ← re² + im²
  Sqrt     : 1   ← magnitude = sqrt(re² + im²)
  Unsqueeze: 1   ← shape bookkeeping for DFT input
  Gemm     : 3   ← linear layers (BatchNorm folded by ONNX optimizer)
  Relu     : 2
```

The graph contains a real `DFT` node — the FFT is a first-class operation in the exported
artifact, not a matrix multiply approximation. Magnitude is computed via explicit real/imag
decomposition (`Gather → Pow → Add → Sqrt`) rather than `ReduceL2`, which avoids an
opset-18 attribute incompatibility in some onnxruntime builds. The 128-float Hanning window
buffer accounts for the difference between learned params (16,838) and total initializers (16,966).

</details>

---

## 3. Signal Model

The classifier distinguishes two signal types that model real radar returns:

| Class | Signal | Physical analogue |
|---|---|---|
| **Target** (label 1) | Sinusoid + AWGN | Coherent Doppler return from a moving object |
| **Clutter** (label 0) | Cumulative-sum noise (low-frequency) | Ground/sea clutter — correlated, slow-varying |

SNR is drawn uniformly from 5–20 dB per sample.

The model's first operation is a Hanning window followed by a magnitude spectrum projection:
a 128-point time-domain signal is mapped to 65 frequency bins before the MLP layers. The
projection is implemented as two matrix multiplications against pre-computed cosine and sine
DFT basis buffers — mathematically identical to `rfft` but expressed entirely in standard
ONNX primitives. Targets produce a sharp spectral peak; clutter produces a diffuse
low-frequency spectrum. These are more linearly separable in frequency domain, which is why
the model converges in the first few epochs.

The FFT is baked into the ONNX graph. The deployed model accepts raw time-domain signals
directly from the radar receiver — no separate preprocessing step, no preprocessing contract
between the model and the hardware it runs on.

**Hardware integration note:** The 128-sample input vector maps directly to one PRI sample
buffer as it exits the radar receiver's ADC. In a typical pulsed radar the signal processor
collects range-gate samples over a single pulse repetition interval; those 128 samples are
passed as-is to the ONNX graph. No resampling, windowing, or FFT is required upstream —
those operations live inside the graph. The signal processor treats the model as a black-box
function: `float32[128] → int(argmax)`. This is the integration boundary that FPGA toolchains
such as Xilinx Vitis AI compile against.

---

## 4. Run the Pipeline Locally

```bash
git clone https://github.com/Michael-Bach/weibel-mlops-demo
cd weibel-mlops-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

**Step 1 — Lint and tests (no data needed, ~10 seconds)**
```bash
python -m ruff check .
pytest tests/ -v
```

**Step 2 — Generate synthetic radar data**
```bash
python src/data/generate.py
# Writes data/raw/X.npy, data/raw/y.npy
# Writes data/processed/X_train.npy, X_val.npy, y_train.npy, y_val.npy
```

**Step 3 — Train the model**
```bash
python src/training/train.py
# Logs loss + val_accuracy to MLflow every epoch
# Writes artifacts/model_best.pt and artifacts/metrics.json
# Exits with code 1 if val_accuracy < 0.80
```

**Step 4 — Export to ONNX**
```bash
python scripts/export_onnx.py
# Writes artifacts/model.onnx with dynamic batch axis
```

**Step 5 — Run ONNX inference**
```bash
python src/inference/validate_onnx.py
# Loads model.onnx via onnxruntime — no PyTorch needed
# Prints label + confidence for a sample batch
```

---

## 5. CI/CD

The workflow at `.github/workflows/ml_pipeline.yml` runs the full pipeline on every push:

```
ruff lint → pytest → generate data → train → evaluate → export ONNX → upload artifacts
```

CI runs training as a fast validation pass — confirms the pipeline is not broken. Production
training runs happen on Kubernetes (section 6), where multiple param sweeps run in parallel
and land in MLflow for comparison.

The quality gate is in `train.py`:
```python
if best_acc < baseline_accuracy:
    sys.exit(1)  # fails the CI job
```

**Run the pipeline locally with `act`**

`act` executes the same workflow file in Docker without pushing to GitHub:

```bash
# Install (macOS)
brew install act

# Install (Linux)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the full pipeline locally
act push
```

`.actrc` is already configured to use a lightweight (~1 GB) runner image instead of the
default 12 GB one. The first run pulls the image; subsequent runs are fast.

No secrets required — MLflow writes to `./mlruns` locally.

---

## 6. Kubernetes — Training at Scale

The training pipeline runs as a one-shot Kubernetes Job:

```bash
docker build -f Dockerfile.train -t YOUR_REGISTRY/weibel-radar-train:latest .
docker push YOUR_REGISTRY/weibel-radar-train:latest
kubectl apply -f k8s/training-job.yaml
```

`Dockerfile.train` contains the full training stack — PyTorch, MLflow, DVC (~4 GB). The Job
runs `generate → train → export` and writes `model.onnx` to a PersistentVolume. Multiple
Jobs can be kicked off with different `params.yaml` values to produce a sweep of runs in
MLflow. The Job reads `MLFLOW_TRACKING_URI` from the environment to find the on-prem server.

---

## 7. Experiment Tracking — MLflow

Each training run logs all hyperparameters from `params.yaml`, `train_loss` and
`val_accuracy` per epoch, and `best_val_accuracy` as a summary metric. The best checkpoint
is stored as a run artifact. Runs are written to `./mlruns` by default — no network, no
account, no external service.

**View the experiment dashboard:**
```bash
mlflow ui
# Open http://localhost:5000
```

Multiple runs with different hyperparameters appear side by side. The operator selects the
best run before triggering ONNX export — this is the human gate before any model touches
hardware.

To ablate the FFT step: set `use_fft: false` in `params.yaml`, retrain, and compare runs
side by side in the MLflow UI.

**Ablation results** (3 runs, 30 epochs, SNR 5–20 dB training range):

| Config | Val accuracy | Train loss (ep 30) | Params |
|---|---|---|---|
| FFT on, hidden [64] | 100% | 0.0003 | 4,482 |
| FFT off, hidden [64] | 100% | 0.0058 | 8,514 |
| FFT on, hidden [128] | 100% | 0.0002 | 8,962 |

All three configurations reach 100% validation accuracy on this synthetic dataset — the
classification problem is not the bottleneck. The FFT projection is the useful observation:
`FFT on / hidden [64]` reaches near-zero loss with half the parameters of the no-FFT
equivalent, because the frequency-domain representation makes target vs. clutter linearly
separable within a few epochs. The no-FFT model gets there too, but needs 19× higher final
training loss at the same epoch count, indicating slower convergence and a harder
optimisation landscape. See `artifacts/snr_benchmark.png` for accuracy across the −10–25 dB range.

> **Note:** these ablations used single hidden layers ([64] or [128]) and were run before
> the Hanning window was added. The deployed model uses `hidden_dims: [128, 64]` (two
> layers) with Hanning + DFT magnitude preprocessing. Re-running the ablation with the
> current code produces the same accuracy — the synthetic problem is not sensitive to these
> differences — but the convergence numbers will differ slightly.

**On a shared cluster**, point all training Jobs at a central MLflow server:
```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```
The server runs entirely on-prem — no data leaves the network.

---

## 8. Key Design Decisions

**`params.yaml` as single source of truth**
All hyperparameters live in one file. Every component reads from it. Changing an experiment
means one edit, not hunting through script arguments.

**FFT baked into the model graph**
`torch.fft.rfft` runs inside `RadarClassifier.forward()`. The ONNX export therefore accepts
raw time-domain signals — no preprocessing contract between the model and the hardware it
runs on. The signal processor feeds samples directly into the graph.

**ONNX as the deployment artifact**
ONNX decouples training from runtime. The exported `.onnx` file can be compiled for the
target hardware without retraining — ONNX Runtime for embedded Linux, TensorRT for NVIDIA
Jetson, OpenVINO for Intel silicon, or FPGA toolchains such as Xilinx Vitis AI that accept
ONNX as input. No retraining, no re-export.

**`sys.exit(1)` as the CI gate**
The quality gate is intrinsic to the training step. The pipeline can't silently promote
a bad model — it has to actively pass.

**ONNX with dynamic batch axis**
The same exported model handles single-sample real-time returns and large batched evaluation
jobs. No re-export needed when the batch size changes.

**DVC local cache only**
Data is versioned but no remote is configured. In production: one line —
`dvc remote add -d s3 s3://your-bucket/dvc-cache`.

---

## 9. What I'd Do Differently at Production Scale

**GPU training**
Add `resources.limits: nvidia.com/gpu: 1` to `training-job.yaml` and use a CUDA base image.
The training code selects the device automatically — no code changes needed.

**Retraining trigger**
Retraining should be triggered by data drift (PSI or KL divergence on incoming signal
statistics), not code commits. The model should retrain when the operating environment changes.

**Model registry**
MLflow run artifacts provide basic lineage. A proper registry adds promotion stages
(candidate → validated → released), rollback triggers, and audit trails.

**Hardware-in-the-loop testing**
Before flashing to the radar signal processor, the ONNX model should be validated against
recorded real returns — not just synthetic data. Replay captured returns through the ONNX
graph and compare the output distribution against known labels.

**Signal model**
A production classifier on range-Doppler maps would use a 2-D CNN operating on the full
velocity-range plane. The pipeline infrastructure is identical — only `classifier.py` and
`generate.py` change.

**CFAR comparison**
The baseline detector in most operational radars is CFAR (Constant False Alarm Rate) —
a threshold that adapts to the local clutter power to hold Pfa fixed regardless of
background level. CFAR is interpretable, requires no training data, and is well understood
by the signal processing community. The classifier's advantage is not in raw Pd at high SNR
— both approaches saturate there — but in two specific regimes:

1. **Low SNR / heterogeneous clutter.** CFAR assumes a locally stationary clutter model
   (usually Rayleigh or Gaussian). When the clutter structure deviates — sea spikes,
   multipath, rain — the threshold estimate degrades and Pfa spikes. A trained classifier
   can learn the clutter's actual spectral structure from data and generalise across
   non-stationary backgrounds without explicit modelling.

2. **Pfa operating point flexibility.** CFAR enforces a fixed Pfa by construction but
   can't jointly optimise Pd and Pfa as a pair. A classifier outputs a score that can be
   thresholded post-hoc — sweep the threshold on `softmax(logits)[1]` to trade Pd against
   Pfa without retraining, and the result is a full ROC curve rather than a single accuracy
   number. Raw softmax outputs are not calibrated probabilities — `scripts/calibrate.py`
   fits Platt scaling on the validation set and writes coefficients to
   `artifacts/calibration.json`. The ROC curve is in `artifacts/roc_curve.png`
   (AUC = 1.00 on this synthetic dataset; the tooling is the deliverable, not the number).

The practical test before replacing CFAR is hardware-in-the-loop validation on recorded
real returns, not synthetic benchmarks. Synthetic data sets the floor; real clutter
statistics set the ceiling.

---

## 10. On-Prem CI — Gitea

For classified data, CI must run entirely within the network. Gitea is a self-hosted Git
server with built-in Actions that uses the same YAML syntax as GitHub Actions.

**Start Gitea and the runner:**
```bash
docker compose -f docker-compose.gitea.yml up -d gitea
# Wait ~10 seconds for Gitea to initialise, then open http://localhost:3000
# Create an admin account, then enable Actions:
#   Admin panel → Settings → Actions → Enable
```

**Get a runner registration token:**
```
http://localhost:3000/<your-username>/weibel-mlops-demo → Settings → Actions → Runners
```

**Start the runner with that token:**
```bash
GITEA_RUNNER_TOKEN=<token> docker compose -f docker-compose.gitea.yml up -d runner
```

**Push the repo to Gitea:**
```bash
# Create the repo in the Gitea UI first, then:
git remote add gitea http://localhost:3000/<your-username>/weibel-mlops-demo.git
git push gitea master
```

The workflow at `.gitea/workflows/ml_pipeline.yml` triggers automatically. It is identical
to the GitHub Actions workflow — same steps, same quality gate, same ONNX artifact.

**Air-gapped note:** by default the runner fetches `actions/checkout` and `actions/setup-python`
from GitHub. To go fully offline, mirror those action repos into your Gitea instance and
update the `uses:` paths to point at your local mirror.

---

## Stack

| Tool | Role |
|---|---|
| PyTorch | Model training, FFT preprocessing layer |
| MLflow | Experiment tracking and artifact lineage — fully self-hosted |
| DVC | Data and pipeline versioning (`dvc.yaml` defines generate → train → export stages) |
| ONNX + onnxruntime | Framework-agnostic export — targets edge hardware |
| Docker | Training image |
| Kubernetes | Job (training) |
| GitHub Actions / Gitea | CI/CD — lint, test, train, gate, export (Gitea for on-prem) |
| ruff | Linting |
| pytest | Unit tests — data contracts, model interface, ONNX inference, PyTorch/ONNX equivalence |
| scikit-learn | Platt scaling calibration (`scripts/calibrate.py`), ROC curve (`scripts/plot_roc.py`) |
