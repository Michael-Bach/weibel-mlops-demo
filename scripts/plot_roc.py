# scripts/plot_roc.py
"""
Plots the ROC curve (Pd vs Pfa) for the exported ONNX model on the validation set.
Saves artifacts/roc_curve.png.

For a radar classifier, accuracy is the wrong metric — the operator chooses an
operating point on the ROC curve based on mission requirements (high Pd at low Pfa).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, roc_curve


def main(
    model_path: str = "artifacts/model.onnx",
    val_X_path: str = "data/processed/X_val.npy",
    val_y_path: str = "data/processed/y_val.npy",
    out_path: str = "artifacts/roc_curve.png",
) -> None:
    X = np.load(val_X_path).astype(np.float32)
    y = np.load(val_y_path)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: X})[0]

    # Softmax score for the target class (label 1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    scores = probs[:, 1]  # P(target)

    fpr, tpr, thresholds = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, color="#1f77b4", label=f"ONNX model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("Pfa (False Alarm Rate)")
    ax.set_ylabel("Pd (Detection Rate)")
    ax.set_title("ROC Curve — Target vs. Clutter")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"AUC: {auc:.4f}")
    if auc > 0.999:
        print(
            "WARNING: AUC = 1.00 — the problem is trivially separable on this dataset. "
            "Validate on out-of-distribution or real data before drawing conclusions."
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
