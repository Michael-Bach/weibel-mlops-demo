# scripts/calibrate.py
"""
Fits Platt scaling on the ONNX model's validation-set scores and saves a calibrated
threshold to artifacts/calibration.json.

Why calibration matters for radar: the softmax score from an uncalibrated classifier
is not a probability. Confidence 0.8 does not mean Pd = 0.8. Before the operating
point is set against a Pfa requirement, the score needs to be mapped to a true
probability via calibration. Platt scaling fits a logistic regression on the raw
logit scores — cheap, interpretable, and standard practice.

Note on MCE for highly accurate models: when a model achieves near-perfect separation,
most calibrated scores cluster near 0 or 1. sklearn's calibration_curve only reports
non-empty bins, so MCE is computed over the actual score distribution — a high MCE
here indicates the Platt fit is not mapping the extreme logits to true probabilities,
which can happen when the val set is too small for reliable bin estimates.

Usage:
    python scripts/calibrate.py
    # Reads  artifacts/model.onnx + data/processed/X_val.npy
    # Writes artifacts/calibration.json
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


def main(
    model_path: str = "artifacts/model.onnx",
    val_X_path: str = "data/processed/X_val.npy",
    val_y_path: str = "data/processed/y_val.npy",
    out_path: str = "artifacts/calibration.json",
) -> None:
    X = np.load(val_X_path).astype(np.float32)
    y = np.load(val_y_path)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: X})[0]

    # Raw target logit (pre-softmax) as the feature for Platt scaling
    target_logit = logits[:, 1].reshape(-1, 1)

    platt = LogisticRegression(C=1.0)
    platt.fit(target_logit, y)

    calibrated_scores = platt.predict_proba(target_logit)[:, 1]

    # Calibration quality: fraction_of_positives should track mean_predicted_value.
    # Use strategy="quantile" so bins are populated even when scores are clustered
    # at the extremes — gives a more reliable MCE for near-perfect models.
    fraction_pos, mean_pred = calibration_curve(
        y, calibrated_scores, n_bins=10, strategy="quantile"
    )
    mean_calibration_error = float(np.mean(np.abs(fraction_pos - mean_pred)))

    print(f"Platt coef: {platt.coef_[0][0]:.4f}  intercept: {platt.intercept_[0]:.4f}")
    print(f"Mean calibration error (quantile bins): {mean_calibration_error:.4f}")
    print("\nCalibration bins (mean_predicted → fraction_positive):")
    for mp, fp in zip(mean_pred, fraction_pos):
        bar = "█" * int(fp * 20)
        print(f"  {mp:.3f} → {fp:.3f}  {bar}")

    if mean_calibration_error > 0.05:
        print(
            "\nNOTE: MCE > 5%. For a radar Pfa requirement, re-fit calibration on a "
            "larger held-out set or use cross-validated isotonic regression."
        )

    result = {
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(platt.intercept_[0]),
        "mean_calibration_error": mean_calibration_error,
        "n_val_samples": int(len(y)),
        "note": (
            "Apply: calibrated_prob = sigmoid(coef * raw_target_logit + intercept). "
            "Set operating threshold on calibrated_prob to enforce a Pfa target."
        ),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
