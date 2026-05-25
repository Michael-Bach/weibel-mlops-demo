"""
Classical radar baseline detector: MTI filter + spectral peak threshold.

Real radar target detection uses two stages:
  1. MTI (Moving Target Indication) filter — suppresses stationary and
     slow-moving clutter, which concentrates energy in the lowest Doppler
     frequency bins (near DC).
  2. Peak detector — declares a target when the strongest bin in the
     remaining Doppler band stands sufficiently above the local noise floor.

This is the standard classical pipeline that ML-based detectors are
compared against. The threshold is set on a held-out validation set at
a representative SNR, analogous to tuning a CFAR threshold for a target
Pfa in an operational system.
"""

import numpy as np


SIGNAL_LENGTH = 128
# Bins 0–4 are suppressed by the MTI stage (stationary / near-stationary clutter).
_MTI_MIN_BIN = 5
# Peak-to-mean ratio threshold, tuned on 1 000 validation signals at 10 dB SNR.
_THRESHOLD = 7.87
DEFAULT_THRESHOLD = _THRESHOLD  # public alias used by the Streamlit app


def _peak_to_mean(spectrum: np.ndarray) -> float:
    """Peak-to-mean ratio on the Doppler band (MTI filter applied)."""
    doppler_band = spectrum[_MTI_MIN_BIN:]
    mean = float(np.mean(doppler_band))
    return float(np.max(doppler_band)) / mean if mean > 0 else 0.0


class MTIThresholdDetector:
    """
    MTI + spectral peak detector.

    Mimics the classical two-stage radar pipeline:
    - Stage 1: zero out the clutter-dominated low-frequency bins (MTI).
    - Stage 2: declare a target if peak-to-mean ratio in the Doppler band
               exceeds the threshold.
    """

    def __init__(self, threshold: float = _THRESHOLD):
        self.threshold = threshold

    def detect(self, spectrum: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Classify a single magnitude spectrum.
        Returns (target_detected, threshold_curve) for plotting.
        """
        pmr = _peak_to_mean(spectrum)
        detected = pmr > self.threshold

        # Build a flat threshold line in the Doppler band for the plot.
        # Below _MTI_MIN_BIN the threshold is shown as zero (MTI suppression).
        noise_floor = float(np.mean(spectrum[_MTI_MIN_BIN:]))
        threshold_curve = np.zeros_like(spectrum)
        threshold_curve[_MTI_MIN_BIN:] = self.threshold * noise_floor / self.threshold
        # Threshold level = what the peak must exceed: threshold * mean
        threshold_curve[_MTI_MIN_BIN:] = self.threshold * float(
            np.mean(spectrum[_MTI_MIN_BIN:])
        )

        return bool(detected), threshold_curve

    def score(self, spectrum: np.ndarray) -> float:
        """Continuous detection score (peak-to-mean ratio). Used for ROC analysis."""
        return _peak_to_mean(spectrum)

    def score_batch(self, spectra: np.ndarray) -> np.ndarray:
        """Continuous scores for a batch of spectra shaped (n_samples, n_bins)."""
        return np.array([_peak_to_mean(s) for s in spectra])

    def detect_batch(self, spectra: np.ndarray) -> np.ndarray:
        """
        Classify a batch of magnitude spectra shaped (n_samples, n_bins).
        Returns a boolean array of shape (n_samples,).
        """
        return np.array([_peak_to_mean(s) > self.threshold for s in spectra])
