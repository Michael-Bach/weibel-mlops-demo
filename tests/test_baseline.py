# tests/test_baseline.py
"""
Tests for the Kalman filter + CA-CFAR baseline detector.
Includes the required Pd > 0.8 check at 10 dB SNR on 200 sequences.
"""

import numpy as np
import pytest

from src.baseline.kalman_cfar import KalmanCFARDetector, _cfar_peaks
from src.data.generator import N_DOPPLER, N_RANGE, T, generate_dataset


@pytest.fixture(scope="module")
def detector() -> KalmanCFARDetector:
    return KalmanCFARDetector()


@pytest.fixture(scope="module")
def dataset_10db():
    """400 sequences (200 target, 200 clutter) at 10 dB SNR."""
    return generate_dataset(400, snr_db=10.0, seed=42)


def test_detect_batch_shape(detector, dataset_10db):
    X, _ = dataset_10db
    preds = detector.detect_batch(X)
    assert preds.shape == (400,)
    assert preds.dtype == bool


def test_score_batch_shape(detector, dataset_10db):
    X, _ = dataset_10db
    scores = detector.score_batch(X)
    assert scores.shape == (400,)
    assert scores.dtype == float


def test_scores_in_unit_interval(detector, dataset_10db):
    X, _ = dataset_10db
    scores = detector.score_batch(X)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_pd_above_0_8_at_10db(detector, dataset_10db):
    """KF tracker must achieve Pd > 0.8 at 10 dB SNR on 200 target sequences."""
    X, y = dataset_10db
    preds = detector.detect_batch(X)
    Pd = float(preds[y == 1].mean())
    assert Pd > 0.8, f"Pd = {Pd:.3f} < 0.8 at 10 dB SNR"


def test_track_positions_length(detector, dataset_10db):
    X, _ = dataset_10db
    pos = detector.track_positions(X[0])
    assert len(pos) == T


def test_track_positions_type(detector, dataset_10db):
    X, _ = dataset_10db
    pos = detector.track_positions(X[0])
    for p in pos:
        assert p is None or (isinstance(p, tuple) and len(p) == 2)


def test_cfar_peaks_return_type():
    rdmap = np.random.randn(N_RANGE, N_DOPPLER).astype(np.float32)
    peaks = _cfar_peaks(rdmap)
    assert isinstance(peaks, list)
    for r, d in peaks:
        assert 0 <= r < N_RANGE
        assert 0 <= d < N_DOPPLER


def test_cfar_suppresses_near_dc():
    """No peaks should be returned at Doppler bin < 5 (MTI gate)."""
    rdmap = np.zeros((N_RANGE, N_DOPPLER), dtype=np.float32)
    # Strong spike near DC
    rdmap[:, 2] = 1000.0
    peaks = _cfar_peaks(rdmap)
    for _, d in peaks:
        assert d >= 5, f"Peak at Doppler bin {d} — MTI gate failed"
