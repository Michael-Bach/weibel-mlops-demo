# tests/test_datagen.py
"""
Tests for the synthetic radar data generator.
Checks data contracts (shape, dtype, label values) without generating large datasets.
"""

import numpy as np

from src.data.generate import generate_clutter, generate_dataset, generate_target

SIGNAL_LENGTH = 64
SNR_DB = 10.0

MINIMAL_PARAMS = {
    "n_samples": 20,
    "signal_length": SIGNAL_LENGTH,
    "snr_range": [5.0, 15.0],
    "clutter_ratio": 0.5,
    "noise_seed": 0,
    "val_split": 0.2,
}


def test_target_shape_and_dtype():
    sig = generate_target(SIGNAL_LENGTH, SNR_DB, np.random.default_rng(0))
    assert sig.shape == (SIGNAL_LENGTH,)
    assert sig.dtype == np.float32


def test_clutter_shape_and_dtype():
    sig = generate_clutter(SIGNAL_LENGTH, SNR_DB, np.random.default_rng(0))
    assert sig.shape == (SIGNAL_LENGTH,)
    assert sig.dtype == np.float32


def test_dataset_shapes():
    X, y = generate_dataset(MINIMAL_PARAMS)
    assert X.shape == (20, SIGNAL_LENGTH)
    assert y.shape == (20,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_dataset_labels_are_binary():
    X, y = generate_dataset(MINIMAL_PARAMS)
    assert set(y.tolist()).issubset({0, 1})


def test_dataset_clutter_ratio():
    params = {**MINIMAL_PARAMS, "n_samples": 100}
    _, y = generate_dataset(params)
    clutter_frac = (y == 0).sum() / len(y)
    assert 0.4 < clutter_frac < 0.6


def test_dataset_reproducible_with_same_seed():
    X1, y1 = generate_dataset(MINIMAL_PARAMS)
    X2, y2 = generate_dataset(MINIMAL_PARAMS)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
