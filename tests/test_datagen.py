# tests/test_datagen.py
"""
Tests for the Range-Doppler sequence generator.
Verifies output shapes, dtypes, label values, and reproducibility.
"""

import numpy as np

from src.data.generator import N_DOPPLER, N_RANGE, T, generate_dataset


def test_dataset_shape():
    X, y = generate_dataset(20, snr_db=10.0)
    assert X.shape == (20, T, N_RANGE, N_DOPPLER)
    assert y.shape == (20,)


def test_dataset_dtype():
    X, y = generate_dataset(20, snr_db=10.0)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_dataset_shape_explicit():
    """Output is exactly (n, 8, 64, 128)."""
    X, _ = generate_dataset(12, snr_db=5.0)
    assert X.shape == (12, 8, 64, 128)


def test_labels_are_binary():
    _, y = generate_dataset(40, snr_db=10.0)
    assert set(y.tolist()).issubset({0, 1})


def test_balanced_classes():
    _, y = generate_dataset(100, snr_db=10.0)
    assert (y == 1).sum() == 50
    assert (y == 0).sum() == 50


def test_reproducible_with_same_seed():
    X1, y1 = generate_dataset(20, snr_db=10.0, seed=0)
    X2, y2 = generate_dataset(20, snr_db=10.0, seed=0)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    X1, _ = generate_dataset(20, snr_db=10.0, seed=0)
    X2, _ = generate_dataset(20, snr_db=10.0, seed=1)
    assert not np.allclose(X1, X2)


def test_finite_values():
    X, _ = generate_dataset(20, snr_db=10.0)
    assert np.isfinite(X).all()


def test_higher_snr_stronger_signal():
    """Same noise realisation, higher SNR → higher target blob mean (blob is amplitude-scaled).

    Clutter in target sequences is NOT amplitude-scaled, so max() can be identical;
    mean() captures the blob contribution reliably.
    """
    from src.data.generator import _target_sequence
    t_low  = _target_sequence(np.random.default_rng(42), snr_db=-5.0)
    t_high = _target_sequence(np.random.default_rng(42), snr_db=20.0)
    assert float(t_high.mean()) > float(t_low.mean())
