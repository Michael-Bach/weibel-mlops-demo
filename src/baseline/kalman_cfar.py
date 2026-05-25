"""
Classical baseline: CA-CFAR detections fed into a Kalman filter tracker.

Pipeline per sequence of T scans:
  1. 1D vectorized CA-CFAR (cumsum-based, O(N_RANGE × N_DOPPLER)) per scan.
  2. Nearest-neighbour association within a Mahalanobis gate.
  3. KF state update; count scans with an associated detection.
  4. Declare target present if ≥ 4 of 8 scans have a track hit.

Compatible with the detect_batch() / score_batch() interface used by the ROC tab.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from src.data.generator import N_DOPPLER, N_RANGE, T

_MIN_DOPPLER_BIN = 5    # suppress near-DC clutter (MTI equivalent)
_MIN_TRACK_HITS = 4     # minimum scan hits to declare target present
DEFAULT_CFAR_SCALE = 8.0


# ── Vectorized CA-CFAR ────────────────────────────────────────────────────────

def _cfar_peaks(
    rdmap: np.ndarray,
    guard: int = 3,
    ref: int = 8,
    scale: float = DEFAULT_CFAR_SCALE,
    n_peaks: int = 5,
) -> list[tuple[int, int]]:
    """Vectorized 1D CA-CFAR per range row → local maxima above threshold.

    Uses prefix-sum windows for O(N_RANGE × N_DOPPLER) complexity.
    Returns up to n_peaks (range_bin, doppler_bin) positions sorted by power.
    """
    power = rdmap ** 2                              # (N_RANGE, N_DOPPLER)
    n_r, n_d = power.shape

    start = guard + ref
    end = n_d - guard - ref
    if start >= end:
        return []

    # Prefix sums along Doppler axis for O(1) window queries
    cs = np.concatenate([np.zeros((n_r, 1)), np.cumsum(power, axis=1)], axis=1)

    d = np.arange(start, end)                       # valid Doppler indices
    left  = cs[:, d - guard]      - cs[:, d - guard - ref]      # (n_r, len(d))
    right = cs[:, d + guard + ref + 1] - cs[:, d + guard + 1]   # (n_r, len(d))
    noise = (left + right) / (2 * ref)

    det = np.zeros((n_r, n_d), dtype=bool)
    det[:, start:end] = power[:, start:end] > scale * noise

    det[:, :_MIN_DOPPLER_BIN] = False              # MTI: suppress near-DC

    local_max = (power == maximum_filter(power, size=5)) & det
    positions = np.argwhere(local_max)
    if len(positions) == 0:
        return []

    powers = power[positions[:, 0], positions[:, 1]]
    order = np.argsort(powers)[::-1]
    positions = positions[order[:n_peaks]]
    return [(int(r), int(d)) for r, d in positions]


# ── Kalman filter track ───────────────────────────────────────────────────────

class _KalmanTrack:
    """Constant-velocity Kalman filter in range-Doppler space.

    State  x = [range, range_rate, doppler]
    Measurement z = [range, doppler]
    """

    # Transition: range += range_rate each scan; Doppler static
    F = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
    H = np.array([[1., 0., 0.], [0., 0., 1.]])
    Q = np.diag([1.0, 0.25, 0.5])   # process noise
    R = np.diag([4.0, 4.0])          # measurement noise

    def __init__(self, r: float, d: float) -> None:
        self.x = np.array([r, 0.0, d])
        self.P = np.diag([9.0, 1.0, 9.0])

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def mahalanobis_sq(self, z: np.ndarray) -> float:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P


# ── Tracker ───────────────────────────────────────────────────────────────────

def _track_sequence(
    sequence: np.ndarray,
    gate_mahal_sq: float = 25.0,
) -> tuple[int, list]:
    """Run KF tracker over T scans of a Range-Doppler sequence.

    Returns (n_hits, per_scan_positions) where positions is [(r,d) or None].
    """
    hits = 0
    positions: list = []
    track: _KalmanTrack | None = None

    for t in range(sequence.shape[0]):
        peaks = _cfar_peaks(sequence[t])

        if track is None:
            if peaks:
                r0, d0 = peaks[0]
                track = _KalmanTrack(float(r0), float(d0))
                hits += 1
                positions.append((r0, d0))
            else:
                positions.append(None)
        else:
            track.predict()
            best_z, best_d2 = None, float("inf")
            for r, d in peaks:
                z = np.array([float(r), float(d)])
                d2 = track.mahalanobis_sq(z)
                if d2 < gate_mahal_sq and d2 < best_d2:
                    best_z, best_d2 = z, d2

            if best_z is not None:
                track.update(best_z)
                hits += 1
                est_r = int(round(float(track.x[0])))
                est_d = int(round(float(track.x[2])))
                positions.append((est_r, est_d))
            else:
                positions.append(None)

    return hits, positions


# ── Public detector class ─────────────────────────────────────────────────────

class KalmanCFARDetector:
    """Kalman filter tracker with CA-CFAR front-end.

    Declares a sequence as target-present if a track is initiated and maintained
    across ≥ min_hits of T scans.

    Implements detect_batch() / score_batch() for compatibility with ROC tab.
    """

    def __init__(self, min_hits: int = _MIN_TRACK_HITS) -> None:
        self.min_hits = min_hits

    def detect_sequence(self, sequence: np.ndarray) -> bool:
        hits, _ = _track_sequence(sequence)
        return hits >= self.min_hits

    def score_sequence(self, sequence: np.ndarray) -> float:
        """Continuous score: track hit fraction ∈ [0, 1]."""
        hits, _ = _track_sequence(sequence)
        return hits / max(sequence.shape[0], 1)

    def track_positions(self, sequence: np.ndarray) -> list:
        """Per-scan (range_bin, doppler_bin) or None — for visualization."""
        _, positions = _track_sequence(sequence)
        return positions

    def detect_batch(self, sequences: np.ndarray) -> np.ndarray:
        """sequences: (n, T, N_RANGE, N_DOPPLER) → bool array (n,)"""
        return np.array([self.detect_sequence(s) for s in sequences])

    def score_batch(self, sequences: np.ndarray) -> np.ndarray:
        """sequences: (n, T, N_RANGE, N_DOPPLER) → float array (n,)"""
        return np.array([self.score_sequence(s) for s in sequences])
