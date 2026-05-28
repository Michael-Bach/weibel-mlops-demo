"""
Classical radar baselines for the PPI pipeline.

Two detectors
-------------
1. PPICFARDetector  — 2-D Cell-Averaging CFAR applied per sweep.
   Uses prefix-sum trick so cost is O(n_az × n_range) regardless of
   guard/reference window size.

2. PPIKalmanTracker — Kalman filter tracker operating in (range, azimuth)
   state space across sweeps.  Associates CFAR peaks to tracks via
   nearest-neighbour gating.  A target is declared present if it is
   confirmed over ≥ min_hits sweeps.

Public interface
----------------
    detector = PPICFARDetector()
    tracker  = PPIKalmanTracker()

    for sweep in sequence:                   # (n_az, n_range) float32
        det_map  = detector.detect(sweep)    # (n_az, n_range) bool
        track_map = tracker.update(det_map)  # (n_az, n_range) float32 in [0,1]

    prob_map = tracker.probability_map()     # accumulated across sweeps
"""

import numpy as np


def _rng_norm(n_range: int) -> np.ndarray:
    """Range-dependent noise floor (R⁻²) used to pre-whiten sweeps before CFAR."""
    r = np.maximum(np.arange(n_range, dtype=np.float32), 1.0) / n_range
    return (r ** (-2.0)).clip(min=0.1)


# ── 2-D CA-CFAR ───────────────────────────────────────────────────────────────

class PPICFARDetector:
    """
    2-D Cell-Averaging CFAR with azimuth wrap-around.

    The sweep is first divided by the R⁻² noise-floor model so that CFAR
    operates on whitened data and the threshold is range-invariant.

    Parameters
    ----------
    guard_az, guard_r  : guard cells each side
    ref_az,   ref_r    : reference cells each side
    threshold_factor   : CFAR scaling constant (higher → fewer false alarms)
    """

    def __init__(
        self,
        guard_az: int = 2,
        guard_r:  int = 2,
        ref_az:   int = 5,
        ref_r:    int = 5,
        threshold_factor: float = 2.5,
    ):
        self.ga  = guard_az
        self.gr  = guard_r
        self.ra  = ref_az
        self.rr  = ref_r
        self.thr = threshold_factor

    def detect(self, sweep: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        sweep : float32  (n_az, n_range)

        Returns
        -------
        det : bool  (n_az, n_range)   True = detection
        """
        # Pre-whiten: divide by R⁻² floor so CFAR threshold is range-invariant
        sweep = sweep / _rng_norm(sweep.shape[1])[np.newaxis, :]
        n_az, n_range = sweep.shape
        ga, gr = self.ga, self.gr
        ra, rr = self.ra, self.rr

        outer_az = ga + ra
        outer_r  = gr + rr

        # 2-D prefix sum (azimuth wraps)
        padded = np.concatenate(
            [sweep[-outer_az:], sweep, sweep[:outer_az]], axis=0
        )  # wrap azimuth edges
        # Pad range with zeros (no wrap)
        padded = np.pad(padded, ((0, 0), (outer_r, outer_r)), mode="edge")

        # Prefix sum
        ps = padded.cumsum(axis=0).cumsum(axis=1)

        def _box_sum(ps_arr, r0, c0, r1, c1):
            """Inclusive 2-D box sum using prefix sums."""
            s = ps_arr[r1, c1]
            if r0 > 0: s -= ps_arr[r0 - 1, c1]
            if c0 > 0: s -= ps_arr[r0:r1+1, c0 - 1].diagonal()  # column strip
            # Use broadcast-safe version
            return (
                ps_arr[r1, c1]
                - (ps_arr[r0 - 1, c1] if r0 > 0 else 0)
                - (ps_arr[r1, c0 - 1] if c0 > 0 else 0)
                + (ps_arr[r0 - 1, c0 - 1] if (r0 > 0 and c0 > 0) else 0)
            )

        # Vectorised: iterate over range only (azimuth vectorised via numpy)
        n_az_pad = padded.shape[0]
        det = np.zeros((n_az, n_range), dtype=bool)

        for ri in range(n_range):
            # Map to padded column index
            ci = ri + outer_r

            # Row extents in padded array (azimuth axis)
            az_start = outer_az   # first real azimuth row in padded
            az_end   = az_start + n_az - 1

            rows_lo  = np.arange(az_start, az_start + n_az)
            rows_hi  = rows_lo

            def box(r0_off, r1_off, c0_off, c1_off):
                R0 = rows_lo + r0_off
                R1 = rows_hi + r1_off
                C0 = ci + c0_off
                C1 = ci + c1_off
                return (
                    ps[R1, C1]
                    - ps[R0 - 1, C1]
                    - ps[R1, C0 - 1]
                    + ps[R0 - 1, C0 - 1]
                )

            outer_sum = box(-outer_az, outer_az, -outer_r, outer_r)
            inner_sum = box(-ga, ga, -gr, gr)
            ref_sum   = outer_sum - inner_sum

            ref_cells = (2 * ra + 2 * ga + 1) * (2 * rr + 2 * gr + 1) \
                      - (2 * ga + 1) * (2 * gr + 1)
            noise_est = ref_sum / ref_cells

            cell_val = padded[rows_lo, ci]
            det[:, ri] = cell_val > self.thr * noise_est

        return det

    def detect_sequence(self, ppi_seq: np.ndarray) -> np.ndarray:
        """Apply per-sweep CFAR; return bool (n_sweeps, n_az, n_range)."""
        return np.stack([self.detect(ppi_seq[t]) for t in range(len(ppi_seq))])

    def prob_map(self, ppi_seq: np.ndarray) -> np.ndarray:
        """Fraction of sweeps declaring detection per cell → [0, 1]."""
        return self.detect_sequence(ppi_seq).mean(axis=0).astype(np.float32)


# ── Kalman Filter Tracker ─────────────────────────────────────────────────────

class _KalmanTrack2D:
    """
    Constant-velocity Kalman track in (range, azimuth) space.

    State  x = [r, vr, az, vaz]   (range bin, range-rate, azimuth bin, az-rate)
    Obs    z = [r, az]
    """

    F = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=float)

    Q = np.diag([1.0, 0.5, 1.0, 0.5])   # process noise
    R = np.diag([4.0, 4.0])              # observation noise

    def __init__(self, z: np.ndarray):
        self.x = np.array([z[0], 0.0, z[1], 0.0])
        self.P = np.eye(4) * 10.0
        self.hits = 1
        self.age  = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def mahal_sq(self, z: np.ndarray) -> float:
        inn = z - self.H @ self.x
        S   = self.H @ self.P @ self.H.T + self.R
        return float(inn @ np.linalg.inv(S) @ inn)

    def update(self, z: np.ndarray):
        inn = z - self.H @ self.x
        S   = self.H @ self.P @ self.H.T + self.R
        K   = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ inn
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1


class PPIKalmanTracker:
    """
    Multi-target KF tracker driven by CFAR peaks.

    Parameters
    ----------
    cfar            : PPICFARDetector (or None → uses default)
    gate_mahal      : Mahalanobis² gate threshold
    min_hits        : sweeps a track must survive to count as a detection
    max_peaks       : maximum CFAR peaks to consider per sweep
    """

    def __init__(
        self,
        cfar: PPICFARDetector = None,
        gate_mahal: float = 16.0,
        min_hits:   int   = 4,
        max_peaks:  int   = 10,
    ):
        self.cfar       = cfar or PPICFARDetector()
        self.gate       = gate_mahal
        self.min_hits   = min_hits
        self.max_peaks  = max_peaks
        self._tracks: list[_KalmanTrack2D] = []
        self._confirmed_cells: list = []   # (az_idx, r_idx) of confirmed tracks

    def _cfar_peaks(self, sweep: np.ndarray) -> np.ndarray:
        """Return up to max_peaks detection centroids as (r, az) pairs."""
        det = self.cfar.detect(sweep)
        ys, xs = np.where(det)          # ys=az, xs=range
        # Ignore the first 5 range bins — dominated by near-range clutter
        valid = xs >= 5
        ys, xs = ys[valid], xs[valid]
        if len(xs) == 0:
            return np.empty((0, 2))
        # Sort by range-whitened amplitude so far-range targets aren't crowded out
        # by brighter near-range clutter in raw amplitude
        noise_floor = _rng_norm(sweep.shape[1])
        vals = sweep[ys, xs] / noise_floor[xs]
        order = np.argsort(vals)[::-1][: self.max_peaks]
        return np.column_stack([xs[order], ys[order]]).astype(float)  # (r, az)

    def _process_peaks(self, peaks: np.ndarray, n_az: int, n_range: int) -> None:
        """Predict, associate, spawn, prune, and record — shared by both update paths."""
        # Predict all existing tracks
        for tr in self._tracks:
            tr.predict()

        # Associate peaks → tracks (nearest in Mahalanobis distance)
        used_peaks: set = set()

        for tr in self._tracks:
            best_d, best_pi = np.inf, -1
            for pi, z in enumerate(peaks):
                if pi in used_peaks:
                    continue
                d = tr.mahal_sq(z)
                if d < best_d:
                    best_d, best_pi = d, pi
            if best_d < self.gate and best_pi >= 0:
                tr.update(peaks[best_pi])
                used_peaks.add(best_pi)

        # Spawn new tracks for unassociated peaks
        for pi, z in enumerate(peaks):
            if pi not in used_peaks:
                self._tracks.append(_KalmanTrack2D(z))

        # Prune tracks that have missed more than 2 consecutive sweeps
        self._tracks = [tr for tr in self._tracks if tr.age - tr.hits < 3]

        # Record confirmed track positions
        for tr in self._tracks:
            if tr.hits >= self.min_hits:
                r_est  = int(np.round(tr.x[0]).clip(0, n_range - 1))
                az_est = int(np.round(tr.x[2]).clip(0, n_az - 1))
                self._confirmed_cells.append((az_est, r_est))

    def update(self, sweep: np.ndarray) -> None:
        """Process one sweep via CFAR peak extraction then association."""
        peaks = self._cfar_peaks(sweep)
        self._process_peaks(peaks, sweep.shape[0], sweep.shape[1])

    def update_from_peaks(self, peaks: np.ndarray, n_az: int, n_range: int) -> None:
        """
        Feed pre-extracted peaks directly, bypassing internal CFAR.

        Parameters
        ----------
        peaks : float (N, 2)  columns = [range_bin, az_bin]  (same as _cfar_peaks output)
        n_az, n_range : grid dimensions for clipping confirmed positions
        """
        self._process_peaks(peaks, n_az, n_range)

    def prob_map(self, n_az: int, n_range: int,
                 spread: int = 3) -> np.ndarray:
        """
        Rasterise confirmed track positions to a probability map.
        Each confirmed position contributes a Gaussian blob of radius ``spread``.
        """
        result = np.zeros((n_az, n_range), dtype=np.float32)
        az_idx = np.arange(n_az)
        r_idx  = np.arange(n_range)

        for (az, r) in self._confirmed_cells:
            daz = (az_idx - az + n_az // 2) % n_az - n_az // 2
            dr  = r_idx - r
            blob = np.exp(
                -0.5 * (daz[:, None] ** 2 + dr[None, :] ** 2) / spread ** 2
            )
            result = np.maximum(result, blob.astype(np.float32))

        return result

    def track_positions(self) -> list:
        """Return list of (az_deg, range_bin) for each confirmed track."""
        out = []
        for tr in self._tracks:
            if tr.hits >= self.min_hits:
                out.append((float(tr.x[2]), float(tr.x[0])))
        return out

    def process_sequence(self, ppi_seq: np.ndarray) -> np.ndarray:
        """
        Convenience: run full sequence and return probability map.

        Parameters
        ----------
        ppi_seq : float32  (n_sweeps, n_az, n_range)

        Returns
        -------
        prob : float32  (n_az, n_range)
        """
        self._tracks = []
        self._confirmed_cells = []
        for t in range(len(ppi_seq)):
            self.update(ppi_seq[t])
        n_az, n_range = ppi_seq.shape[1], ppi_seq.shape[2]
        return self.prob_map(n_az, n_range)
