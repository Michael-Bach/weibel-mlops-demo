"""
PPI (Plan Position Indicator) rotating-radar sequence generator.

Coordinate convention
---------------------
  Azimuth 0° = North (top of display), increases clockwise.
  Range bin 0 = radar; bin N_RANGE-1 = maximum instrumented range.

Two public APIs
---------------
generate_ppi_sequence(params, seed)
    Low-level: returns raw amplitude cube + soft Gaussian label + positions.

make_sample(params, has_target, rng)
    High-level: returns (3, n_az, n_range) temporal-feature map + label.
    The three channels are [max, mean, std] of the noise-normalised sweeps —
    this pre-processing suppresses static clutter and makes moving targets
    visible as bright streaks in the max/std channels.
"""

import numpy as np


# ── Physical helpers ──────────────────────────────────────────────────────────

def _beam_gain(az_bins: np.ndarray, target_az_bin: float,
               beam_deg: float, n_az: int) -> np.ndarray:
    """Gaussian one-way azimuth beam pattern."""
    diff  = (az_bins - target_az_bin + n_az / 2) % n_az - n_az / 2
    sigma = (beam_deg / 360 * n_az) / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * (diff / sigma) ** 2)


def _range_gate(r_bins: np.ndarray, target_r: float,
                pulse_bins: float) -> np.ndarray:
    """Gaussian pulse envelope in range."""
    return np.exp(-0.5 * ((r_bins - target_r) / pulse_bins) ** 2)


def _noise_floor(n_range: int) -> np.ndarray:
    """Range-dependent clutter amplitude floor: ∝ R^-2."""
    r = np.maximum(np.arange(n_range, dtype=np.float32), 1.0) / n_range
    return r ** (-2.0)


# ── Raw sequence generator ─────────────────────────────────────────────────────

def generate_ppi_sequence(params: dict, seed: int = 42):
    """
    Generate one PPI sweep sequence with a moving target in clutter.

    Returns
    -------
    ppi       : float32  (n_sweeps, n_az, n_range)
    label     : float32  (n_sweeps, n_az, n_range)  soft Gaussian label
    positions : list of (az_deg, range_bin) per sweep
    """
    rng = np.random.default_rng(seed)
    r, t = params["radar"], params["target"]

    n_sw      = int(r["n_sweeps"])
    n_az      = int(r["n_azimuths"])
    n_range   = int(r["n_ranges"])
    beam_deg  = float(r["beam_width_deg"])
    pulse_bin = float(r["pulse_width_bins"])

    amplitude = 10.0 ** (float(t["snr_db"]) / 20.0)
    r0        = float(t["range_bin"])
    az0       = float(t["azimuth_deg"])
    vr        = float(t["radial_velocity"])
    vaz       = float(t["tangential_velocity"])

    az_bins    = np.arange(n_az,    dtype=np.float32)
    range_bins = np.arange(n_range, dtype=np.float32)
    clutter_fl = _noise_floor(n_range)                    # (n_range,)

    _LABEL_SIGMA = 5.0

    ppi       = np.empty((n_sw, n_az, n_range), dtype=np.float32)
    label     = np.zeros((n_sw, n_az, n_range), dtype=np.float32)
    positions = []

    for sw in range(n_sw):
        r_t  = r0  + sw * vr
        az_t = (az0 + sw * vaz) % 360.0
        positions.append((float(az_t), float(r_t)))

        az_t_bin = az_t * n_az / 360.0

        # Clutter: Rayleigh with range-dependent floor + log-normal speckle
        speckle = np.maximum(1.0 + 0.3 * rng.standard_normal((n_az, n_range)), 0.1)
        clutter  = rng.rayleigh(scale=clutter_fl[np.newaxis, :] * speckle).astype(np.float32)

        # Target signal — scaled by local noise floor so that `amplitude`
        # equals the whitened amplitude ratio after CFAR pre-whitening.
        # Without this, a "10 dB SNR" target at mid-range has whitened
        # amplitude << threshold regardless of SNR setting.
        r_t_int   = int(np.round(r_t).clip(0, n_range - 1))
        local_fl  = clutter_fl[r_t_int]
        beam   = _beam_gain(az_bins, az_t_bin, beam_deg, n_az)
        pulse  = _range_gate(range_bins, r_t, pulse_bin)
        signal = (amplitude * local_fl * np.outer(beam, pulse)).astype(np.float32)

        ppi[sw] = clutter + signal

        # Soft Gaussian label centred on target
        daz = (az_bins - az_t_bin + n_az / 2) % n_az - n_az / 2
        dr  = range_bins - r_t
        label[sw] = np.exp(
            -0.5 * (daz[:, None] ** 2 + dr[None, :] ** 2) / _LABEL_SIGMA ** 2
        )

    return ppi, label, positions


def generate_multitarget_sweep(
    sweep_idx: int,
    targets: list,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one PPI sweep containing any number of moving targets in clutter.

    Parameters
    ----------
    sweep_idx : absolute sweep index (used to derive each target's current position)
    targets   : list of dicts with keys —
                  born   (int)   : sweep index when this target was spawned
                  az0    (float) : azimuth at birth (degrees)
                  r0     (float) : range bin at birth
                  vr     (float) : radial velocity (bins/sweep, negative = inward)
                  vt     (float) : tangential velocity (degrees/sweep)
                  snr_db (float) : signal-to-noise ratio in dB
    params    : radar params dict (must contain "radar" sub-dict)
    rng       : numpy Generator — advanced by this call (clutter drawn from it)

    Returns
    -------
    sweep : float32 (n_az, n_range)
    """
    r         = params["radar"]
    n_az      = int(r["n_azimuths"])
    n_range   = int(r["n_ranges"])
    beam_deg  = float(r["beam_width_deg"])
    pulse_bin = float(r["pulse_width_bins"])

    az_bins    = np.arange(n_az,    dtype=np.float32)
    range_bins = np.arange(n_range, dtype=np.float32)
    clutter_fl = _noise_floor(n_range)

    # Clutter base — same model as generate_ppi_sequence
    speckle = np.maximum(1.0 + 0.3 * rng.standard_normal((n_az, n_range)), 0.1)
    sweep   = rng.rayleigh(scale=clutter_fl[np.newaxis, :] * speckle).astype(np.float32)

    for tgt in targets:
        age  = sweep_idx - tgt["born"]
        r_t  = tgt["r0"]  + age * tgt["vr"]
        az_t = (tgt["az0"] + age * tgt["vt"]) % 360.0

        if r_t < 2 or r_t >= n_range - 2:
            continue

        amplitude  = 10.0 ** (float(tgt["snr_db"]) / 20.0)
        az_t_bin   = az_t * n_az / 360.0
        r_t_int    = int(np.round(r_t).clip(0, n_range - 1))
        local_fl   = clutter_fl[r_t_int]

        beam   = _beam_gain(az_bins, az_t_bin, beam_deg, n_az)
        pulse  = _range_gate(range_bins, r_t, pulse_bin)
        sweep += (amplitude * local_fl * np.outer(beam, pulse)).astype(np.float32)

    return sweep


def generate_clutter_only(params: dict, seed: int = 0) -> np.ndarray:
    """Pure clutter sequence — no target."""
    rng = np.random.default_rng(seed)
    r   = params["radar"]
    n_sw, n_az, n_range = int(r["n_sweeps"]), int(r["n_azimuths"]), int(r["n_ranges"])
    fl  = _noise_floor(n_range)
    out = np.empty((n_sw, n_az, n_range), dtype=np.float32)
    for sw in range(n_sw):
        speckle = np.maximum(1.0 + 0.3 * rng.standard_normal((n_az, n_range)), 0.1)
        out[sw] = rng.rayleigh(scale=fl[np.newaxis, :] * speckle).astype(np.float32)
    return out


def generate_heterogeneous_ppi(
    params: dict,
    seed: int = 42,
    patch_az_lo: int = 60,
    patch_az_hi: int = 100,
    patch_r_lo: int = 20,
    patch_r_hi: int = 50,
    patch_gain: float = 15.0,
    target_snr_db: float = 10.0,
    target_range_bin: float = 38.0,
    target_azimuth_deg: float = 110.0,
    target_vr: float = -1.2,
    target_vaz: float = 1.5,
) -> tuple:
    """
    Generate a PPI sequence with a strong heterogeneous clutter patch and a target
    just outside the patch boundary.

    The patch simulates a land-sea clutter edge or a rain cell: a rectangular region
    of azimuth–range cells with Rayleigh amplitude scaled by `patch_gain` relative to
    the background.  CFAR's cell-averaging reference window straddles this boundary,
    producing a contaminated noise estimate — resulting in false alarms inside the patch
    and missed detections of targets immediately outside it.

    Parameters
    ----------
    patch_az_lo/hi : azimuth bin bounds of the clutter patch (exclusive hi)
    patch_r_lo/hi  : range bin bounds of the clutter patch (exclusive hi)
    patch_gain     : multiplicative amplitude factor inside the patch
    target_*       : target parameters (position just outside the patch)

    Returns
    -------
    ppi       : (n_sweeps, n_az, n_range) float32
    positions : list of (az_deg, range_bin) per sweep
    patch_mask: (n_az, n_range) bool mask of the clutter patch
    """
    rng = np.random.default_rng(seed)
    r   = params["radar"]

    n_sw      = int(r["n_sweeps"])
    n_az      = int(r["n_azimuths"])
    n_range   = int(r["n_ranges"])
    beam_deg  = float(r["beam_width_deg"])
    pulse_bin = float(r["pulse_width_bins"])

    amplitude  = 10.0 ** (target_snr_db / 20.0)
    az_bins    = np.arange(n_az,    dtype=np.float32)
    range_bins = np.arange(n_range, dtype=np.float32)
    clutter_fl = _noise_floor(n_range)

    patch_mask = np.zeros((n_az, n_range), dtype=bool)
    patch_mask[patch_az_lo:patch_az_hi, patch_r_lo:patch_r_hi] = True

    # Scale factor array: patch_gain inside the patch, 1.0 outside
    scale = np.where(patch_mask, patch_gain, 1.0).astype(np.float32)

    ppi       = np.empty((n_sw, n_az, n_range), dtype=np.float32)
    positions = []

    for sw in range(n_sw):
        r_t  = target_range_bin  + sw * target_vr
        az_t = (target_azimuth_deg + sw * target_vaz) % 360.0
        positions.append((float(az_t), float(r_t)))

        az_t_bin = az_t * n_az / 360.0

        # Heterogeneous clutter: Rayleigh with spatially varying scale
        speckle  = np.maximum(1.0 + 0.3 * rng.standard_normal((n_az, n_range)), 0.1)
        clutter  = rng.rayleigh(
            scale=clutter_fl[np.newaxis, :] * scale * speckle
        ).astype(np.float32)

        # Target signal
        r_t_int  = int(np.round(r_t).clip(0, n_range - 1))
        local_fl = clutter_fl[r_t_int]
        beam   = _beam_gain(az_bins, az_t_bin, beam_deg, n_az)
        pulse  = _range_gate(range_bins, r_t, pulse_bin)
        signal = (amplitude * local_fl * np.outer(beam, pulse)).astype(np.float32)

        ppi[sw] = clutter + signal

    return ppi, positions, patch_mask


# ── Temporal feature extractor ────────────────────────────────────────────────

def temporal_features(ppi: np.ndarray) -> np.ndarray:
    """
    Convert raw (n_sw, n_az, n_range) amplitude cube to a 3-channel
    2-D feature map that emphasises moving targets over static clutter.

    Channel 0 — normalised temporal max  : bright where beam ever hit target
    Channel 1 — normalised temporal mean : baseline clutter level
    Channel 2 — normalised temporal std  : high where signal varied (motion)

    All three channels are divided by the per-range noise-floor estimate
    (10th percentile across azimuth × sweeps) so range attenuation is
    mostly removed and the target SNR is approximately range-independent.
    """
    # Estimate noise floor per range bin (robust against target influence)
    noise_fl = np.percentile(ppi, 10, axis=(0, 1), keepdims=True).clip(1e-3)
    norm     = ppi / noise_fl                    # (n_sw, n_az, n_range)

    ch_max  = norm.max(axis=0).astype(np.float32)
    ch_mean = norm.mean(axis=0).astype(np.float32)
    ch_std  = norm.std(axis=0).astype(np.float32)

    return np.stack([ch_max, ch_mean, ch_std])   # (3, n_az, n_range)


# ── High-level sample builder ─────────────────────────────────────────────────

def make_sample(params: dict, has_target: bool,
                rng: np.random.Generator, sample_seed: int):
    """
    Build one (X, y) training sample.

    Parameters
    ----------
    has_target  : bool — whether this sample contains a target
    rng         : pre-seeded Generator for drawing random target parameters
    sample_seed : seed passed to the signal/noise generator

    Returns
    -------
    X : float32  (3, n_az, n_range)  temporal feature map
    y : float32  (n_az, n_range)     soft label (0 everywhere if no target)
    """
    r      = params["radar"]
    d      = params["data"]
    n_range = int(r["n_ranges"])
    n_az    = int(r["n_azimuths"])
    margin  = int(d.get("range_margin", 10))
    snr_lo, snr_hi = d["snr_range"]

    if has_target:
        t_params = {
            "radar": r,
            "target": {
                "snr_db":              float(rng.uniform(snr_lo, snr_hi)),
                "range_bin":           float(rng.integers(margin, n_range - margin)),
                "azimuth_deg":         float(rng.uniform(0, 360)),
                "radial_velocity":     float(rng.uniform(-3.0, 3.0)),
                "tangential_velocity": float(rng.uniform(-4.0, 4.0)),
            },
        }
        ppi, label, _ = generate_ppi_sequence(t_params, seed=sample_seed)
        X = temporal_features(ppi)
        y = label.max(axis=0).astype(np.float32)   # union Gaussian across sweeps
    else:
        ppi = generate_clutter_only(params, seed=sample_seed)
        X   = temporal_features(ppi)
        y   = np.zeros((n_az, n_range), dtype=np.float32)

    return X, y
