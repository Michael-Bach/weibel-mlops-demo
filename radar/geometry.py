"""
radar/geometry.py — Polar → Cartesian LUT and coordinate helpers.
"""

import numpy as np
import streamlit as st

from radar.sessions import N_AZ, N_RANGE, N_GRID


@st.cache_data
def build_lut(n_az: int, n_range: int, n_grid: int):
    cx = cy = n_grid // 2
    r_max = n_grid // 2 - 5
    gy, gx = np.mgrid[0:n_grid, 0:n_grid]
    dx = gx.astype(float) - cx
    dy = float(cy) - gy
    dist = np.sqrt(dx**2 + dy**2)
    r_frac = dist / r_max
    r_idx  = (r_frac * (n_range - 1)).round().clip(0, n_range - 1).astype(int)
    az_rad = np.arctan2(dx, dy)
    az_rad = np.where(az_rad < 0, az_rad + 2 * np.pi, az_rad)
    az_idx = (az_rad / (2 * np.pi) * n_az).astype(int) % n_az
    mask   = r_frac <= 1.0
    return az_idx, r_idx, mask


def get_lut():
    return build_lut(N_AZ, N_RANGE, N_GRID)


def p2c(polar: np.ndarray) -> np.ndarray:
    az_lut, r_lut, mask = get_lut()
    out = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)
    out[mask] = polar[az_lut[mask], r_lut[mask]]
    return out


def az_r_to_px(az_deg: float, r_bin: float):
    """Polar (az degrees from North, range bin) → Cartesian pixel coords."""
    cx = cy = N_GRID // 2
    r_max = N_GRID // 2 - 5
    rad  = np.radians(az_deg)
    frac = r_bin / (N_RANGE - 1)
    return (cx + frac * r_max * np.sin(rad),
            cy - frac * r_max * np.cos(rad))
