"""
radar/live.py — live_tgt_pos, live_tgt_alive, live_init, live_tick.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

from radar.sessions import (
    N_SW, N_AZ, N_RANGE,
    MAX_LIVE_TGTS, _LIVE_FRAME_LO, _LIVE_FRAME_HI,
    _LIVE_SPAWN_LO, _LIVE_SPAWN_HI,
    load_params, load_gru_session,
)
from radar.inference import _gru_peaks
from src.data.ppi_generator import generate_multitarget_sweep
from src.baseline.ppi_cfar_kf import PPICFARDetector, PPIKalmanTracker


def _live_tgt_pos(tgt: dict, sw: int) -> tuple:
    """(az_deg, r_bin) of a target at absolute sweep index sw."""
    age = sw - tgt["born"]
    return (tgt["az0"] + age * tgt["vt"]) % 360.0, tgt["r0"] + age * tgt["vr"]


def _live_tgt_alive(tgt: dict, sw: int) -> bool:
    _, r = _live_tgt_pos(tgt, sw)
    return 5 <= r <= N_RANGE - 3


def _live_init():
    """Initialise (or reset) the live-radar session state."""
    seed = int(np.random.default_rng().integers(0, 999_999))
    st.session_state["live"] = {
        "running":        False,
        "sweep_count":    0,
        "seed":           seed,
        "targets":        [],
        "next_id":        0,
        "spawn_in":       2,
        "sweep_buf":      np.zeros((N_SW, N_AZ, N_RANGE), dtype=np.float32),
        "gru_h":          np.zeros((1, 1, N_AZ, N_RANGE), dtype=np.float32),
        "gru_nfloor":     np.ones(N_RANGE, dtype=np.float32),
        "kf_cfar":        PPIKalmanTracker(),
        "kf_gru":         PPIKalmanTracker(),
        "conf_hist":      {},   # {id: {"cfar": [], "gru": []}}
        "confirm":        {},   # {id: {"cfar_kf": latency|None, "gru_kf": latency|None}}
        "lat_cfar":       [],
        "lat_gru":        [],
        "last_sweep":     np.zeros((N_AZ, N_RANGE), dtype=np.float32),
        "last_cfar_det":  np.zeros((N_AZ, N_RANGE), dtype=bool),
        "last_gru_map":   np.zeros((N_AZ, N_RANGE), dtype=np.float32),
        "frame_s":        0.6,
    }


def _live_tick():
    """Advance the live radar by one full antenna sweep."""
    params = load_params()
    gru_session = load_gru_session()

    s   = st.session_state["live"]
    sw  = s["sweep_count"]
    rng = np.random.default_rng(s["seed"] + sw * 97)

    # Expire targets that have left the display area
    s["targets"] = [t for t in s["targets"] if _live_tgt_alive(t, sw)]

    # Spawn a new target?
    s["spawn_in"] -= 1
    if s["spawn_in"] <= 0:
        if len(s["targets"]) < MAX_LIVE_TGTS:
            tid = s["next_id"]
            tgt = {
                "id":     tid,
                "az0":    float(rng.uniform(0, 360)),
                "r0":     float(rng.uniform(N_RANGE * 0.75, N_RANGE - 6)),
                "vr":     float(rng.uniform(-2.5, -0.8)),
                "vt":     float(rng.uniform(-2.0, 2.0)),
                "snr_db": float(rng.uniform(-5, 20)),
                "born":   sw,
            }
            s["targets"].append(tgt)
            s["next_id"] += 1
            s["conf_hist"][tid] = {"cfar": [], "gru": []}
            s["confirm"][tid]   = {"cfar_kf": None, "gru_kf": None}
        s["spawn_in"] = int(rng.integers(_LIVE_SPAWN_LO, _LIVE_SPAWN_HI + 1))

    # Generate this sweep
    sweep = generate_multitarget_sweep(sw, s["targets"], params, rng)
    s["last_sweep"] = sweep

    # Rolling buffer — shift left, push new sweep at back
    buf = s["sweep_buf"]
    buf[:-1] = buf[1:]
    buf[-1]  = sweep

    # CFAR detection map (for display) + CFAR+KF update
    cfar_det = PPICFARDetector().detect(sweep)
    s["last_cfar_det"] = cfar_det
    s["kf_cfar"].update(sweep)   # uses internal CFAR for peak extraction

    # GRU single-step update
    if gru_session is not None:
        nf       = s["gru_nfloor"].clip(1e-3)
        sw_norm  = (sweep / nf)[np.newaxis, np.newaxis].astype(np.float32)
        gru_prob, h_next = gru_session.run(None, {"sweep_norm": sw_norm, "h_in": s["gru_h"]})
        s["gru_h"]      = h_next
        s["gru_nfloor"] = 0.9 * s["gru_nfloor"] + 0.1 * np.percentile(sweep, 10, axis=0)
        gru_map = gru_prob[0, 0]
    else:
        gru_map = np.zeros((N_AZ, N_RANGE), dtype=np.float32)
    s["last_gru_map"] = gru_map
    s["kf_gru"].update_from_peaks(_gru_peaks(gru_map), N_AZ, N_RANGE)

    # Per-target confidence sampling and confirmation check
    cfar_trks = s["kf_cfar"].track_positions()
    gru_trks  = s["kf_gru"].track_positions()

    for tgt in s["targets"]:
        tid  = tgt["id"]
        az_t, r_t = _live_tgt_pos(tgt, sw)
        az_b = int(az_t / 360 * N_AZ) % N_AZ
        r_b  = int(np.clip(r_t, 0, N_RANGE - 1))

        if tid in s["conf_hist"]:
            s["conf_hist"][tid]["cfar"].append(float(cfar_det[az_b, r_b]))
            s["conf_hist"][tid]["gru"].append(float(gru_map[az_b, r_b]))

        if tid in s["confirm"]:
            conf = s["confirm"][tid]
            for key, trks in [("cfar_kf", cfar_trks), ("gru_kf", gru_trks)]:
                if conf[key] is not None:
                    continue
                for az_tr, r_tr in trks:
                    az_tr_b = int(float(az_tr) % N_AZ)
                    daz = min(abs(az_tr_b - az_b), N_AZ - abs(az_tr_b - az_b))
                    if np.sqrt(daz ** 2 + (float(r_tr) - r_b) ** 2) < 8:
                        lat = sw - tgt["born"] + 1
                        conf[key] = lat
                        {"cfar_kf": s["lat_cfar"], "gru_kf": s["lat_gru"]}[key].append(lat)
                        break

    s["frame_s"]     = float(rng.uniform(_LIVE_FRAME_LO, _LIVE_FRAME_HI))
    s["sweep_count"] += 1
