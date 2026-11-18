"""Adapt dwell ratios to the respiration estimator and calculate observation SQI."""

from __future__ import annotations
import warnings
import numpy as np
from . import farsense
from .sqi import (
    compute_observation_sqi,
    reconstruct_farsense_signals,
    RESP_BANDPASS_ORDER,
)
from .io import check_time, finite, warn


def estimate_ratio(ratio, fs, time=None):
    ratio = np.asarray(ratio, dtype=np.complex128)
    fs = finite(fs, "fs_hz")
    if fs <= 2 * 35 / 60:
        raise ValueError("fs_hz must exceed the 35 bpm SQI Nyquist requirement")
    minimum_length = max(
        3 * (2 * RESP_BANDPASS_ORDER + 1) + 1, farsense.ensure_odd(2 * fs)
    )
    if ratio.ndim != 2 or ratio.shape[1] != 30 or ratio.shape[0] < minimum_length:
        raise ValueError(
            f"ratio must have shape (N, 30), N >= {minimum_length} for the existing SG and SQI filter padding"
        )
    if not np.isfinite(ratio).all():
        raise ValueError("ratio contains NaN/Inf")
    if ratio.shape[0] > farsense.BNR_NFFT:
        warn(
            "BNR uses an 8192-point FFT; longer input is truncated for this calculation"
        )
    if time is not None:
        time = check_time(time, len(ratio))
        dt = np.diff(time)
        if not np.allclose(dt, 1 / fs, rtol=1e-6, atol=1e-9):
            warn(
                f"Nonuniform dwell times: median gap={np.median(dt):.9g}s, maximum gap={np.max(dt):.9g}s; estimator uses fs_hz={fs:.9g}, without resampling"
            )
    adapter = np.zeros((len(ratio), 1, 3, 30), dtype=np.complex128)
    adapter[:, 0, 1, :] = ratio
    adapter[:, 0, 2, :] = 1.0
    result = farsense.estimate_window(adapter, fs=fs, sg_window_sec=2.0)
    for key in ("bpm", "signal", "ac", "selected_bnr", "angles"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Non-finite estimator output: {key}")
    if result["bpm"] == 0:
        warn("NO_ACF_RATE: zero indicates that no candidate period was found")
    if result["bpm"] and not 6 <= result["bpm"] <= 30:
        warn(
            "ACF_RATE_OUTSIDE_BNR_BAND: the ACF rate search is not restricted to 6--30 bpm"
        )
    pre, rebuilt = reconstruct_farsense_signals(
        ratio,
        fs,
        result["angles"],
        result["selected"],
        result["selected_bnr"],
        result["selected_sign"],
    )
    if not np.allclose(rebuilt, result["signal"], rtol=1e-10, atol=1e-12):
        raise ValueError("Reconstructed SQI waveform differs from FarSense output")
    quality = compute_observation_sqi(pre, result["signal"], fs)
    if quality["failure_reasons"]:
        warn("SQI_COMPONENT_FAILURE: " + "; ".join(quality["failure_reasons"]))
    return result, quality


def validate_observation_arrays(arrays, key):
    ratio = np.asarray(arrays[key + "__dwell_ratio"])
    fs_array = np.asarray(arrays[key + "__processed_fs"])
    if fs_array.shape != (1,):
        raise ValueError(f"{key}: processed_fs must contain exactly one value")
    time = check_time(arrays[key + "__state_time"], len(ratio), key + " time")
    return ratio, float(fs_array[0]), time
