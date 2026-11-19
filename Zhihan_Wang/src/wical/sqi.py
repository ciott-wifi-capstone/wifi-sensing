"""WiCAL-adapted VitalCSI observation-level SQI (v1.0.1).

Score ratio-derived signals using breathing-band power, rate agreement
and spectral purity. Reference rates are not used.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.signal import (
    butter,
    detrend,
    filtfilt,
    find_peaks,
    savgol_filter,
    sosfiltfilt,
)


LOW_BPM = 5.0
HIGH_BPM = 35.0
AGREEMENT_BRPM = 4.0
SG_WINDOW_SEC = 2.0
SG_POLYORDER = 2
HIGHPASS_CUTOFF_HZ = 0.08
RESP_BANDPASS_ORDER = 8
TIE_ATOL = 1e-12

STATE_ORDER = ("G1", "G2", "G3", "G4", "C1", "C2")
PAIR_ORDER = ("Rx0/Rx1", "Rx0/Rx2", "Rx1/Rx2")
OBSERVATION_ORDER = tuple((state, pair) for state in STATE_ORDER for pair in PAIR_ORDER)
ORDER_INDEX = {key: index for index, key in enumerate(OBSERVATION_ORDER)}


def _as_finite_1d(signal: np.ndarray, name: str = "signal") -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 8:
        raise ValueError(
            f"{name} must be a one-dimensional signal with at least 8 samples"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return values


def _validate_fs(fs: float) -> float:
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 2.0 * HIGH_BPM / 60.0:
        raise ValueError(
            "sampling rate is invalid or below the 35 brpm Nyquist requirement"
        )
    return fs


def _ensure_odd(value: float) -> int:
    points = max(3, int(value))
    return points if points % 2 else points + 1


def _respiration_bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    values = _as_finite_1d(signal)
    fs = _validate_fs(fs)
    sos = butter(
        RESP_BANDPASS_ORDER,
        [LOW_BPM / 60.0, HIGH_BPM / 60.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, values)
    filtered -= np.mean(filtered)
    if not np.all(np.isfinite(filtered)):
        raise ValueError("bandpass output contains NaN or Inf")
    return filtered


def reconstruct_farsense_signals(
    dwell_ratio: np.ndarray,
    fs: float,
    angles: np.ndarray,
    selected: np.ndarray,
    selected_bnr: np.ndarray,
    selected_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct signals before and after the estimator high-pass filter."""

    ratio = np.asarray(dwell_ratio, dtype=np.complex128)
    fs = _validate_fs(fs)
    angles = np.asarray(angles, dtype=np.float64)
    selected = np.asarray(selected, dtype=int)
    gains = np.asarray(selected_bnr, dtype=np.float64)
    signs = np.asarray(selected_sign, dtype=int)
    if any(a.ndim != 1 for a in (angles, selected, gains, signs)):
        raise ValueError("selected metadata must be one-dimensional")
    if (
        not np.all(np.isfinite(gains))
        or np.any(gains < 0)
        or not np.isin(signs, [-1, 1]).all()
    ):
        raise ValueError(
            "selected gains must be finite/nonnegative and signs must be -1 or +1"
        )
    if len(set(selected.tolist())) != len(selected):
        raise ValueError("duplicate selected subcarrier index")
    if ratio.ndim != 2 or ratio.shape[1] != len(angles):
        raise ValueError("dwell_ratio/angles shape mismatch")
    if not (len(selected) == len(gains) == len(signs)) or len(selected) == 0:
        raise ValueError("selected metadata lengths are inconsistent")
    if not np.all(np.isfinite(ratio)) or not np.all(np.isfinite(angles)):
        raise ValueError("reconstruction inputs contain NaN or Inf")
    if np.any(selected < 0) or np.any(selected >= ratio.shape[1]):
        raise ValueError("selected subcarrier index is out of range")

    sg_points = _ensure_odd(SG_WINDOW_SEC * fs)
    if sg_points > ratio.shape[0]:
        raise ValueError("signal is too short for the SG window")
    b, a = butter(2, HIGHPASS_CUTOFF_HZ / (0.5 * fs), btype="high", analog=False)
    pre_filter = np.zeros(ratio.shape[0], dtype=np.float64)
    processed = np.zeros(ratio.shape[0], dtype=np.float64)

    for sc, gain, sign in zip(selected, gains, signs, strict=True):
        real = savgol_filter(np.real(ratio[:, sc]), sg_points, SG_POLYORDER)
        imag = savgol_filter(np.imag(ratio[:, sc]), sg_points, SG_POLYORDER)
        projected = real * np.cos(angles[sc]) + imag * np.sin(angles[sc])
        filtered = filtfilt(b, a, projected)
        filtered -= np.mean(filtered)
        pre_filter += float(gain) * int(sign) * projected
        processed += float(gain) * int(sign) * filtered
    return pre_filter, processed


def adapted_band_power_ratio(pre_filter_signal: np.ndarray, fs: float) -> float:
    """Power fraction in 5--35 brpm, computed before respiratory bandpass."""

    values = detrend(_as_finite_1d(pre_filter_signal), type="linear")
    fs = _validate_fs(fs)
    spectrum = np.fft.rfft(values * np.hanning(len(values)))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1.0 / fs)
    positive = freqs > 0.0
    total = float(np.sum(power[positive]))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("DEGENERATE_PRE_FILTER_POWER")
    band = (freqs >= LOW_BPM / 60.0) & (freqs <= HIGH_BPM / 60.0)
    return float(np.clip(np.sum(power[band]) / total, 0.0, 1.0))


def fft_respiration_rate(respiration_signal: np.ndarray, fs: float) -> float:
    """Estimate rate from Hann-windowed FFT bins in the 5--35 brpm band."""

    values = _as_finite_1d(respiration_signal)
    fs = _validate_fs(fs)
    values = values - np.mean(values)
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1.0 / fs)
    band_indices = np.flatnonzero(
        (freqs >= LOW_BPM / 60.0) & (freqs <= HIGH_BPM / 60.0)
    )
    if len(band_indices) == 0 or float(np.max(power[band_indices])) <= 0.0:
        raise ValueError("FFT_RATE_FAILED")
    # np.argmax returns the first maximum, fixing spectral ties to lower frequency.
    peak_index = int(band_indices[int(np.argmax(power[band_indices]))])
    return float(60.0 * freqs[peak_index])


def adapted_peak_count(respiration_signal: np.ndarray, fs: float) -> tuple[float, int]:
    """Adapted extrema/prominence breath count; this is explicitly not BSSF."""

    values = _as_finite_1d(respiration_signal)
    fs = _validate_fs(fs)
    maxima, _ = find_peaks(values)
    minima, _ = find_peaks(-values)
    extrema = np.sort(np.concatenate((maxima, minima)))
    if len(extrema) < 3:
        raise ValueError("TIME_COUNT_FAILED: insufficient extrema")
    differences = np.abs(np.diff(values[extrema]))
    q3 = float(np.quantile(differences, 0.75))
    if not np.isfinite(q3) or q3 <= 0.0:
        raise ValueError("TIME_COUNT_FAILED: degenerate extrema")
    min_distance = max(1, int(math.ceil(fs * 60.0 / HIGH_BPM)))
    peaks, _ = find_peaks(values, prominence=0.3 * q3, distance=min_distance)
    if len(peaks) < 2:
        raise ValueError("TIME_COUNT_FAILED: fewer than two peaks")
    duration = float(peaks[-1] - peaks[0]) / fs
    if duration <= 0.0:
        raise ValueError("TIME_COUNT_FAILED: non-positive peak span")
    return float((len(peaks) - 1) * 60.0 / duration), int(len(peaks))


def spectral_purity_index(respiration_signal: np.ndarray, fs: float) -> float:
    """Two-sided spectral-moment SPI with scale-invariant PSD normalization.

    v1.0.1 numerical clarification: the waveform and then its non-zero PSD are
    normalized before moments are evaluated.  This preserves the mathematical
    SPI ratio while preventing amplitude-only degeneracy.
    """

    values = _as_finite_1d(respiration_signal)
    fs = _validate_fs(fs)
    values = values - np.mean(values)
    amplitude = float(np.max(np.abs(values)))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        raise ValueError("DEGENERATE_SPECTRUM")
    values = values / amplitude
    window = np.hanning(len(values))
    spectrum = np.fft.fft(values * window)
    psd = np.abs(spectrum) ** 2 / (fs * float(np.sum(window**2)))
    psd[0] = 0.0
    total = float(np.sum(psd))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("DEGENERATE_SPECTRUM")
    psd /= total

    frequencies = np.fft.fftfreq(len(values), d=1.0 / fs)
    omega = 2.0 * np.pi * frequencies / fs
    delta_omega = 2.0 * np.pi / len(values)
    m0 = float(np.sum(psd) * delta_omega)
    m2 = float(np.sum((omega**2) * psd) * delta_omega)
    m4 = float(np.sum((omega**4) * psd) * delta_omega)
    denominator = m0 * m4
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise ValueError("DEGENERATE_SPECTRUM")
    return float(np.clip((m2 * m2) / denominator, 0.0, 1.0))


def _agreement_score(rr_fft: float, rr_time: float) -> float:
    if not np.isfinite(rr_fft) or not np.isfinite(rr_time):
        return 0.0
    return float(abs(float(rr_fft) - float(rr_time)) <= AGREEMENT_BRPM)


def compute_observation_sqi(
    pre_filter_signal: np.ndarray,
    processed_signal: np.ndarray,
    fs: float,
) -> dict[str, Any]:
    """Compute SQI components without accepting GT, error, or instructed RR."""

    pre_filter_signal = _as_finite_1d(pre_filter_signal, "pre_filter_signal")
    processed_signal = _as_finite_1d(processed_signal, "processed_signal")
    if len(pre_filter_signal) != len(processed_signal):
        raise ValueError("pre-filter and processed signal lengths differ")
    fs = _validate_fs(fs)
    reasons: list[str] = []
    try:
        gamma_band = adapted_band_power_ratio(pre_filter_signal, fs)
    except ValueError as exc:
        gamma_band = 0.0
        reasons.append(str(exc))

    try:
        respiration_signal = _respiration_bandpass(processed_signal, fs)
    except ValueError as exc:
        respiration_signal = None
        reasons.append(f"BANDPASS_FAILED: {exc}")

    rr_fft = float("nan")
    rr_time = float("nan")
    n_time_peaks = 0
    gamma_agree = 0.0
    gamma_spi = 0.0
    if respiration_signal is not None:
        try:
            rr_fft = fft_respiration_rate(respiration_signal, fs)
        except ValueError as exc:
            reasons.append(str(exc))
        try:
            rr_time, n_time_peaks = adapted_peak_count(respiration_signal, fs)
        except ValueError as exc:
            reasons.append(str(exc))
        if np.isfinite(rr_fft) and np.isfinite(rr_time):
            gamma_agree = _agreement_score(rr_fft, rr_time)
        try:
            gamma_spi = spectral_purity_index(respiration_signal, fs)
        except ValueError as exc:
            reasons.append(str(exc))

    sqi = float(gamma_band * gamma_agree * gamma_spi)
    return {
        "gamma_band": float(gamma_band),
        "rr_fft": rr_fft,
        "rr_time": rr_time,
        "n_time_peaks": int(n_time_peaks),
        "gamma_agree": float(gamma_agree),
        "gamma_spi": float(gamma_spi),
        "sqi": sqi,
        "status": "OK" if not reasons else "QUALITY_COMPONENT_FAILED",
        "failure_reasons": reasons,
    }


def select_by_sqi(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select the highest non-zero SQI; break ties using the fixed observation order."""

    candidates = list(rows)
    if not candidates:
        return {"status": "NO_VALID_OBSERVATION", "selected": None, "tie_count": 0}
    seen = set()
    for row in candidates:
        key = (str(row["state"]), str(row["pair"]))
        if key not in ORDER_INDEX:
            raise ValueError(f"unknown observation identity: {key}")
        if key in seen:
            raise ValueError(f"duplicate observation identity: {key}")
        seen.add(key)
        score = float(row["sqi"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid SQI for {key}: {score}")
    maximum = max(float(row["sqi"]) for row in candidates)
    if maximum <= TIE_ATOL:
        return {
            "status": "NO_QUALITY_DIFFERENTIATION",
            "selected": None,
            "tie_count": sum(
                abs(float(row["sqi"]) - maximum) <= TIE_ATOL for row in candidates
            ),
        }
    tied = [row for row in candidates if abs(float(row["sqi"]) - maximum) <= TIE_ATOL]
    chosen = min(
        tied, key=lambda row: ORDER_INDEX[(str(row["state"]), str(row["pair"]))]
    )
    return {
        "status": "SELECTED",
        "selected": chosen,
        "tie_count": len(tied),
    }
