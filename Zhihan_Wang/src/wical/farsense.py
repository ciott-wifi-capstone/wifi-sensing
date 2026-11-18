"""FarSense-style respiration estimator adapted for switched-array ratios.
See docs/SOURCES.md for the local source and method reference.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, detrend, filtfilt, find_peaks, savgol_filter

ORIGINAL_FS = 1000.0
DOWNSAMPLE_FACTOR = 10
TX_IDX = 0
RX_NUM = 1
RX_DEN = 2
BREATH_BPM_BAND = (6.0, 30.0)
NUM_PROJECTION_ANGLES = 100
BNR_NFFT = 8192
SG_WINDOW_SEC = 5.0
SG_POLYORDER = 2
HIGHPASS_CUTOFF_HZ = 0.08
TOP_N_SUBCARRIERS = 3
BNR_THRESHOLD = 0.015
EPS = 1e-12


def ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def highpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 2
) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs), btype="high", analog=False)
    return filtfilt(b, a, data)


def normalized_autocorr_fft(signal: np.ndarray) -> np.ndarray:
    signal = detrend(np.asarray(signal, dtype=np.float64), type="linear")
    std = float(np.std(signal))
    if std < EPS:
        return np.zeros(len(signal), dtype=np.float64)
    signal /= std
    n = len(signal)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(signal, n=nfft)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n]
    return ac / ac[0] if abs(ac[0]) > EPS else ac


def extract_bpm_from_ac(ac: np.ndarray, fs: float) -> float:
    peaks, _ = find_peaks(ac)
    troughs, _ = find_peaks(-ac)
    if len(peaks) == 0 or len(troughs) == 0:
        return 0.0
    later_peaks = peaks[peaks > troughs[0]]
    if len(later_peaks) == 0:
        return 0.0
    period_sec = float(later_peaks[0]) / fs
    return 60.0 / period_sec if period_sec > 0 else 0.0


def compute_bnr(signal: np.ndarray, fs: float) -> float:
    signal = detrend(signal, type="linear")
    spectrum = np.fft.rfft(signal * np.hanning(len(signal)), n=BNR_NFFT)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(BNR_NFFT, d=1.0 / fs)
    band = (freqs >= BREATH_BPM_BAND[0] / 60.0) & (freqs <= BREATH_BPM_BAND[1] / 60.0)
    total = float(np.sum(power[1:]))
    if not np.any(band) or total <= EPS:
        return 0.0
    return float(np.max(power[band]) / total)


def estimate_window(csi: np.ndarray, *, fs: float, sg_window_sec: float = 2.0) -> dict:
    thetas = np.linspace(0.0, np.pi, NUM_PROJECTION_ANGLES, endpoint=False)
    signals: list[np.ndarray] = []
    autocorrs: list[np.ndarray] = []
    bnrs: list[float] = []
    angles: list[float] = []

    for sc in range(csi.shape[-1]):
        ratio = csi[:, TX_IDX, RX_NUM, sc] / (csi[:, TX_IDX, RX_DEN, sc] + EPS)
        sg_points = ensure_odd(sg_window_sec * fs)
        real = savgol_filter(
            np.real(ratio), window_length=sg_points, polyorder=SG_POLYORDER
        )
        imag = savgol_filter(
            np.imag(ratio), window_length=sg_points, polyorder=SG_POLYORDER
        )
        best_bnr = -1.0
        best_theta = 0.0

        for theta in thetas:
            projected = real * np.cos(theta) + imag * np.sin(theta)
            score = compute_bnr(projected, fs)
            if score > best_bnr:
                best_bnr = score
                best_theta = float(theta)

        projected = real * np.cos(best_theta) + imag * np.sin(best_theta)
        filtered = highpass_filter(projected, HIGHPASS_CUTOFF_HZ, fs)
        filtered -= np.mean(filtered)
        signals.append(filtered)
        autocorrs.append(normalized_autocorr_fft(filtered))
        bnrs.append(best_bnr)
        angles.append(best_theta)

    bnr_array = np.asarray(bnrs)
    passed = np.flatnonzero(bnr_array >= BNR_THRESHOLD)
    if len(passed):
        selected = passed[np.argsort(bnr_array[passed])[::-1]][:TOP_N_SUBCARRIERS]
    else:
        selected = np.asarray([int(np.argmax(bnr_array))])

    reference = signals[int(selected[0])]
    combined_signal = np.zeros_like(reference)
    combined_ac = np.zeros_like(autocorrs[0])
    signs: list[int] = []

    for sc in selected:
        sign = 1 if np.corrcoef(reference, signals[int(sc)])[0, 1] >= 0 else -1
        gain = bnr_array[int(sc)]
        combined_signal += gain * sign * signals[int(sc)]
        combined_ac += gain * autocorrs[int(sc)]
        signs.append(sign)

    if abs(combined_ac[0]) > EPS:
        combined_ac /= combined_ac[0]

    return {
        "bpm": extract_bpm_from_ac(combined_ac, fs),
        "signal": combined_signal,
        "ac": combined_ac,
        "fs": fs,
        "selected": selected.astype(int),
        "selected_bnr": bnr_array[selected],
        "selected_sign": np.asarray(signs),
        "angles": np.asarray(angles),
    }
