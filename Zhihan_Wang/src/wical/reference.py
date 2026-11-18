"""Read NeuLog belt recordings and estimate rates from breath landmarks.

Reference windows and their alignment must be supplied separately.
This module is not used by the automatic observation selectors.
"""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from .io import check_time, warn


def parse_neulog_time(value):
    value = str(value).strip().replace("'", "")
    sign = -1 if value.startswith("-") else 1
    hours, minutes, seconds = map(float, value.lstrip("-").split(":"))
    result = sign * (hours * 3600.0 + minutes * 60.0 + seconds)
    if not np.isfinite(result):
        raise ValueError("Non-finite NeuLog timestamp")
    return result


def load_neulog_gt(path):
    pairs = []
    footer_started = False
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for _ in range(7):
            next(reader, None)
        for line, row in enumerate(reader, 8):
            if not row or all(not x.strip() for x in row):
                warn(f"Blank NeuLog row at line {line}")
                continue
            if len(row) == 1 and (
                row[0].startswith("sys info:[")
                or row[0]
                == "Attention! Any change in the file may prevent reading its data."
            ):
                footer_started = True
                warn(f"NeuLog export footer at line {line}; not a measurement")
                continue
            if footer_started:
                raise ValueError(f"Unexpected data after NeuLog footer at line {line}")
            if len(row) < 2:
                raise ValueError(f"NeuLog line {line}: requires time and value")
            try:
                t = parse_neulog_time(row[0])
                value = float(row[1])
                if not np.isfinite(value):
                    raise ValueError("non-finite value")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"NeuLog line {line}: invalid time/value") from exc
            # Store a sample only when both timestamp and value are valid.
            pairs.append((t, value))
    if len(pairs) < 2:
        raise ValueError("NeuLog contains fewer than two valid samples")
    data = np.asarray(pairs, float)
    time = check_time(data[:, 0], len(data), "NeuLog time")
    time = time - time[0]
    gaps = np.diff(time)
    if not np.allclose(gaps, np.median(gaps), atol=1e-9, rtol=1e-6):
        warn(
            f"NeuLog timestamp gaps: maximum {np.max(gaps):.9g}s; detector retains reciprocal median sampling rate"
        )
    return time, data[:, 1], float(1 / np.median(gaps))


def detect_gt_rr(time, signal, fs):
    signal = np.asarray(signal, float)
    time = check_time(time, len(signal), "reference window time")
    if (
        signal.ndim != 1
        or not np.isfinite(signal).all()
        or not np.isfinite(fs)
        or fs <= 0
    ):
        raise ValueError("Invalid reference signal or sampling rate")
    # Smooth the belt signal, then count landmarks over their time span.
    sg_points = max(5, int(round(0.35 * fs)))
    if sg_points % 2 == 0:
        sg_points += 1
    max_odd = len(signal) if len(signal) % 2 else len(signal) - 1
    sg_points = min(sg_points, max_odd)
    smoothed = signal.copy() if sg_points < 5 else savgol_filter(signal, sg_points, 2)
    minimum_distance = max(1, int(round(0.70 * (60.0 / 30.0) * fs)))
    robust_range = np.nanpercentile(smoothed, 95) - np.nanpercentile(smoothed, 5)
    indices, _ = find_peaks(
        smoothed, distance=minimum_distance, prominence=0.15 * robust_range
    )
    landmarks = time[indices]
    if len(landmarks) < 2:
        raise ValueError("NeuLog detector returned fewer than two landmarks")
    return float(60.0 * (len(landmarks) - 1) / (landmarks[-1] - landmarks[0]))
