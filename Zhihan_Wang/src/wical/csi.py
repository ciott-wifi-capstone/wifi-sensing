"""Read PicoScenes CSI and assign packets to switching states.

Packet times are approximated from the first and last host timestamps.
See docs/REPRODUCTION.md for the capture format and timing approximation.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from .io import warn
from .sqi import STATE_ORDER

READY = re.compile(r"^([\d.]+)\s+READY\s+(\w+)\s+chains=([\d,]+)")
MAPPING = dict(
    zip(
        STATE_ORDER,
        ("1,2,3", "4,5,6", "7,8,9", "10,11,12", "1,8,6", "10,8,6"),
        strict=True,
    )
)


def validate_decoded(raw):
    info = raw["CSI"]["Info"]
    if not len(info):
        raise ValueError("No decoded CSI records")
    shapes = np.c_[
        info["NumTones"],
        info["NumRx"],
        info["NumTx"].astype(int) + info["NumESS"].astype(int),
    ]
    if np.any(shapes != [30, 3, 1]):
        raise ValueError(
            "CSI metadata is not 30 tones x 3 Rx x 1 Tx; fixed buffers may truncate/pad other layouts"
        )
    csi = raw["CSI"]["CSI"]
    if csi.ndim != 4 or csi.shape[1:] != (30, 3, 1):
        raise ValueError("Unexpected decoded CSI tensor dimension")
    return np.transpose(csi, (0, 3, 2, 1))


def assign_states(h, event_ns, states, first_ns, last_ns):
    h = np.asarray(h)
    event_ns = np.asarray(event_ns, dtype=np.int64)
    if h.ndim != 4 or h.shape[1:] != (1, 3, 30) or len(h) < 2:
        raise ValueError("CSI must have shape (N>=2, 1 Tx, 3 Rx, 30 subcarriers)")
    if not np.isfinite(h).all():
        raise ValueError("CSI contains NaN/Inf")
    if (
        len(event_ns) != len(states)
        or not len(event_ns)
        or np.any(np.diff(event_ns) <= 0)
    ):
        raise ValueError(
            "READY times must be nonempty, strictly increasing and match states"
        )
    if set(states) - set(STATE_ORDER):
        raise ValueError("Unknown READY state")
    if last_ns <= first_ns:
        raise ValueError("Invalid host clock span")
    # Approximate packet times with uniform spacing across the host interval.
    times = first_ns + (last_ns - first_ns) * np.arange(len(h), dtype=np.int64) // (
        len(h) - 1
    )
    event_index = np.searchsorted(event_ns, times, side="right") - 1
    labels = np.full(len(h), "", dtype=object)
    valid = event_index >= 0
    labels[valid] = np.asarray(states)[event_index[valid]]
    if (~valid).any():
        warn(
            f"{int((~valid).sum())} packets precede the first READY event and remain unlabelled"
        )
    warn(
        "APPROXIMATE_PACKET_TIME: uniform interpolation of QC first/last host times; packet loss and exact state-boundary alignment are not established"
    )
    return {
        "H": h,
        "packet_state": labels,
        "event_index": event_index,
        "event_time_ns": event_ns,
        "packet_time_ns": times,
    }


def read_run(directory, *, crop_motion=False):
    directory = Path(directory)
    paths = sorted(directory.glob("*.csi"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one .csi file per run")
    qc = json.loads((directory / "metadata_qc.json").read_text())
    times = []
    states = []
    for line_no, line in enumerate(
        (directory / "events.log").read_text().splitlines(), 1
    ):
        match = READY.search(line)
        if "READY" in line and match is None:
            raise ValueError(f"Malformed READY event on line {line_no}")
        if match:
            state, chains = match.group(2, 3)
            if MAPPING.get(state) != chains:
                raise ValueError(f"Unexpected state/Rx chain mapping on line {line_no}")
            times.append(float(match.group(1)))
            states.append(state)
    if set(states) != set(STATE_ORDER):
        raise ValueError("Missing switching state in events.log")
    try:
        import csiread
    except ImportError as exc:
        raise ValueError("Raw decoding requires requirements-csi.txt") from exc
    reader = csiread.Picoscenes(str(paths[0]), {"CSI": [30, 3, 1]}, if_report=False)
    reader.read()
    h = validate_decoded(reader.raw)
    values = assign_states(
        h,
        np.rint(np.asarray(times) * 1e9).astype(np.int64),
        states,
        int(qc["host"]["first"]),
        int(qc["host"]["last"]),
    )
    values["run_dir"] = directory
    if "maximum_gap_s" in qc["host"]:
        warn(f"QC recorded maximum host gap: {qc['host']['maximum_gap_s']} seconds")
    else:
        warn("QC host maximum-gap metadata missing")
    if crop_motion:
        run = json.loads((directory / "run.json").read_text())
        start, end = int(run["start_ns"]), int(run["end_ns"])
        if end <= start:
            raise ValueError("Invalid run clock span")
        keep = (
            (values["packet_time_ns"] >= start)
            & (values["packet_time_ns"] < end)
            & (values["event_index"] >= 0)
        )
        for name in ("H", "packet_time_ns", "packet_state", "event_index"):
            values[name] = values[name][keep]
        if len(values["H"]) < 2:
            raise ValueError("No usable packets in run interval")
        values.update(run_start_ns=start, run_end_ns=end)
    return values


def state_ratio(data, state, rx_num, rx_den):
    if state not in STATE_ORDER or (rx_num, rx_den) not in ((0, 1), (0, 2), (1, 2)):
        raise ValueError("Invalid state or ordered receive-chain pair")
    mask = data["packet_state"] == state
    h = data["H"][mask]
    visits = data["event_index"][mask]
    ids = np.unique(visits)
    if len(ids) < 2:
        raise ValueError(f"Insufficient dwells for {state}")
    den = h[:, 0, rx_den, :]
    if np.any(np.abs(den) <= 1e-12):
        warn(
            f"{state}: near-zero CSI denominator; ratios may amplify noise despite epsilon protection"
        )
    packet = h[:, 0, rx_num, :] / (den + 1e-12)
    ratio = np.asarray(
        [
            np.median(packet[visits == v].real, axis=0)
            + 1j * np.median(packet[visits == v].imag, axis=0)
            for v in ids
        ]
    )
    t = (data["event_time_ns"][ids] - data["event_time_ns"][ids[0]]) / 1e9
    return ratio, t, float(1 / np.median(np.diff(t)))
