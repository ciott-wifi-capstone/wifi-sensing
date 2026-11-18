"""CSI-change detection, spatial response features and fingerprint comparison."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from .sqi import STATE_ORDER, PAIR_ORDER, OBSERVATION_ORDER
from .io import warn

STATES = STATE_ORDER
OBSERVATIONS = OBSERVATION_ORDER
OBS_LABELS = tuple(f"{state}\n{pair}" for state, pair in OBSERVATIONS)
# Shared detection and feature parameters for all recordings.
BIN_WIDTH_S = 0.50
EVENT_MIN_SEPARATION_S = 12.0
EVENT_HEIGHT_Z = 3.0
EVENT_PROMINENCE_Z = 1.0
EVENT_SUPPORT_Z = 3.0
EVENT_MIN_SUPPORT_OBSERVATIONS = 3
BASELINE_START_S = -8.0
BASELINE_END_S = -3.0
RESPONSE_START_S = -1.0
RESPONSE_END_S = 4.0
MIN_WINDOW_DWELLS = 15
AMPLITUDE_SCALE_FLOOR_DB = 0.05
EPS = 1e-12


@dataclass
class Observation:
    state: str
    pair: str
    time_s: np.ndarray
    ratio: np.ndarray
    amplitude_db_sc: np.ndarray
    amplitude_db: np.ndarray
    diff_time_s: np.ndarray
    diff_energy: np.ndarray
    diff_energy_sc: np.ndarray


def robust_scale(values: np.ndarray, floor: float = EPS) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, floor
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return center, max(scale, floor)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center, scale = robust_scale(values)
    return (values - center) / scale


def bin_median(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Median-bin samples without interpolating switching blanks or missing data."""

    output = np.full(len(grid), np.nan, dtype=float)
    edges = np.concatenate(([grid[0] - BIN_WIDTH_S / 2.0], grid + BIN_WIDTH_S / 2.0))
    indices = np.searchsorted(edges, times, side="right") - 1
    for index in range(len(grid)):
        selected = values[indices == index]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            output[index] = float(np.median(selected))
    return output


def fill_detection_gaps(values: np.ndarray) -> np.ndarray:
    """Interpolate missing detection bins, including long gaps and edge extension."""

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.all():
        warn(
            f"DETECTION_GAP_FILL: {int((~valid).sum())}/{len(values)} bins missing; interpolation includes long gaps and edge extension; observation arrays are not interpolated"
        )
    if valid.sum() < 2:
        return np.zeros_like(values)
    indices = np.arange(len(values))
    return np.interp(indices, indices[valid], values[valid])


def build_observation(
    run_data: dict[str, Any], state: str, pair: str, rx_num: int, rx_den: int
) -> Observation:
    state_mask = run_data["packet_state"] == state
    h_state = run_data["H"][state_mask]
    dwell_index = run_data["event_index"][state_mask]
    dwell_ids = np.unique(dwell_index)
    dwell_ids = dwell_ids[
        (run_data["event_time_ns"][dwell_ids] >= run_data["run_start_ns"])
        & (run_data["event_time_ns"][dwell_ids] < run_data["run_end_ns"])
    ]
    if len(dwell_ids) < 2:
        raise RuntimeError(f"insufficient dwells for {state} {pair}")

    packet_ratio = h_state[:, 0, rx_num, :] / (h_state[:, 0, rx_den, :] + EPS)
    ratio = np.asarray(
        [
            np.median(packet_ratio[dwell_index == dwell].real, axis=0)
            + 1j * np.median(packet_ratio[dwell_index == dwell].imag, axis=0)
            for dwell in dwell_ids
        ],
        dtype=np.complex128,
    )
    time_s = (run_data["event_time_ns"][dwell_ids] - run_data["run_start_ns"]) / 1e9
    amplitude_db_sc = 20.0 * np.log10(np.maximum(np.abs(ratio), EPS))
    amplitude_db = np.median(amplitude_db_sc, axis=1)

    delta = np.diff(ratio, axis=0)
    reference_power = np.maximum(np.abs(ratio[:-1]) ** 2, EPS)
    diff_energy_sc = np.abs(delta) ** 2 / reference_power
    diff_energy = np.median(diff_energy_sc, axis=1)
    diff_time_s = (time_s[:-1] + time_s[1:]) / 2.0
    return Observation(
        state=state,
        pair=pair,
        time_s=time_s,
        ratio=ratio,
        amplitude_db_sc=amplitude_db_sc,
        amplitude_db=amplitude_db,
        diff_time_s=diff_time_s,
        diff_energy=diff_energy,
        diff_energy_sc=diff_energy_sc,
    )


def build_detection_score(
    observations: dict[tuple[str, str], Observation], duration_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(BIN_WIDTH_S / 2.0, duration_s, BIN_WIDTH_S)
    observation_scores = []
    for observation in observations.values():
        amplitude_step = np.abs(np.diff(observation.amplitude_db))
        amplitude_time = (observation.time_s[:-1] + observation.time_s[1:]) / 2.0
        amp_binned = bin_median(amplitude_time, amplitude_step, grid)
        diff_binned = bin_median(
            observation.diff_time_s,
            np.log10(np.maximum(observation.diff_energy, EPS)),
            grid,
        )
        amp_z = np.maximum(0.0, robust_z(fill_detection_gaps(amp_binned)))
        diff_z = np.maximum(0.0, robust_z(fill_detection_gaps(diff_binned)))
        observation_scores.append(np.maximum(amp_z, diff_z))

    score_matrix = np.vstack(observation_scores)
    consensus = np.median(score_matrix, axis=0)
    # Fixed 3-bin centered median smoothing on the 0.5 s grid.
    padded = np.pad(consensus, (1, 1), mode="edge")
    smooth = np.asarray(
        [np.median(padded[index : index + 3]) for index in range(len(consensus))]
    )
    support = np.sum(score_matrix >= EVENT_SUPPORT_Z, axis=0)
    return grid, smooth, support, score_matrix


def detect_candidate_events(
    grid: np.ndarray,
    score: np.ndarray,
    support: np.ndarray,
    duration_s: float,
) -> list[dict[str, Any]]:
    distance = max(1, int(round(EVENT_MIN_SEPARATION_S / BIN_WIDTH_S)))
    peaks, properties = find_peaks(
        score,
        height=EVENT_HEIGHT_Z,
        prominence=EVENT_PROMINENCE_Z,
        distance=distance,
    )
    events: list[dict[str, Any]] = []
    for peak, height, prominence in zip(
        peaks, properties["peak_heights"], properties["prominences"], strict=True
    ):
        event_time = float(grid[peak])
        window_valid = (
            event_time + BASELINE_START_S >= 0.0
            and event_time + RESPONSE_END_S <= duration_s
        )
        support_valid = int(support[peak]) >= EVENT_MIN_SUPPORT_OBSERVATIONS
        if not (window_valid and support_valid):
            continue
        events.append(
            {
                "event_index": len(events) + 1,
                "event_time_s": event_time,
                "score": float(height),
                "prominence": float(prominence),
                "support_count": int(support[peak]),
                "event_status": "CSI_CANDIDATE",
                "action_marker_status": "NO_INDEPENDENT_ACTION_MARKER",
            }
        )
    return events


def event_features(
    dataset: str,
    motion: str,
    run_name: str,
    event: dict[str, Any],
    observation: Observation,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_time = float(event["event_time_s"])
    baseline_start = event_time + BASELINE_START_S
    baseline_end = event_time + BASELINE_END_S
    response_start = event_time + RESPONSE_START_S
    response_end = event_time + RESPONSE_END_S

    baseline = (observation.time_s >= baseline_start) & (
        observation.time_s < baseline_end
    )
    response = (observation.time_s >= response_start) & (
        observation.time_s < response_end
    )
    baseline_diff = (observation.diff_time_s >= baseline_start) & (
        observation.diff_time_s < baseline_end
    )
    response_diff = (observation.diff_time_s >= response_start) & (
        observation.diff_time_s < response_end
    )
    valid = (
        int(baseline.sum()) >= MIN_WINDOW_DWELLS
        and int(response.sum()) >= MIN_WINDOW_DWELLS
        and int(baseline_diff.sum()) >= MIN_WINDOW_DWELLS - 1
        and int(response_diff.sum()) >= MIN_WINDOW_DWELLS - 1
    )
    if not valid:
        raise RuntimeError(
            f"window coverage failed: {dataset}/{run_name}/E{event['event_index']}/"
            f"{observation.state}/{observation.pair}"
        )

    baseline_amp = observation.amplitude_db[baseline]
    response_amp = observation.amplitude_db[response]
    baseline_center, baseline_scale = robust_scale(
        baseline_amp, floor=AMPLITUDE_SCALE_FLOOR_DB
    )
    amplitude_shift_db = float(np.median(response_amp) - baseline_center)
    amplitude_abs_change_db = float(np.median(np.abs(response_amp - baseline_center)))
    amplitude_shift_z = amplitude_shift_db / baseline_scale
    amplitude_abs_response_z = amplitude_abs_change_db / baseline_scale

    baseline_energy = float(np.median(observation.diff_energy[baseline_diff]))
    response_energy = float(np.median(observation.diff_energy[response_diff]))
    differential_energy_db = float(
        10.0 * np.log10((response_energy + EPS) / (baseline_energy + EPS))
    )

    baseline_complex = np.median(
        observation.ratio[baseline].real, axis=0
    ) + 1j * np.median(observation.ratio[baseline].imag, axis=0)
    complex_displacement = np.abs(
        observation.ratio[response] - baseline_complex[None, :]
    ) / np.maximum(np.abs(baseline_complex[None, :]), EPS)
    complex_displacement_norm = float(np.median(complex_displacement))

    base_amp_sc = np.median(observation.amplitude_db_sc[baseline], axis=0)
    response_amp_sc = np.median(observation.amplitude_db_sc[response], axis=0)
    amp_shift_sc = response_amp_sc - base_amp_sc
    base_diff_sc = np.median(observation.diff_energy_sc[baseline_diff], axis=0)
    response_diff_sc = np.median(observation.diff_energy_sc[response_diff], axis=0)
    diff_db_sc = 10.0 * np.log10((response_diff_sc + EPS) / (base_diff_sc + EPS))

    row = {
        "dataset": dataset,
        "motion": motion,
        "run": run_name,
        "event_id": f"E{event['event_index']}",
        "event_time_s": event_time,
        "event_status": event["event_status"],
        "action_marker_status": event["action_marker_status"],
        "state": observation.state,
        "pair": observation.pair,
        "baseline_start_s": baseline_start,
        "baseline_end_s": baseline_end,
        "response_start_s": response_start,
        "response_end_s": response_end,
        "baseline_dwells": int(baseline.sum()),
        "response_dwells": int(response.sum()),
        "amplitude_shift_db": amplitude_shift_db,
        "amplitude_abs_change_db": amplitude_abs_change_db,
        "baseline_amplitude_scale_db": baseline_scale,
        "amplitude_shift_z": amplitude_shift_z,
        "amplitude_abs_response_z": amplitude_abs_response_z,
        "baseline_diff_energy": baseline_energy,
        "response_diff_energy": response_energy,
        "differential_energy_db": differential_energy_db,
        "complex_displacement_norm": complex_displacement_norm,
    }
    subcarrier_rows = [
        {
            "dataset": dataset,
            "motion": motion,
            "run": run_name,
            "event_id": f"E{event['event_index']}",
            "event_time_s": event_time,
            "event_status": event["event_status"],
            "state": observation.state,
            "pair": observation.pair,
            "subcarrier_index": subcarrier,
            "amplitude_shift_db": float(amp_shift_sc[subcarrier]),
            "differential_energy_db": float(diff_db_sc[subcarrier]),
        }
        for subcarrier in range(observation.ratio.shape[1])
    ]
    return row, subcarrier_rows


def normalized_run_vector(run_rows: list[dict[str, Any]]) -> np.ndarray:
    by_observation: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in OBSERVATIONS
    }
    for row in run_rows:
        by_observation[(row["state"], row["pair"])].append(row)
    amp = []
    diff = []
    for key in OBSERVATIONS:
        rows = by_observation[key]
        if not rows:
            raise RuntimeError(f"run fingerprint missing observation {key}")
        amp.append(float(np.median([row["amplitude_abs_response_z"] for row in rows])))
        diff.append(float(np.median([row["differential_energy_db"] for row in rows])))
    return np.concatenate((robust_z(np.asarray(amp)), robust_z(np.asarray(diff))))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if np.isfinite(value) else float("nan")


FINGERPRINT_FIELDS = [
    "dataset",
    "motion",
    "run",
    "candidate_event_count",
    "state",
    "pair",
    "median_amplitude_shift_db",
    "median_amplitude_shift_z",
    "median_amplitude_abs_response_z",
    "median_differential_energy_db",
    "median_complex_displacement_norm",
]
RUN_SIMILARITY_FIELDS = [
    "left_motion",
    "left_dataset",
    "left_run",
    "right_motion",
    "right_dataset",
    "right_run",
    "same_action_condition",
    "spearman_r",
]
ACTION_SIMILARITY_FIELDS = ["left_action", "right_action", "spearman_r"]


def build_run_fingerprints(feature_rows, run_results):
    """Compute per-run median responses and run/condition correlations."""
    registered = {(r["dataset"], r["run"]): r for r in run_results}
    if len(registered) != len(run_results):
        raise ValueError("Duplicate dataset/run identity in motion summaries")
    events = {}
    numeric = (
        "amplitude_shift_db",
        "amplitude_shift_z",
        "amplitude_abs_response_z",
        "differential_energy_db",
        "complex_displacement_norm",
    )
    for row in feature_rows:
        key = (row["dataset"], row["run"])
        if key not in registered or row["motion"] != registered[key]["motion"]:
            raise ValueError("Unregistered or inconsistent motion/run identity")
        observation = (row["state"], row["pair"])
        if observation not in OBSERVATIONS:
            raise ValueError("Unexpected observation in fingerprint features")
        if not all(np.isfinite(float(row[field])) for field in numeric):
            raise ValueError("Nonfinite event features cannot form a run fingerprint")
        identities = events.setdefault((*key, row["event_id"]), set())
        if observation in identities:
            raise ValueError("Duplicate event/observation in fingerprint features")
        identities.add(observation)
    if any(s != set(OBSERVATIONS) for s in events.values()):
        raise ValueError("Run fingerprint requires all 18 observations per event")
    for key, result in registered.items():
        count = sum(k[:2] == key for k in events)
        if count != len(result["events"]):
            raise ValueError("Candidate count and fingerprint event coverage disagree")
        if not count:
            warn(f"{key}: no candidate events; run has no fingerprint or similarity")
    if not feature_rows:
        return {
            "run_fingerprint_rows": [],
            "similarity_rows": [],
            "action_similarity_rows": [],
            "run_keys": [],
            "motions": [],
            "run_similarity": np.empty((0, 0)),
            "action_similarity": np.empty((0, 0)),
            "within_condition_repeatability": {},
        }
    run_fingerprint_rows: list[dict[str, Any]] = []
    run_keys: list[tuple[str, str, str]] = []
    run_vectors: list[np.ndarray] = []
    for result in run_results:
        key = (result["motion"], result["dataset"], result["run"])
        rows = [
            row
            for row in feature_rows
            if row["dataset"] == result["dataset"] and row["run"] == result["run"]
        ]
        if not rows:
            continue
        run_keys.append(key)
        run_vectors.append(normalized_run_vector(rows))
        for observation_key in OBSERVATIONS:
            selected = [
                row for row in rows if (row["state"], row["pair"]) == observation_key
            ]
            run_fingerprint_rows.append(
                {
                    "dataset": result["dataset"],
                    "motion": result["motion"],
                    "run": result["run"],
                    "candidate_event_count": len(result["events"]),
                    "state": observation_key[0],
                    "pair": observation_key[1],
                    "median_amplitude_shift_db": float(
                        np.median([row["amplitude_shift_db"] for row in selected])
                    ),
                    "median_amplitude_shift_z": float(
                        np.median([row["amplitude_shift_z"] for row in selected])
                    ),
                    "median_amplitude_abs_response_z": float(
                        np.median([row["amplitude_abs_response_z"] for row in selected])
                    ),
                    "median_differential_energy_db": float(
                        np.median([row["differential_energy_db"] for row in selected])
                    ),
                    "median_complex_displacement_norm": float(
                        np.median(
                            [row["complex_displacement_norm"] for row in selected]
                        )
                    ),
                }
            )

    vectors = np.vstack(run_vectors)
    run_similarity = np.asarray(
        [
            [safe_spearman(vectors[row], vectors[col]) for col in range(len(vectors))]
            for row in range(len(vectors))
        ]
    )
    similarity_rows: list[dict[str, Any]] = []
    for row, left in enumerate(run_keys):
        for col, right in enumerate(run_keys):
            if col <= row:
                continue
            similarity_rows.append(
                {
                    "left_motion": left[0],
                    "left_dataset": left[1],
                    "left_run": left[2],
                    "right_motion": right[0],
                    "right_dataset": right[1],
                    "right_run": right[2],
                    "same_action_condition": left[0] == right[0],
                    "spearman_r": float(run_similarity[row, col]),
                }
            )

    motions = list(dict.fromkeys(key[0] for key in run_keys))
    action_vectors = []
    for motion in motions:
        indices = [index for index, key in enumerate(run_keys) if key[0] == motion]
        action_vectors.append(np.median(vectors[indices], axis=0))
    action_similarity = np.asarray(
        [
            [
                safe_spearman(action_vectors[row], action_vectors[col])
                for col in range(len(motions))
            ]
            for row in range(len(motions))
        ]
    )
    action_similarity_rows = [
        {
            "left_action": motions[row],
            "right_action": motions[col],
            "spearman_r": float(action_similarity[row, col]),
        }
        for row in range(len(motions))
        for col in range(row + 1, len(motions))
    ]

    if (
        not np.isfinite(run_similarity).all()
        or not np.isfinite(action_similarity).all()
    ):
        warn(
            "UNDEFINED_CORRELATION: constant fingerprints retain NaN correlations in CSV; summary statistics are null"
        )
    within = {}
    for motion in motions:
        values = [
            r["spearman_r"]
            for r in similarity_rows
            if r["same_action_condition"] and r["left_motion"] == motion
        ]
        valid = bool(values) and bool(np.isfinite(values).all())
        within[motion] = {
            "pair_count": len(values),
            "median_spearman_r": float(np.median(values)) if valid else None,
            "min_spearman_r": float(np.min(values)) if valid else None,
            "max_spearman_r": float(np.max(values)) if valid else None,
        }
    return {
        "run_fingerprint_rows": run_fingerprint_rows,
        "similarity_rows": similarity_rows,
        "action_similarity_rows": action_similarity_rows,
        "run_keys": run_keys,
        "motions": motions,
        "run_similarity": run_similarity,
        "action_similarity": action_similarity,
        "within_condition_repeatability": within,
    }


def plot_run_heatmaps(run_fingerprint_rows: list[dict[str, Any]], output: Path) -> Path:
    run_order: list[tuple[str, str, str]] = []
    for row in run_fingerprint_rows:
        key = (row["motion"], row["dataset"], row["run"])
        if key not in run_order:
            run_order.append(key)
    amplitude = []
    energy = []
    labels = []
    for motion, dataset, run in run_order:
        rows = [
            row
            for row in run_fingerprint_rows
            if row["dataset"] == dataset and row["run"] == run
        ]
        lookup = {(row["state"], row["pair"]): row for row in rows}
        amplitude.append(
            [float(lookup[key]["median_amplitude_shift_z"]) for key in OBSERVATIONS]
        )
        energy.append(
            [
                float(lookup[key]["median_differential_energy_db"])
                for key in OBSERVATIONS
            ]
        )
        labels.append(f"{motion} {run}")
    amplitude_array = np.asarray(amplitude)
    energy_array = np.asarray(energy)
    amp_limit = max(1.0, float(np.nanpercentile(np.abs(amplitude_array), 95)))
    energy_limit = max(1.0, float(np.nanpercentile(np.abs(energy_array), 95)))
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    images = [
        axes[0].imshow(
            amplitude_array,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-amp_limit,
            vmax=amp_limit,
        ),
        axes[1].imshow(
            energy_array,
            aspect="auto",
            cmap="PuOr_r",
            vmin=-energy_limit,
            vmax=energy_limit,
        ),
    ]
    for axis, image, title in zip(
        axes,
        images,
        (
            "Median signed amplitude response (z)",
            "Median differential-energy response (dB)",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(OBS_LABELS)), OBS_LABELS, rotation=90, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    axes[0].set_yticks(range(len(labels)), labels)
    figure.suptitle("Run-level median spatial fingerprints", fontsize=13)
    path = output / "run_spatial_fingerprints.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_similarity(
    run_keys: list[tuple[str, str, str]],
    run_similarity: np.ndarray,
    motions: list[str],
    action_similarity: np.ndarray,
    output: Path,
) -> Path:
    run_labels = [f"{motion}\n{run}" for motion, _dataset, run in run_keys]
    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 0.75]}
    )
    image0 = axes[0].imshow(run_similarity, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(run_labels)), run_labels, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(run_labels)), run_labels, fontsize=8)
    axes[0].set_title("Run fingerprint Spearman correlation")
    for row in range(len(run_labels)):
        for col in range(len(run_labels)):
            axes[0].text(
                col,
                row,
                f"{run_similarity[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)

    image1 = axes[1].imshow(action_similarity, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(motions)), motions, rotation=45, ha="right")
    axes[1].set_yticks(range(len(motions)), motions)
    axes[1].set_title("Condition-median similarity")
    for row in range(len(motions)):
        for col in range(len(motions)):
            axes[1].text(
                col,
                row,
                f"{action_similarity[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(image1, ax=axes[1], fraction=0.075, pad=0.04)
    figure.suptitle(
        "Spatial fingerprint repeatability and cross-condition similarity", fontsize=13
    )
    path = output / "fingerprint_similarity.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
