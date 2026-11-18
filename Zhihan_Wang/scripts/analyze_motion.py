"""Analyze CSI-change candidates and spatial responses from saved ratios or a capture."""

import _bootstrap
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wical import motion as m
from wical.io import (
    execution,
    read_csv,
    write_csv,
    write_json,
    check_time,
    finite,
    warn,
)


def observation_from_ratio(state, pair, time, ratio):
    ratio = np.asarray(ratio, dtype=complex)
    if ratio.ndim != 2 or ratio.shape[1] != 30 or not np.isfinite(ratio).all():
        raise ValueError("Motion ratio must be a finite (N, 30) array")
    time = check_time(time, len(ratio))
    gaps = np.diff(time)
    if not np.allclose(gaps, np.median(gaps), rtol=1e-6, atol=1e-9):
        warn(
            f"{state}/{pair}: nonuniform dwells; maximum gap {np.max(gaps):.9g}s. No waveform interpolation applied"
        )
    amplitude_sc = 20 * np.log10(np.maximum(np.abs(ratio), m.EPS))
    diff_sc = np.abs(np.diff(ratio, axis=0)) ** 2 / np.maximum(
        np.abs(ratio[:-1]) ** 2, m.EPS
    )
    return m.Observation(
        state,
        pair,
        time,
        ratio,
        amplitude_sc,
        np.median(amplitude_sc, axis=1),
        (time[:-1] + time[1:]) / 2,
        np.median(diff_sc, axis=1),
        diff_sc,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--archive",
        type=Path,
        help="Private spatial_fingerprint directory with assets and run_inventory.csv",
    )
    group.add_argument(
        "--run-dir", type=Path, help="Private PicoScenes run with run.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.archive:
            inventory_path = args.archive / "run_inventory.csv"
            inventory = read_csv(inventory_path)
            identities = [(r["motion"], r["run"]) for r in inventory]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate motion/run in inventory")
            paths = [
                args.archive
                / "assets"
                / f"{r['motion'].lower()}__{r['run']}_complex_assets.npz"
                for r in inventory
            ]
            inputs = [inventory_path, *paths]
        else:
            inputs = [args.run_dir]
            inventory = [
                {"dataset": "PRIVATE", "motion": "UNLABELLED", "run": args.run_dir.name}
            ]
            paths = [None]
        config = {
            "task": "exploratory_motion",
            "independent_action_ground_truth": False,
            **{
                name: getattr(m, name)
                for name in (
                    "BIN_WIDTH_S",
                    "EVENT_MIN_SEPARATION_S",
                    "EVENT_HEIGHT_Z",
                    "EVENT_PROMINENCE_Z",
                    "EVENT_SUPPORT_Z",
                    "EVENT_MIN_SUPPORT_OBSERVATIONS",
                    "BASELINE_START_S",
                    "BASELINE_END_S",
                    "RESPONSE_START_S",
                    "RESPONSE_END_S",
                    "MIN_WINDOW_DWELLS",
                    "AMPLITUDE_SCALE_FLOOR_DB",
                )
            },
        }
        with execution(args.out, inputs, config) as out:
            candidates = []
            features = []
            regressions = []
            run_summaries = []
            plot_rows = []
            fingerprint_runs = []
            warn(
                "No synchronized motion labels supplied; detected events are labelled CSI_CANDIDATE"
            )
            for record, path in zip(inventory, paths, strict=True):
                if path is not None:
                    duration = finite(record["duration_s"], "duration_s")
                    if duration <= 0:
                        raise ValueError("duration_s must be positive")
                    with np.load(path, allow_pickle=False) as saved:
                        observations = {}
                        for i, (state, pair) in enumerate(m.OBSERVATIONS):
                            prefix = f"obs_{i:02d}"
                            observations[(state, pair)] = observation_from_ratio(
                                state,
                                pair,
                                saved[prefix + "_time_s"],
                                saved[prefix + "_ratio"],
                            )
                        expected = np.asarray(saved["candidate_event_time_s"])
                        old_grid = np.asarray(saved["detection_grid_s"])
                        old_score = np.asarray(saved["detection_consensus_score"])
                else:
                    from wical.csi import read_run

                    data = read_run(args.run_dir, crop_motion=True)
                    duration = (data["run_end_ns"] - data["run_start_ns"]) / 1e9
                    observations = {}
                    for state, pair in m.OBSERVATIONS:
                        num, den = [int(v.removeprefix("Rx")) for v in pair.split("/")]
                        observations[(state, pair)] = m.build_observation(
                            data, state, pair, num, den
                        )
                    expected = None
                if duration <= m.BIN_WIDTH_S:
                    raise ValueError("Motion duration too short for detection grid")
                for ob in observations.values():
                    if ob.time_s[0] < 0 or ob.time_s[-1] >= duration:
                        raise ValueError(
                            "Observation times fall outside the declared run duration"
                        )
                counts = {len(ob.time_s) for ob in observations.values()}
                if len(counts) > 1:
                    warn(
                        "UNEQUAL_DWELL_LENGTHS across sequential observations: "
                        + str(sorted(counts))
                    )
                grid, score, support, matrix = m.build_detection_score(
                    observations, duration
                )
                events = m.detect_candidate_events(grid, score, support, duration)
                event_times = np.asarray([e["event_time_s"] for e in events])
                if expected is not None:
                    regressions.append(
                        {
                            "motion": record["motion"],
                            "run": record["run"],
                            "candidate_times_match": bool(
                                np.array_equal(event_times, expected)
                            ),
                            "grid_match": bool(np.array_equal(grid, old_grid)),
                            "score_match": bool(
                                np.allclose(score, old_score, rtol=1e-12, atol=1e-12)
                            ),
                        }
                    )
                if not events:
                    warn(
                        f"{record['motion']}/{record['run']}: no events passed the detection thresholds"
                    )
                for e in events:
                    candidates.append(
                        {
                            "dataset": record["dataset"],
                            "motion": record["motion"],
                            "run": record["run"],
                            "event_id": f"E{e['event_index']}",
                            "event_time_s": e["event_time_s"],
                            "event_status": e["event_status"],
                        }
                    )
                    for ob in observations.values():
                        row, _ = m.event_features(
                            record["dataset"], record["motion"], record["run"], e, ob
                        )
                        features.append(row)
                run_summaries.append(
                    {
                        "motion": record["motion"],
                        "run": record["run"],
                        "candidate_count": len(events),
                    }
                )
                plot_rows.append((record, grid, score, event_times))
                fingerprint_runs.append(
                    {
                        "dataset": record["dataset"],
                        "motion": record["motion"],
                        "run": record["run"],
                        "events": events,
                    }
                )
                print(
                    record["motion"],
                    record["run"],
                    len(events),
                    "CSI candidates",
                    flush=True,
                )
            write_csv(
                out / "candidate_events.csv",
                candidates,
                [
                    "dataset",
                    "motion",
                    "run",
                    "event_id",
                    "event_time_s",
                    "event_status",
                ],
            )
            if features:
                write_csv(out / "event_observation_features.csv", features)
            if regressions:
                write_csv(out / "regression.csv", regressions)
            fingerprints = m.build_run_fingerprints(features, fingerprint_runs)
            write_csv(
                out / "run_spatial_fingerprints.csv",
                fingerprints["run_fingerprint_rows"],
                m.FINGERPRINT_FIELDS,
            )
            write_csv(
                out / "run_pair_similarity.csv",
                fingerprints["similarity_rows"],
                m.RUN_SIMILARITY_FIELDS,
            )
            write_csv(
                out / "action_similarity.csv",
                fingerprints["action_similarity_rows"],
                m.ACTION_SIMILARITY_FIELDS,
            )
            if fingerprints["run_keys"]:
                m.plot_run_heatmaps(fingerprints["run_fingerprint_rows"], out)
                m.plot_similarity(
                    fingerprints["run_keys"],
                    fingerprints["run_similarity"],
                    fingerprints["motions"],
                    fingerprints["action_similarity"],
                    out,
                )

            fig, axes = plt.subplots(
                len(plot_rows), 1, figsize=(9, 2.1 * len(plot_rows)), squeeze=False
            )
            for ax, (r, t, s, ev) in zip(axes[:, 0], plot_rows, strict=True):
                ax.plot(t, s, lw=1)
                ax.axhline(m.EVENT_HEIGHT_Z, color="gray", ls="--")
                for value in ev:
                    ax.axvline(value, color="#bd4f21", alpha=0.6)
                ax.set(
                    title=f"{r['motion']} / {r['run']} - CSI candidates",
                    xlabel="Run-relative time (s)",
                    ylabel="Score",
                )
            fig.tight_layout()
            fig.savefig(out / "candidate_timelines.png", dpi=150)
            plt.close(fig)
            write_json(
                out / "summary.json",
                {
                    "runs": run_summaries,
                    "candidate_events": len(candidates),
                    "event_observation_rows": len(features),
                    "run_fingerprint_rows": len(fingerprints["run_fingerprint_rows"]),
                    "run_similarity_pairs": len(fingerprints["similarity_rows"]),
                    "condition_similarity_pairs": len(
                        fingerprints["action_similarity_rows"]
                    ),
                    "within_condition_repeatability": fingerprints[
                        "within_condition_repeatability"
                    ],
                    "ground_truth_action_markers": False,
                    "respiration_accuracy_evaluated": False,
                    "regression_match": (
                        all(
                            all(
                                r[k]
                                for k in (
                                    "candidate_times_match",
                                    "grid_match",
                                    "score_match",
                                )
                            )
                            for r in regressions
                        )
                        if regressions
                        else None
                    ),
                },
            )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
