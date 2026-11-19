"""Estimate respiration rates from synthetic signals, saved ratios or CSI captures."""

import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wical.static import estimate_ratio, validate_observation_arrays
from wical.sqi import OBSERVATION_ORDER
from wical.io import execution, read_csv, write_csv, write_json, warn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", type=Path, help="JSON example specification")
    mode.add_argument(
        "--waveforms",
        type=Path,
        help="Directory of original *_waveforms.npz checkpoints",
    )
    mode.add_argument(
        "--run-dir", type=Path, help="PicoScenes run with approximate host-span timing"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        v for v in (args.synthetic, args.waveforms, args.run_dir) if v is not None
    ]
    try:
        config = {
            "task": "static_estimation",
            "sg_window_sec": 2.0,
            "synthetic": args.synthetic is not None,
            "reference_used_for_selection": False,
            "raw_timing": (
                "historical_host_span_approximation" if args.run_dir else None
            ),
        }
        with execution(args.out, inputs, config) as out:
            records = []
            qualities = []
            regressions = []
            plot = None

            def process(dataset, run, state, pair, ratio, fs, time, old=None):
                nonlocal plot
                result, quality = estimate_ratio(ratio, fs, time)
                row = {
                    "dataset": dataset,
                    "run": run,
                    "state": state,
                    "pair": pair,
                    "pred_bpm": float(result["bpm"]),
                    "max_selected_bnr": float(np.max(result["selected_bnr"])),
                    "selected_count": len(result["selected"]),
                    "selected_subcarriers": json.dumps(result["selected"].tolist()),
                    "effective_fs_hz": fs,
                    "n_dwells": len(ratio),
                }
                records.append(row)
                qualities.append(
                    {
                        **{k: row[k] for k in ("dataset", "run", "state", "pair")},
                        "frozen_pred_bpm": row["pred_bpm"],
                        **{
                            k: quality[k]
                            for k in (
                                "gamma_band",
                                "gamma_agree",
                                "gamma_spi",
                                "sqi",
                                "status",
                            )
                        },
                        "failure_reasons": " | ".join(quality["failure_reasons"]),
                    }
                )
                if old is not None:
                    for name, new in [
                        ("bpm", np.asarray([result["bpm"]])),
                        ("processed_signal", result["signal"]),
                        ("acf", result["ac"]),
                        ("selected", result["selected"]),
                        ("selected_bnr", result["selected_bnr"]),
                        ("angles", result["angles"]),
                    ]:
                        ref = np.asarray(old[name])
                        equal = new.shape == ref.shape and np.allclose(
                            new, ref, atol=1e-10, rtol=1e-10
                        )
                        regressions.append(
                            {
                                "dataset": dataset,
                                "run": run,
                                "state": state,
                                "pair": pair,
                                "field": name,
                                "match": bool(equal),
                                "max_absolute_difference": (
                                    float(np.max(np.abs(new - ref)))
                                    if new.shape == ref.shape
                                    else None
                                ),
                            }
                        )
                if plot is None:
                    plot = (np.asarray(time), result["signal"], result["ac"], fs)
                print(
                    f'{dataset}/{run}/{state}/{pair}: {row["pred_bpm"]:.6f} bpm',
                    flush=True,
                )

            if args.synthetic:
                spec = json.loads(args.synthetic.read_text())
                if spec != {
                    "kind": "synthetic",
                    "fs_hz": 10.0,
                    "duration_s": 90.0,
                    "rate_bpm": 15.0,
                    "seed": 20261118,
                }:
                    raise ValueError(
                        "Unsupported synthetic configuration. Use examples/synthetic.json."
                    )
                fs = spec["fs_hz"]
                t = np.arange(int(fs * spec["duration_s"])) / fs
                rng = np.random.default_rng(spec["seed"])
                for i, (state, pair) in enumerate(OBSERVATION_ORDER):
                    ratio = np.empty((len(t), 30), complex)
                    for sc in range(30):
                        ratio[:, sc] = (
                            2
                            + 0.12
                            * np.exp(1j * (sc / 30 * np.pi))
                            * np.sin(2 * np.pi * spec["rate_bpm"] / 60 * t)
                            + 0.001 * rng.normal(size=len(t))
                        )
                    process("SYNTHETIC", "run1", state, pair, ratio, fs, t)
            elif args.waveforms:
                paths = sorted(args.waveforms.glob("*_waveforms.npz"))
                if not paths:
                    raise ValueError("No *_waveforms.npz files found")
                from wical.selection import RUN_ORDER

                expected_stems = {
                    dataset.replace(" ", "_") + "__" + run + "_waveforms"
                    for dataset, run in RUN_ORDER
                }
                actual_stems = {p.stem for p in paths}
                if actual_stems - expected_stems:
                    raise ValueError(
                        "Unexpected checkpoint/run identity; expected one of the eight accepted recordings"
                    )
                if expected_stems - actual_stems:
                    warn(
                        "PARTIAL_REPLAY missing checkpoints: "
                        + ", ".join(sorted(expected_stems - actual_stems))
                    )
                expected = {
                    "10_bpm_formal": "10 bpm formal",
                    "15_bpm_new": "15 bpm new",
                    "20_bpm_new": "20 bpm new",
                }
                for path in paths:
                    slug, run = path.stem.removesuffix("_waveforms").split("__")
                    dataset = expected[slug]
                    with np.load(path, allow_pickle=False) as arrays:
                        for state, pair in OBSERVATION_ORDER:
                            key = f'{state}__{pair.replace("/","_")}'
                            ratio, fs, t = validate_observation_arrays(arrays, key)
                            old = {
                                name: arrays[key + "__" + name]
                                for name in (
                                    "bpm",
                                    "processed_signal",
                                    "acf",
                                    "selected",
                                    "selected_bnr",
                                    "angles",
                                )
                            }
                            process(dataset, run, state, pair, ratio, fs, t, old)
            else:
                from wical.csi import read_run, state_ratio

                data = read_run(args.run_dir)
                for state, pair in OBSERVATION_ORDER:
                    num, den = [int(v.removeprefix("Rx")) for v in pair.split("/")]
                    ratio, t, fs = state_ratio(data, state, num, den)
                    process("PRIVATE", args.run_dir.name, state, pair, ratio, fs, t)
            write_csv(out / "predictions.csv", records)
            write_csv(out / "quality.csv", qualities)
            if regressions:
                write_csv(out / "regression.csv", regressions)
            match = all(r["match"] for r in regressions) if regressions else None
            if match is False:
                warn(
                    "REGRESSION_MISMATCH: outputs differ from saved values; see regression.csv"
                )
            t, sig, ac, fs = plot
            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
            axes[0].plot(t, sig)
            axes[0].set(xlabel="Dwell time (s)", ylabel="Processed ratio")
            axes[1].plot(np.arange(len(ac)) / fs, ac)
            axes[1].set(xlabel="ACF lag (s)", ylabel="Normalized ACF")
            fig.suptitle(
                "Synthetic software check"
                if args.synthetic
                else "Respiration waveform and autocorrelation"
            )
            fig.tight_layout()
            fig.savefig(out / "diagnostic.png", dpi=160)
            plt.close(fig)
            write_json(
                out / "summary.json",
                {
                    "observations": len(records),
                    "runs": len({(r["dataset"], r["run"]) for r in records}),
                    "synthetic": bool(args.synthetic),
                    "saved_estimator_outputs_match": match,
                    "accuracy_evaluated": False,
                    "raw_decode_executed": bool(args.run_dir),
                },
            )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
