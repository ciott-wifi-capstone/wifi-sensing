"""Compare observation selectors using the saved static result tables."""

import _bootstrap
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from wical.io import read_csv, write_csv, write_json, execution, sha256, ROOT
from wical.selection import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.tables / "observations.csv", args.tables / "quality.csv"]
    try:
        with execution(
            args.out,
            inputs,
            {
                "task": "selection_evaluation",
                "fixed": "G1 Rx1/Rx2",
                "parameters_tuned": False,
            },
        ) as out:
            rows, summary = evaluate(read_csv(inputs[0]), read_csv(inputs[1]))
            write_csv(out / "run_comparison.csv", rows)
            write_csv(out / "method_summary.csv", summary)
            # Compare recomputed metrics with the saved summary.
            reference = args.tables / "frozen_method_summary.csv"
            regression = None
            if reference.exists():
                frozen = read_csv(reference)
                regression = all(
                    np.isclose(r[k], float(old[k]), atol=1e-12, rtol=1e-12)
                    for r, old in zip(summary, frozen, strict=True)
                    for k in (
                        "mae_bpm",
                        "median_absolute_error_bpm",
                        "worst_absolute_error_bpm",
                    )
                )
                write_json(
                    out / "frozen_reference.json",
                    {"sha256": sha256(reference), "metrics_match": regression},
                )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                ["Fixed", "Max-BNR", "SQI", "Oracle"],
                [r["mae_bpm"] for r in summary],
                color=["#777777", "#2878a5", "#d67b20", "#4b8f60"],
            )
            ax.set_ylabel("Mean absolute error (bpm)")
            ax.set_title("Eight accepted recordings; Oracle is retrospective")
            fig.tight_layout()
            fig.savefig(out / "selection_error.png", dpi=160)
            plt.close(fig)
            write_json(
                out / "summary.json",
                {
                    "runs": 8,
                    "observations": 144,
                    "unit_of_evaluation": "recording",
                    "independent_participants": 1,
                    "metrics": summary,
                    "frozen_metrics_match": regression,
                    "limitation": "Recording-level reference intervals are not independently verified for every run. Motion accuracy is not evaluated.",
                },
            )
        for row in summary:
            print(
                f"{row['method']}: MAE {row['mae_bpm']:.9f} bpm ({row['n_returned']}/8)"
            )
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
