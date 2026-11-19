"""Select observations using CSI fields, then evaluate against belt reference rates."""

from __future__ import annotations
import numpy as np
from .sqi import OBSERVATION_ORDER, ORDER_INDEX, select_by_sqi
from .io import finite

RUN_ORDER = tuple(
    [("10 bpm formal", f"run{i}") for i in (1, 2, 3)]
    + [("15 bpm new", f"run{i}") for i in (1, 3)]
    + [("20 bpm new", f"run{i}") for i in (1, 2, 3)]
)
FIXED = ("G1", "Rx1/Rx2")


def key(row):
    return tuple(row[k] for k in ("dataset", "run", "state", "pair"))


def index_rows(rows):
    if not rows:
        raise ValueError("No observation rows")
    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate dataset/run/state/pair identity")
    for rid in {k[:2] for k in keys}:
        if {k[2:] for k in keys if k[:2] == rid} != set(OBSERVATION_ORDER):
            raise ValueError(f"{rid}: expected exactly six states x three Rx pairs")
    return dict(zip(keys, rows, strict=True))


def select_without_reference(observations, quality):
    # Exclude reference values and errors from the selector inputs.
    pred = [
        {
            "dataset": r["dataset"],
            "run": r["run"],
            "state": r["state"],
            "pair": r["pair"],
            "pred_bpm": finite(r["pred_bpm"], "pred_bpm"),
            "max_selected_bnr": finite(r["max_selected_bnr"], "BNR"),
        }
        for r in observations
    ]
    scores = [
        {
            **{k: r[k] for k in ("dataset", "run", "state", "pair")},
            "sqi": finite(r["sqi"], "SQI"),
            "frozen_pred_bpm": finite(r["frozen_pred_bpm"], "frozen prediction"),
        }
        for r in quality
    ]
    p_index, q_index = index_rows(pred), index_rows(scores)
    if p_index.keys() != q_index.keys():
        raise ValueError("Prediction and SQI identities differ")
    for identity, r in p_index.items():
        q = q_index[identity]
        if (
            r["pred_bpm"] <= 0
            or not 0 <= r["max_selected_bnr"] <= 1
            or not 0 <= q["sqi"] <= 1
        ):
            raise ValueError(f"{identity}: invalid rate or quality score")
        if r["pred_bpm"] != q["frozen_pred_bpm"]:
            raise ValueError(
                f"{identity}: prediction changed between quality and evaluation"
            )
    choices = {}
    for rid in sorted({k[:2] for k in p_index}):
        ordered = [p_index[rid + o] for o in OBSERVATION_ORDER]
        maximum = max(r["max_selected_bnr"] for r in ordered)
        bnr = next(
            r
            for r in ordered
            if np.isclose(r["max_selected_bnr"], maximum, atol=1e-12, rtol=1e-12)
        )
        sqi = select_by_sqi([q_index[rid + o] for o in OBSERVATION_ORDER])
        choices[rid] = {
            "fixed": FIXED,
            "max_bnr": (bnr["state"], bnr["pair"]),
            "sqi": (
                None
                if sqi["selected"] is None
                else (sqi["selected"]["state"], sqi["selected"]["pair"])
            ),
        }
    return choices


def evaluate(observations, quality, *, require_thesis_set=True):
    choices = select_without_reference(observations, quality)
    if require_thesis_set and set(choices) != set(RUN_ORDER):
        raise ValueError(
            "Expected the eight accepted thesis recordings; 15 bpm run2 remains excluded"
        )
    indexed = index_rows(observations)
    rows = []
    for rid in RUN_ORDER if require_thesis_set else sorted(choices):
        reference = None
        errors = {}
        for o in OBSERVATION_ORDER:
            r = indexed[rid + o]
            if r["gt_status"] != "accepted":
                raise ValueError(f"{rid}: reference is not accepted")
            gt = finite(r["gt_rr_bpm"], "gt_rr_bpm")
            if gt <= 0 or (reference is not None and gt != reference):
                raise ValueError(f"{rid}: inconsistent reference rates")
            reference = gt
            error = abs(finite(r["pred_bpm"], "pred_bpm") - gt)
            if not np.isclose(
                error,
                finite(r["abs_error_vs_gt"], "stored absolute error"),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(
                    f"{rid+o}: stored absolute error does not match prediction/reference"
                )
            errors[o] = error
        minimum = min(errors.values())
        oracle = next(
            o
            for o in OBSERVATION_ORDER
            if np.isclose(errors[o], minimum, atol=1e-12, rtol=1e-12)
        )
        row = {"dataset": rid[0], "run": rid[1], "gt_rr_bpm": reference}
        for method, o in {**choices[rid], "oracle": oracle}.items():
            row[method + "_observation"] = None if o is None else f"{o[0]} · {o[1]}"
            row[method + "_pred_bpm"] = (
                None if o is None else float(indexed[rid + o]["pred_bpm"])
            )
            row[method + "_abs_error_bpm"] = None if o is None else errors[o]
        rows.append(row)
    summary = []
    for method in ("fixed", "max_bnr", "sqi", "oracle"):
        values = [
            r[method + "_abs_error_bpm"]
            for r in rows
            if r[method + "_abs_error_bpm"] is not None
        ]
        summary.append(
            {
                "method": method,
                "uses_gt_for_selection": method == "oracle",
                "n_runs": len(rows),
                "n_returned": len(values),
                "mae_bpm": float(np.mean(values)) if values else None,
                "median_absolute_error_bpm": (
                    float(np.median(values)) if values else None
                ),
                "worst_absolute_error_bpm": max(values) if values else None,
            }
        )
    return rows, summary
