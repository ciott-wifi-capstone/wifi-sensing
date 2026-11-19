"""Read inputs, validate paths and save analysis outputs and execution details."""

from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    path = Path(path)
    if path.stat().st_size <= 16 * 1024 * 1024:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: missing or duplicate CSV columns")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: empty CSV")
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"{path}: malformed CSV row")
        return rows


def write_csv(path, rows, fields=None):
    if fields is None:
        if not rows:
            raise ValueError("Empty output requires explicit column names")
        fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: boolean is not a numeric measurement")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name}: NaN/Inf is not permitted")
    return number


def check_time(time, length, label="time_s"):
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) != length or length < 2:
        raise ValueError(
            f"{label}: expected one timestamp per sample and at least two samples"
        )
    if not np.isfinite(time).all() or not (np.diff(time) > 0).all():
        raise ValueError(f"{label}: timestamps must be finite and strictly increasing")
    return time


def warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def fresh_output(output, inputs):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    if "01_原始数据" in output.parts:
        raise ValueError("Writing inside the original-data directory is prohibited")
    for protected in [
        ROOT / p
        for p in (
            "src",
            "scripts",
            "configs",
            "examples",
            "results",
            "docs",
            "history",
            "tests",
        )
    ]:
        if output == protected or output.is_relative_to(protected):
            raise ValueError(
                f"Output must not be inside package inputs/source: {protected}"
            )
    if output == ROOT:
        raise ValueError("Output must not replace the project")
    for value in inputs:
        p = Path(value).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")
        if (
            output == p
            or p.is_relative_to(output)
            or (p.is_dir() and output.is_relative_to(p))
        ):
            raise ValueError("Output overlaps input; use a separate fresh directory")
    return output


@contextmanager
def execution(output, inputs, config):
    output = fresh_output(output, inputs)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    manifest = []
    for value in inputs:
        p = Path(value).resolve()
        paths = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in paths:
            if f.is_file():
                manifest.append(
                    {"path": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)}
                )
    write_json(output / "input_manifest.json", manifest)
    provenance = output / "code"
    provenance.mkdir()
    for folder in ("src", "scripts", "configs"):
        shutil.copytree(
            ROOT / folder,
            provenance / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    import scipy, matplotlib

    status = {
        "status": "RUNNING",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "command": sys.argv,
    }
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            yield output
        status["status"] = "PASS"
    except BaseException as exc:
        status["status"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        messages = [
            {"category": w.category.__name__, "message": str(w.message)}
            for w in captured
        ]
        write_csv(output / "warnings.csv", messages, ["category", "message"])
        status["warning_count"] = len(messages)
        write_json(output / "run.json", status)
        (output / "run.log").write_text(
            status["status"]
            + "\n"
            + status.get("error", "")
            + "\n"
            + "\n".join(w["message"] for w in messages)
            + "\n"
        )
        (output / "report.md").write_text(
            "# Run summary\n\nTask: " + str(config.get("task", "analysis")) + "\n\n"
            "Status: "
            + status["status"]
            + "\n\nWarnings: "
            + str(len(messages))
            + "\n\n"
            "Results are listed in summary.json when available. "
            "See warnings.csv for warnings and run.json for execution details. "
            "Failed runs retain partial outputs; use a new directory to retry.\n"
        )
        write_json(
            output / "output_manifest.json",
            {
                str(f.relative_to(output)): sha256(f)
                for f in sorted(output.rglob("*"))
                if f.is_file() and f.name != "output_manifest.json"
            },
        )
