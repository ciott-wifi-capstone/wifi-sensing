"""Reapply the frozen paired-run analysis to Real New C/D, with isolated outputs."""
import os
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEV = ROOT.parent / "new_a_paired_20261105"
BC = ROOT.parent / "paired_bc_20261105"
SITES = ("C", "D")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hashes(directory, filename):
    records = json.loads((directory/filename).read_text(encoding="utf8"))
    assert all(digest(directory/key) == value for key,value in records.items())


def process(site, engine, supplemental):
    dest = ROOT / site
    dest.mkdir(exist_ok=True)
    engine.RAW = Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop'))) / f"Real New {site}"
    engine.ROOT = dest
    engine.LABEL = {1:"empty_pre",2:f"{site}_occupied",3:"empty_post"}
    supplemental.p = engine
    supplemental.b = engine.base
    supplemental.ROOT = dest
    samples, deeper = engine.read_all()
    engine.changes(samples, deeper)
    templates = engine.fit_r1_reference(samples)
    spectra, peaks = engine.raw_spectra(samples, templates)
    engine.fixed_branch(samples, spectra, peaks)
    engine.whitened_search(samples)
    supplemental.empirical_model(samples)
    supplemental.joint_check(samples)
    return {str(p.relative_to(ROOT)):digest(p) for p in dest.iterdir() if p.suffix in {".csv",".npz"}}


if __name__ == "__main__":
    verify_hashes(DEV,"frozen_outputs.json")
    verify_hashes(BC,"blind_registry.json")
    prior_files = {p:digest(p) for root in (DEV, BC) for p in root.rglob("*") if p.is_file()}
    engine = load("new_a_methods_for_real_cd", DEV/"analyze.py")
    supplemental = load("new_a_sensitivity_for_real_cd", DEV/"sensitivity.py")
    logs = {p.name:digest(p) for p in (Path(__file__), ROOT/"protocol.json")}
    with threadpool_limits(limits=1):
        for site in SITES:
            logs.update(process(site, engine, supplemental))
    assert all(digest(p) == h for p,h in prior_files.items())
    sources = [DEV, BC/"B", BC/"C", ROOT/"C", ROOT/"D"]
    runs = pd.concat([pd.read_csv(d/"raw_runs.csv") for d in sources])
    assert len(runs) == runs.sha256.nunique() == 15
    (ROOT/"blind_registry.json").write_text(json.dumps(logs,indent=2),encoding="utf8")
    print("Froze results for Real New C/D; historical files verified unchanged.")
