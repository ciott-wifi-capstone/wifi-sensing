"""Decode the two new empty/occupied/empty sequences with the existing blind method."""
import os
import sys
from pathlib import Path
import json

import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "new_a_paired_20261105"
BC = ROOT.parent / "paired_bc_20261105"
CD = ROOT.parent / "real_cd_paired_20261105"
sys.path.insert(0, str(CD))
from process import digest, load, verify_hashes

INPUTS = {"B": "Real New B", "E": "New E"}


def process_batch(site, raw_name, methods, extra):
    dest = ROOT / site
    dest.mkdir(exist_ok=True)
    methods.RAW = Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop'))) / raw_name
    methods.ROOT = dest
    methods.LABEL = {1: "empty_pre", 2: f"{site}_occupied", 3: "empty_post"}
    extra.p, extra.b, extra.ROOT = methods, methods.base, dest
    samples, guarded = methods.read_all()
    methods.changes(samples, guarded)
    template = methods.fit_r1_reference(samples)
    spectra, peaks = methods.raw_spectra(samples, template)
    methods.fixed_branch(samples, spectra, peaks)
    methods.whitened_search(samples)
    extra.empirical_model(samples)
    extra.joint_check(samples)
    print(f"{site}: completed raw, empirical, background and joint spectra", flush=True)
    return {str(p.relative_to(ROOT)): digest(p) for p in dest.iterdir() if p.suffix in {".csv", ".npz"}}


def main():
    for directory, registry in [(BASE, "frozen_outputs.json"), (BC, "blind_registry.json"), (CD, "blind_registry.json")]:
        verify_hashes(directory, registry)
    history = {p: digest(p) for d in [BASE, BC, CD] for p in d.rglob("*") if p.is_file()}
    methods = load("paired_a_be", BASE / "analyze.py")
    extra = load("sensitivity_a_be", BASE / "sensitivity.py")
    logs = {p.name: digest(p) for p in [Path(__file__), ROOT / "protocol.json"]}
    with threadpool_limits(limits=1):
        for site, raw_name in INPUTS.items():
            logs.update(process_batch(site, raw_name, methods, extra))

    sources = [BASE, BC/"B", BC/"C", CD/"C", CD/"D", ROOT/"B", ROOT/"E"]
    records = pd.concat([pd.read_csv(d/"raw_runs.csv") for d in sources])
    assert len(records) == records.sha256.nunique() == 21
    assert not records.matches_old.any()
    assert all(digest(p) == value for p,value in history.items())
    (ROOT/"blind_registry.json").write_text(json.dumps(logs,indent=2),encoding="utf8")
    print(f"Frozen {len(logs)} B/E entries; {len(history)} previous files unchanged; 21 unique raw files.", flush=True)


if __name__ == "__main__":
    main()

