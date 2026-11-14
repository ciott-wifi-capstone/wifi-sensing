"""Run the frozen New A algorithms on two new, separately initialized batches."""
import os
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
DEVELOPMENT=ROOT.parent/"new_a_paired_20261105"


def load_source(module_name, path):
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_batch(label, prior, extra):
    dest=ROOT/label
    dest.mkdir(exist_ok=True)
    prior.ROOT=dest
    prior.RAW=Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/f"New {label}"
    prior.LABEL={1:"empty_pre",2:f"{label}_occupied",3:"empty_post"}
    # The dependency writes only under this instance's ROOT. Do not call its legacy reporting step.
    extra.p=prior
    extra.b=prior.base
    extra.ROOT=dest
    samples,deep=prior.read_all()
    prior.changes(samples,deep)
    templates=prior.fit_r1_reference(samples)
    angular,peaks=prior.raw_spectra(samples,templates)
    prior.fixed_branch(samples,angular,peaks)
    prior.whitened_search(samples)
    extra.empirical_model(samples)
    extra.joint_check(samples)
    outputs=[v for v in sorted(dest.iterdir()) if v.suffix in {".csv",".npz"}]
    return {str(v.relative_to(ROOT)):prior.base.sha(v) for v in outputs}


if __name__=="__main__":
    prior=load_source("reference_paired_a",DEVELOPMENT/"analyze.py")
    extra=load_source("reference_paired_a_sensitivity",DEVELOPMENT/"sensitivity.py")
    base=prior.base
    frozen=base.read(DEVELOPMENT/"frozen_outputs.json")
    assert all(base.sha(DEVELOPMENT/k)==digest for k,digest in frozen.items())
    ver=base.read(DEVELOPMENT/"verification.json")
    for path,digest in ver["dependencies"].items():
        assert base.sha(Path(path))==digest, path
    original_artifacts={p:base.sha(p) for p in DEVELOPMENT.iterdir() if p.is_file()}
    blind={v.name:base.sha(v) for v in [ROOT/"protocol.json",Path(__file__)]}
    with threadpool_limits(limits=1):
        for point in ["B","C"]:
            blind.update(run_batch(point,prior,extra))
    assert all(base.sha(DEVELOPMENT/k)==digest for k,digest in frozen.items())
    assert all(base.sha(p)==digest for p,digest in original_artifacts.items())
    unique=pd.concat([pd.read_csv(d/"raw_runs.csv") for d in (DEVELOPMENT,ROOT/"B",ROOT/"C")])
    assert len(unique)==unique.sha256.nunique()==9
    (ROOT/"blind_registry.json").write_text(json.dumps(blind,indent=2),encoding="utf8")
    print("B and C decoded; per-batch results frozen before geometry scoring.")
