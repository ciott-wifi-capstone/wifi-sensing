"""Replay the A/C/D/E package with fixed algorithms and no target coordinates."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import csv
import numpy as np
import replay as r

class Folders:
    def __init__(self, roots):
        self.roots = roots
        self.names = [f'raw/{root.name}/{p.relative_to(root).as_posix()}' for root in roots.values() for p in root.rglob('*') if p.is_file()]
    def namelist(self): return self.names
    def read(self, name):
        _, session, rel = name.split('/', 2)
        return (self.roots[session[-1]]/rel).read_bytes()
    def load(self, name): return json.loads(self.read(name))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sage-core',type=Path,help='Optional permission-cleared external Widar2 Python file')
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()): ap.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    expected=['New A','Real New C','Real New D','New E']
    missing=[name for name in expected if not (args.data/name).is_dir()]
    if missing: ap.error('Missing private session directories: '+', '.join(missing)+'; see DATA_AVAILABILITY.md')
    source=Folders({p:args.data/name for p,name in {'A':'New A','C':'Real New C','D':'Real New D','E':'New E'}.items()})
    r.ROLE_EVIDENCE='A confirmed; C/D/E follow established Run1 empty, Run2 person, Run3 empty order as an assumption; logs identify run numbers only'
    policy=dict(r.POLICY,points='Original A and current C/D/E; B excluded after result inspection by user',role_evidence=r.ROLE_EVIDENCE)
    (out/'policy_before_fit.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    tables=defaultdict(list)
    core,calls,con=r.optional_core(args.sage_core)
    for point,root in source.roots.items():
        sess='raw/'+root.name
        config,plan,summary=r.audit_session(source,sess,tables)
        datasets={run:r.audit_run(source,sess,run,config,plan,summary[run],tables) for run in r.ROLES}
        r.compare_empty(point,datasets['run1'],datasets['run3'],tables)
        r.estimate_point(point,datasets,core,con,tables)
        print(point,'verified and fitted',flush=True)
    for name,rows in tables.items(): r.csvout(out/f'{name}.csv',rows)
    frozen=('policy_before_fit.json','blind_predictions.csv','spectra.csv','negative_control.csv') + (('native_sage.csv',) if core is not None else ())
    (out/'fit_provenance.json').write_text(json.dumps(dict(native_core_calls=calls,native_crosscheck=bool(core),native_core_sha256=r.sha(args.sage_core) if args.sage_core else None,
        frozen_before_geometry={n:r.sha(out/n) for n in frozen},script_sha256=r.sha(__file__)),indent=2))

if __name__=='__main__': main()
