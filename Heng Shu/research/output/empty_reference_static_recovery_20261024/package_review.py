"""Package verified deliverables; preserve source data and previous outputs."""
import hashlib,json,zipfile
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R/'results/completion_audit.json').read_text(encoding='utf-8'))['all_pass']
files=[]
for p in sorted(R.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='artifact_manifest.csv':continue
    # Per-page PNGs can be regenerated; retain contact sheets and reviewed PDF hashes.
    if p.parent.parent.name=='qa' and p.name.startswith('page-'):continue
    files.append(p)
rows=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
manifest=R/'artifact_manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False);files.append(manifest)
out=R.parent/'empty_reference_static_recovery_20261024_review.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in files:z.write(p,p.relative_to(R).as_posix())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for r in rows:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
summary=dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),members=len(files),all_member_hashes_verified=True,raw_inputs_included=False,notes='Requires original raw input paths and prior Ratnam source outputs for full from-raw reproduction. Six empty parsed caches and all new parameters are included. Per-page QA PNGs can be regenerated.')
(R.parent/'empty_reference_static_recovery_20261024_review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
