from pathlib import Path
import json,hashlib,py_compile
import numpy as np,pandas as pd
from fit_inverse import R,S,IDS,NF,KEYS,ST,MODES
O=R/'scaled'
checks=json.loads((O/'numerical_checks.json').read_text())
checks.update(json.loads((O/'prediction_checks.json').read_text())[0])
frozen=json.loads((O/'frozen_predictions.json').read_text())
assert all(hashlib.sha256((O/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
checks['frozen_prediction_files_verified']=len(frozen)
for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
m=pd.read_csv(O/'source_manifest.csv')
assert len(m)==48
assert all(hashlib.sha256(Path(r.cache).read_bytes()).hexdigest()==r.cache_sha256 for r in m.itertuples())
checks['cache_sources_verified']=len(m)
pred=pd.read_csv(O/'fused_predictions.csv');ph=pd.read_csv(O/'shared_phase_candidates.csv')
assert len(pred)==64 and len(ph)==48
assert not pred.human_verified.any() and not ph.physical_calibration_verified.any()
fits=pd.read_csv(O/'multistart_fits.csv');assert len(fits)==32 and fits.success.all()
checks['scaled_multistarts_converged']=int(fits.success.sum())
cov=np.load(O/'branch_corrected_covariances.npz');maxerr=0.
for mode in MODES:
 ps=np.load(O/f'fit_{mode}.npz')['phase_rad']
 for key in KEYS:
  for s in ST:
   c=cov[f'{key}_{s}'];d=np.repeat(np.exp(-1j*ps[IDS[s]]),NF)
   cc=c*d[:,None]*d[None,:].conj()
   maxerr=max(maxerr,float(abs(np.linalg.eigvalsh(c)-np.linalg.eigvalsh(cc)).max()))
assert maxerr<1e-10
checks['unitary_phase_covariance_eigenvalue_max_difference']=maxerr
checks['visual_review']='All three PNG figures inspected for axes, units, reference roles, legends and cropping.'
checks['physical_calibration_validated']=False
checks['human_path_validated']=False
checks['position_accuracy']='NA'
deps=[S/'common.py',S/'music_baseline.py',S/'geometry.json',S/'config.json',S/'provenance/ratnam/core.py',S/'results/noncoherent_fusion.csv']
checks['dependency_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
(R/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
paths=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_hashes.json']
(R/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in checks.items() if k!='dependency_sha256'},indent=2))
