from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;PRE=R.parent/'empty_reference_static_recovery_20261024';OLD=R.parent/'ratnam_phase_timing_validation'
sys.path.insert(0,str(PRE))
from common import rat,dec,invariants,distance,sha
def main():
 (R/'cache').mkdir(exist_ok=True)
 mm=pd.read_csv(PRE/'empty_manifest.csv');rows=[];sources=[];stats={}
 for row in mm.itertuples(index=False):
  p=Path(row.path);assert sha(p)==row.sha256
  meta=dec.read(p.parent/'run.json');exists=(PRE/'cache'/f'{row.index:03d}_G1.npz').exists()
  if not exists:x,host,nic=dec.decode(p)
  role={'run1':'background_train','run2':'threshold_development','run3':'complete_run_validation'}[row.run]
  for state in rat.MAP:
   old=OLD/'packets'/f'{row.index:03d}_{state}.npz';v=np.load(old);ids=v['packet_id']
   if exists:
    z=np.load(PRE/'cache'/f'{row.index:03d}_{state}.npz');assert np.array_equal(ids,z['packet_id']);raw=z['x']
   else:raw=rat.unit(x[ids].astype(complex))
   inv=invariants(raw);err=distance(inv['covariance'],v['cov_A']);assert err<1e-10
   np.savez_compressed(R/'cache'/f'{row.index:03d}_{state}.npz',x=raw,packet_id=ids,**inv)
   stats[(row.capture_id,row.run,state)]=inv
   sources.append(dict(index=row.index,capture=row.capture_id,run=row.run,state=state,role=role,packets=len(raw),requested_seconds=meta['requested_seconds'],raw_path=str(p),raw_sha256=row.sha256,packet_source=str(old),packet_source_sha256=sha(old),old_covariance_difference=err))
  print('audited',row.capture_id,row.run,flush=True)
 for cap in ['E1','E2','E3','E4']:
  for run in ['run2','run3']:
   for s in rat.MAP:
    a=stats[(cap,'run1',s)];b=stats[(cap,run,s)]
    mask=(a['pair_coherence']>=.5)&(b['pair_coherence']>=.5)
    ph=abs(np.angle(np.exp(1j*(a['pair_angle']-b['pair_angle']))))*180/np.pi
    rows.append(dict(capture=cap,comparison_run=run,state=s,amplitude_distance=distance(a['amplitude'],b['amplitude']),spatial_distance=distance(a['spatial'],b['spatial']),covariance_distance=distance(a['covariance'],b['covariance']),phase_median_deg=float(np.median(ph[mask])) if mask.any() else np.nan,phase_support=mask.mean()))
 pd.DataFrame(sources).to_csv(R/'source_manifest.csv',index=False)
 df=pd.DataFrame(rows);df.to_csv(R/'within_capture_invariants.csv',index=False)
 sm=df.groupby(['capture','comparison_run'])[['amplitude_distance','spatial_distance','covariance_distance','phase_median_deg']].median()
 sm.to_csv(R/'within_capture_summary.csv');print(sm.to_string())
 (R/'audit_checks.json').write_text(json.dumps(dict(raw_files_verified=12,state_caches=72,all_covariances_match=True,split_by_complete_run=True),indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
