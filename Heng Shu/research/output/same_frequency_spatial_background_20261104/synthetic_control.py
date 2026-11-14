from spatial_model import *
from validate_occupied import clusters
def main():
 rng=np.random.default_rng(20261104)
 def noise(n):return (rng.normal(size=(n,3,30))+1j*rng.normal(size=(n,3,30)))/np.sqrt(2)
 stable={};empty_count=0;states=[]
 for state in ['G1','G2','G3','G4']:
  source=noise(2048);dev=noise(2048);empty=noise(2048)
  m,td,ts=model_from_source(source,dev,state)
  e,ep,_=assess(empty,m,td,ts);empty_count+=e['stable_candidates']
  # A known synthetic plane-wave component with equal per-antenna signal/noise power.
  a=steering(state)[:,:,np.where(ANG==21)[0][0]].T
  alpha=(rng.normal(size=2048)+1j*rng.normal(size=2048))/np.sqrt(2)
  x=noise(2048)+alpha[:,None,None]*a[None]
  stat,pp,_=assess(x,m,td,ts);stable[state]=pp
  states.append(dict(state=state,synthetic_direction_deg=21,stable_peaks=[p['angle_parameter_deg'] for p in pp if p['stable']],empty_stable_peaks=e['stable_candidates']))
 cc=clusters(stable);assert any(abs(c['center_deg']-21)<=9 for c in cc),cc
 result=dict(seed=20261104,packets_per_run=2048,synthetic_per_antenna_SNR_db=0,states=states,detected_clusters=cc,empty_stable_state_peaks=empty_count,scope='Ideal calibrated synthetic steering only; not evidence of physical calibration or actual sensitivity.')
 (R/'synthetic_end_to_end.json').write_text(json.dumps(result,indent=2))
 # Apply the exact G-subarray cluster rule to empty controls too.
 d=pd.read_csv(R/'empty_false_candidates.csv');groups=[]
 all_cases=pd.read_csv(R/'empty_validation.csv')[['source','target','index','mode']].drop_duplicates()
 for src,tar,idx,mode in all_cases.itertuples(index=False,name=None):
  z=d[(d.source==src)&(d.target==tar)&(d['index']==idx)&(d['mode']==mode)]
  by={s:[dict(angle_parameter_deg=r.angle_parameter_deg,score=r.score,stable=r.stable) for r in z[z.state==s].itertuples()] for s in ST}
  groups.append(dict(source=src,target=tar,index=idx,mode=mode,cross_G_clusters=len(clusters(by))))
 pd.DataFrame(groups).to_csv(R/'empty_cross_G_clusters.csv',index=False)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
