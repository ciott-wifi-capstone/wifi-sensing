from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101';sys.path.insert(0,str(S))
from common import engine,GEO,MAP
from music_baseline import steering,spectrum,ANG
c0=299792458.;keys=[0,12,21,30,3,15,24,33]
rows=[];fusedrows=[];checks=[];phases=[];sources=[]
wtab=pd.read_csv(S/'results/noncoherent_fusion.csv',dtype={'index':str})
start=time.time()
with threadpool_limits(limits=1):
 rat=engine('HT40');f=rat.fc+rat.F
 for key in keys:
  info=json.loads((S/'results'/f'{key}_complete.json').read_text(encoding='utf-8'));stage=info['stage'];save={};score={};weights={}
  for state,ants in MAP.items():
   path=S/'packets'/f'{key}_{state}.npz';z=np.load(path);x=z['out_C'];ys=np.array([GEO['array_y_local_m'][a-1] for a in ants]);d=np.sqrt(3.65**2+ys**2);D=np.exp(2j*np.pi*d[:,None]*f[None,:]/c0);xc=x*D[None,:,:]
   a,tau,period=steering(rat,state);D0=D[:,rat.windows[0]].reshape(-1)
   cov0=rat.covariance(x);covnaive=rat.covariance(xc)
   # Register the known frequency-dependent rephasing before frequency smoothing.
   xr=rat.unit(xc);covconsistent=np.zeros_like(cov0)
   for w in rat.windows:
    zw=xr[:,:,w].reshape(len(x),-1)*(D0/D[:,w].reshape(-1))[None,:]
    covconsistent+=zw.T@zw.conj()/len(x)/len(rat.windows)
   direct=D0[:,None]*cov0*D0.conj()[None,:]
   p0=spectrum(cov0,a,3,len(ANG),len(tau));pn=spectrum(covnaive,a,3,len(ANG),len(tau));pc=spectrum(covconsistent,D0[:,None]*a,3,len(ANG),len(tau))
   checks.append(dict(index=key,state=state,covariance_relative_error=float(np.linalg.norm(covconsistent-direct)/np.linalg.norm(direct)),spectrum_max_difference=float(abs(pc-p0).max()),legacy_C_covariance_relative_difference=float(np.linalg.norm(cov0-z['cov_C'])/np.linalg.norm(cov0))))
   sources.append(dict(index=key,state=state,cache=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_source=info['source'],raw_sha256=info['raw_sha256'],packets=len(x)))
   if key==0:
    for mi,ant in enumerate(ants):
     for ki in range(30):phases.append(dict(state=state,antenna=ant,tone_frequency_Hz=f[ki],direct_path_m=d[mi],direct_delay_ns=d[mi]/c0*1e9,removal_phase_mod_deg=float(np.angle(D[mi,ki])*180/np.pi),differential_phase_to_center_deg=float(360*f[ki]*(d[mi]-3.65)/c0)))
   for branch,p in [('baseline',p0),('geometry_only_unchanged_model',pn),('geometry_consistent_model',pc)]:
    ii,jj=np.unravel_index(p.argmax(),p.shape);rows.append(dict(index=key,stage=stage,state=state,branch=branch,angle_deg=ANG[ii],delay_parameter_ns=tau[jj],human_valid=False,hardware_calibrated=False));score[(state,branch)]=p.max(1);save[state+'_'+branch]=p
   if state.startswith('G'):
    weights[state]=float(wtab[(wtab['index']==str(key))&(wtab.state==state)&(wtab.method=='C')].iloc[0].weight)
  for branch in ['baseline','geometry_only_unchanged_model','geometry_consistent_model']:
   q=sum(w*score[(s,branch)] for s,w in weights.items())/sum(weights.values());q/=q.max();save['fused_'+branch]=q
   fusedrows.append(dict(index=key,stage=stage,branch=branch,angle_deg=ANG[q.argmax()],human_valid=False,hardware_calibrated=False))
  np.savez_compressed(R/f'spectra_{key}.npz',angle_deg=ANG,tau_ns=tau,**save)
  print('finished',key,flush=True)
pd.DataFrame(rows).to_csv(R/'state_peaks.csv',index=False);pd.DataFrame(fusedrows).to_csv(R/'fused_peaks.csv',index=False);pd.DataFrame(checks).to_csv(R/'consistency_checks.csv',index=False);pd.DataFrame(phases).to_csv(R/'direct_geometric_phase.csv',index=False);pd.DataFrame(sources).to_csv(R/'source_manifest.csv',index=False)
assert max(r['spectrum_max_difference'] for r in checks)<1e-8
summary=dict(records=8,states=48,packets=8*6*768,max_spectrum_difference=max(r['spectrum_max_difference'] for r in checks),max_covariance_error=max(r['covariance_relative_error'] for r in checks),seconds=time.time()-start,physical_human_accuracy_valid=False)
(R/'prediction_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
