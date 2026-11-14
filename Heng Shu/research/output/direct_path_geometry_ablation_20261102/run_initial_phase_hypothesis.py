from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101';sys.path.insert(0,str(S))
from common import engine,GEO,MAP
from music_baseline import steering,spectrum,ANG
rat=engine('HT40');freq=rat.fc+rat.F;c0=299792458.;dev=[0,12,21,30];keys=dev+[3,15,24,33];tpl={};rows=[];peaks=[];fusion=[]
wt=pd.read_csv(S/'results/noncoherent_fusion.csv',dtype={'index':str})
with threadpool_limits(limits=1):
 for state,ants in MAP.items():
  y=np.array([GEO['array_y_local_m'][a-1] for a in ants]);d=np.sqrt(3.65**2+y*y);geom=-2*np.pi*(d[1:]-d[0])[:,None]*freq[None,:]/c0;zs=[]
  for key in dev:
   x=np.load(S/'packets'/f'{key}_{state}.npz')['x'][:64];p=x[:,1:]*x[:,:1].conj();u=p/np.maximum(abs(p),1e-30);zs.append((u*np.exp(-1j*geom)[None,:,:]).mean((0,2)))
  avg=np.mean(zs,axis=0);phi=np.angle(avg);rho=abs(avg);ok=bool(np.all(rho>=.9));tpl[state]=(np.r_[0,phi],ok)
  for i in range(2):rows.append(dict(state=state,chain=i+2,relative_chain=1,phase_candidate_deg=float(phi[i]*180/np.pi),pooled_resultant=float(rho[i]),state_screen_pass=ok,assumption='direct_path_dominant; unverified',calibration_records='0;12;21;30',hardware_valid=False))
 pd.DataFrame(rows).to_csv(R/'initial_phase_template.csv',index=False)
 for key in keys:
  stage='development' if key in dev else 'evaluation';spec={};weights={}
  for state in MAP:
   x=np.load(S/'packets'/f'{key}_{state}.npz')['out_C'];ph,ok=tpl[state];a,tau,period=steering(rat,state)
   for branch,apply in [('conditional_initial_phase',True),('gated_initial_phase',ok)]:
    xc=x*np.exp(-1j*ph)[None,:,None] if apply else x;p=spectrum(rat.covariance(xc),a,3,len(ANG),len(tau));ii,jj=np.unravel_index(p.argmax(),p.shape);spec[state+'_'+branch]=p
    peaks.append(dict(index=key,stage=stage,state=state,branch=branch,angle_deg=ANG[ii],delay_parameter_ns=tau[jj],applied=apply,screen_pass=ok,human_valid=False))
   if state.startswith('G'):weights[state]=float(wt[(wt['index']==str(key))&(wt.state==state)&(wt.method=='C')].iloc[0].weight)
  for branch in ['conditional_initial_phase','gated_initial_phase']:
   q=sum(w*spec[s+'_'+branch].max(1) for s,w in weights.items())/sum(weights.values());q/=q.max();spec['fused_'+branch]=q;fusion.append(dict(index=key,stage=stage,branch=branch,angle_deg=ANG[q.argmax()],human_valid=False))
  np.savez_compressed(R/f'initial_phase_spectra_{key}.npz',angle_deg=ANG,tau_ns=tau,**spec)
pd.DataFrame(peaks).to_csv(R/'initial_phase_state_peaks.csv',index=False);pd.DataFrame(fusion).to_csv(R/'initial_phase_fused_peaks.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(fusion).query('stage=="evaluation"').to_string(index=False))
