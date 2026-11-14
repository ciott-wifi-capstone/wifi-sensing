from models import *
import hashlib,time
from threadpoolctl import threadpool_limits
from music_baseline import steering,spectrum,ANG
from common import distance
start=time.time();decision=json.loads((R/'development_decision.json').read_text());assert hashlib.sha256((R/'PROTOCOL_core.md').read_bytes()).hexdigest()==decision['protocol_sha256']
models={}
for mode,name in [('pi','constant'),('pi','affine'),('pi','tone'),('continuous','affine')]:
 z=np.load(R/f'template_{mode}_{name}.npz');models[mode+'_'+name]=dict(ghat=z['ghat'],anchor=z['anchor'],mode=mode,model=name)
old=pd.read_csv(S.parent/'direct_path_geometry_ablation_20261102/initial_phase_template.csv');wt=pd.read_csv(S/'results/noncoherent_fusion.csv',dtype={'index':str});rows=[];fus=[];checks=[];rots=[];sources=[]
branches=['baseline','old_pooled_constant','pi_only','pi_constant','pi_affine','pi_tone','continuous_affine']
with threadpool_limits(limits=1):
 for key in DEV+EVAL:
  xs,u=load(key);stage='development' if key in DEV else 'evaluation';out={};pw={};rotby={}
  for tag,t in models.items():
   _,rot,diag=predict_phase(u,t);rotby[tag]=rot
   checks.extend([dict(stage=stage,**r) for r in residuals(key,u,t,tag)])
   for j in range(2):rots.append(dict(record=key,stage=stage,model=tag,chain=j+2,phase_change_deg=float(diag['phase_deg'][j]),chosen_change_deg=float(diag['chosen_deg'][j]),coherence=float(diag['coherence'][j]),choice_margin=float(diag['binary_margin'][j])))
  for si,state in enumerate(ST):
   x=xs[si];a,tau,_=steering(rat:=engine('HT40'),state)
   for branch in branches:
    if branch=='baseline':g=np.ones((3,30),complex)
    elif branch=='old_pooled_constant':
     phi=old[old.state==state].sort_values('chain').phase_candidate_deg.to_numpy()*np.pi/180;g=np.exp(1j*np.r_[0,phi])[:,None]*np.ones((1,30))
    elif branch=='pi_only':g=np.r_[1,rotby['pi_tone']][:,None]*np.ones((1,30))
    else:g=np.vstack([np.ones((1,30)),models[branch]['ghat'][si]*rotby[branch][:,None]])
    xc=x/g[None,:,:];p=spectrum(rat.covariance(xc),a,3,len(ANG),len(tau));out[state+'_'+branch]=p;ii,jj=np.unravel_index(p.argmax(),p.shape);rows.append(dict(record=key,stage=stage,state=state,model=branch,angle_deg=ANG[ii],delay_ns=tau[jj],human_valid=False))
   if state.startswith('G'):pw[state]=float(wt[(wt['index']==str(key))&(wt.state==state)&(wt.method=='C')].iloc[0].weight)
   path=S/'packets'/f'{key}_{state}.npz';sources.append(dict(record=key,state=state,cache=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),packets=len(x)))
  for branch in branches:
   q=sum(w*out[s+'_'+branch].max(1) for s,w in pw.items())/sum(pw.values());q/=q.max();out['fused_'+branch]=q;peak=float(ANG[q.argmax()]);fus.append(dict(record=key,stage=stage,model=branch,angle_deg=peak,human_valid=False,physical_angle_calibrated=False))
  np.savez_compressed(R/f'spectra_{key}.npz',angle_deg=ANG,tau_ns=tau,**out)
  print('completed',key,flush=True)
for name,data in [('spectral_peaks',rows),('fused_predictions',fus),('phase_prediction_checks',checks),('record_rotations',rots),('source_manifest',sources)]:pd.DataFrame(data).to_csv(R/f'{name}.csv',index=False)
s=pd.DataFrame(checks).query('stage=="evaluation"').groupby('model').agg(median_block_median_deg=('median_deg','median'),median_block_p90_deg=('p90_deg','median'),blocks_with_p90_le15=('p90_deg',lambda v: int((v<=15).sum())),blocks=('p90_deg','size'));s.to_csv(R/'evaluation_phase_summary.csv');print(s.to_string());print('seconds',time.time()-start)
