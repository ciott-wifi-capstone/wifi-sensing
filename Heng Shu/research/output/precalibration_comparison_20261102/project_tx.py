from models import *
from threadpoolctl import threadpool_limits
from music_baseline import steering,ANG
from scipy.linalg import null_space
import hashlib
rat=engine('HT40');z=np.load(R/'template_pi_tone.npz');t=dict(ghat=z['ghat'],anchor=z['anchor'],mode='pi',model='tone');wt=pd.read_csv(S/'results/noncoherent_fusion.csv',dtype={'index':str});rows=[];fused=[];checks=[]
with threadpool_limits(limits=1):
 for key in DEV+EVAL:
  xs,u=load(key);_,rot,_=predict_phase(u,t);stage='development' if key in DEV else 'evaluation';out={};valids={};weights={}
  for si,state in enumerate(ST):
   y=np.array([GEO['array_y_local_m'][a-1] for a in MAP[state]]);d=np.sqrt(3.65**2+y*y);los=np.exp(-2j*np.pi*rat.fc*(d-d[0])/299792458.);B=null_space(los.conj()[None,:]);W=B.conj().T;full_a,tau,_=steering(rat,state);nf=15;a=np.einsum('am,mkg->akg',W,full_a.reshape(3,nf,-1)).reshape(2*nf,-1);power=np.sum(abs(a)**2,axis=0)/np.sum(abs(full_a)**2,axis=0);valid=(power>=.1).reshape(len(ANG),len(tau));an=a/np.maximum(np.linalg.norm(a,axis=0,keepdims=True),1e-15);valids[state]=valid.all(1)
   exact_los=np.exp(-2j*np.pi*(d-d[0])[:,None]*F[None,:]/299792458.);leak=float(np.max(np.sum(abs(W@exact_los)**2,axis=0)/3));checks.append(dict(record=key,state=state,semiunitary_error=float(np.linalg.norm(W@W.conj().T-np.eye(2))),carrier_direct_null=float(np.linalg.norm(W@los)),max_fullband_direct_power_leakage=leak,observable_angles=int(valids[state].sum()),noise_model='semiunitary projection; identity noise remains identity in retained subspace'))
   for branch in ['baseline_tx_projection','pi_tone_tx_projection']:
    x=xs[si]
    if branch=='pi_tone_tx_projection':x=x/np.vstack([np.ones((1,30)),t['ghat'][si]*rot[:,None]])[None,:,:]
    x=rat.unit(x);xp=np.einsum('am,nmk->nak',W,x);cov=sum((v:=xp[:,:,w].reshape(len(x),-1)).T@v.conj()/len(x) for w in rat.windows)/16;_,ev=np.linalg.eigh(cov);den=np.sum(abs(ev[:,:-2].conj().T@an)**2,axis=0);p=(1/np.maximum(den,1e-12)).reshape(len(ANG),len(tau));p[~valid]=0;p/=p.max();out[state+'_'+branch]=p;ii,jj=np.unravel_index(p.argmax(),p.shape);rows.append(dict(record=key,stage=stage,state=state,model=branch,angle_deg=ANG[ii],delay_ns=tau[jj],human_valid=False))
   if state.startswith('G'):weights[state]=float(wt[(wt['index']==str(key))&(wt.state==state)&(wt.method=='C')].iloc[0].weight)
  common=np.logical_and.reduce([valids[s] for s in weights]);out['observable_angle']=common
  for branch in ['baseline_tx_projection','pi_tone_tx_projection']:
   q=sum(w*out[s+'_'+branch].max(1) for s,w in weights.items())/sum(weights.values());q[~common]=0;q/=q.max();out['fused_'+branch]=q;fused.append(dict(record=key,stage=stage,model=branch,angle_deg=ANG[q.argmax()],human_valid=False,physical_angle_calibrated=False))
  np.savez_compressed(R/f'projected_spectra_{key}.npz',angle_deg=ANG,tau_ns=tau,**out)
pd.DataFrame(rows).to_csv(R/'projection_peaks.csv',index=False);pd.DataFrame(fused).to_csv(R/'projection_fused_predictions.csv',index=False);pd.DataFrame(checks).to_csv(R/'projection_checks.csv',index=False);print(pd.DataFrame(fused).query('stage=="evaluation"').to_string(index=False));print('max leakage',max(x['max_fullband_direct_power_leakage'] for x in checks))
