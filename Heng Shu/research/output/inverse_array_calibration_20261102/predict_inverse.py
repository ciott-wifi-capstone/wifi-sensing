"""Freeze spectral predictions without reading withheld human geometry."""
from fit_inverse import *
from music_baseline import steering,spectrum,ANG
O=R/'scaled'
def fastspec(cov,a):
 _,v=np.linalg.eigh(cov);den=np.maximum(np.sum(abs(a)**2,axis=0)-np.sum(abs(v[:,-3:].conj().T@a)**2,axis=0),1e-12)
 p=(1/den).reshape(len(ANG),-1);return p/p.max()
def main():
 weights=pd.read_csv(S/'results/noncoherent_fusion.csv',dtype={'index':str})
 covs=np.load(O/'branch_corrected_covariances.npz')
 dom=pd.read_csv(O/'dominant_phase_candidates.csv')
 phases={'pi_only':np.zeros(12),'dominant_tx':np.deg2rad(dom.tx_dominant_phase_deg.to_numpy()),'dominant_human':np.deg2rad(dom.human_dominant_phase_deg.to_numpy())}
 for mode in MODES:phases[mode]=np.load(O/f'fit_{mode}.npz')['phase_rad']
 rows=[];fused=[];checks=[]
 for key in KEYS:
  saved={};ww={}
  for state in ST:
   a,tau,_=steering(rat,state)
   z=np.load(S/'packets'/f'{key}_{state}.npz')
   rawcov=rat.covariance(z['out_C'][64:])
   p=fastspec(rawcov,a);saved[state+'_baseline']=p
   if key==0 and state=='G1':
    difference=float(abs(p-spectrum(rawcov,a,3,len(ANG),len(tau))).max());assert difference<1e-9
    checks.append(dict(check='signal_noise_subspace_equivalence',max_difference=difference))
   for mode,phase in phases.items():
    d=np.repeat(np.exp(-1j*phase[IDS[state]]),NF)
    cov=covs[f'{key}_{state}']*d[:,None]*d[None,:].conj()
    p=fastspec(cov,a);saved[state+'_'+mode]=p
   for mode in ['baseline']+list(phases):
    p=saved[state+'_'+mode];i,j=np.unravel_index(p.argmax(),p.shape)
    rows.append(dict(record=key,state=state,model=mode,angle_deg=ANG[i],delay_parameter_ns=tau[j],human_verified=False))
   if state.startswith('G'):ww[state]=float(weights[(weights['index']==str(key))&(weights.state==state)&(weights.method=='C')].iloc[0].weight)
  for mode in ['baseline']+list(phases):
   p=sum(w*saved[s+'_'+mode].max(1) for s,w in ww.items())/sum(ww.values());p/=p.max()
   saved['fused_'+mode]=p;fused.append(dict(record=key,model=mode,angle_deg=ANG[p.argmax()],contrast_db=float(10*np.log10(1/p.min())),human_verified=False))
  np.savez_compressed(O/f'predictions_{key}.npz',angle_deg=ANG,delay_parameter_ns=tau,**saved)
 pd.DataFrame(rows).to_csv(O/'state_predictions.csv',index=False)
 pd.DataFrame(fused).to_csv(O/'fused_predictions.csv',index=False)
 (O/'prediction_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 paths=list(O.glob('predictions_*.npz'))+[O/'fused_predictions.csv',O/'state_predictions.csv']
 (O/'frozen_predictions.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf8')
 print('Frozen',len(fused),'fusion predictions,',len(rows),'state predictions.')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
