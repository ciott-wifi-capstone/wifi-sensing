"""Frozen common-model transfer test; no chain-specific phase fitting."""
from audit_inputs import *
from validate_background import ratio
from scipy.ndimage import maximum_filter
PAIRS=[('E1','E3',78,84),('E1','E2',78,81),('E3','E4',84,87)]
def template(x):
 z=x[:64];g=np.einsum('nmk,nlk->nkml',z,z.conj()).reshape(len(z),-1)
 return z[np.argmin(np.sum(abs(g-g.mean(0))**2,axis=1))]
def masks():
 for j in range(3):
  te=np.zeros((3,30));te[j]=1;yield 'chain'+str(j+1),1-te,te
 te=np.zeros((3,30));te[:,2::3]=1;yield 'held_frequency_mod3',1-te,te
def examine(x,ref):
 rows=[]
 for i in np.linspace(0,len(x)-1,min(64,len(x)),dtype=int):
  a=x[i];chain=[rat.fit(a[j:j+1],ref[j:j+1]) for j in range(3)]
  ds=max(abs(rat.wrap_delay(np.array([c[0] for c in chain])-chain[0][0])))
  ps=max(abs(rat.wrap_phase(np.array([c[1] for c in chain])-chain[0][1])))*180/np.pi
  for name,tr,te in masks():
   t,p,c,e=rat.fit(a,ref,tr);cor=rat.apply(a,t,p);sc=rat.score(cor,ref,te)
   rows.append(dict(packet_index=int(i),mask=name,held_corr=sc[1],held_phase_deg=abs(sc[2])*180/np.pi,delay_spread_ns=ds,phase_spread_deg=ps,edge=bool(e)))
 df=pd.DataFrame(rows);corr=min(df.groupby('mask').held_corr.median());phase=max(df.groupby('mask').held_phase_deg.quantile(.9))
 ds=float(df.delay_spread_ns.quantile(.9));ps=float(df.phase_spread_deg.quantile(.9))
 gate=bool(corr>=.9 and phase<=15 and ds<=5 and ps<=15 and not df.edge.any())
 return df,dict(predictive_corr_min_median=corr,predictive_phase_max_p90_deg=phase,chain_delay_p90_ns=ds,chain_phase_p90_deg=ps,common_model_pass=gate)
def main():
 rat.TAU=rat.TAU[:-1];gates=[];checks=[];scores=[]
 thresholds=pd.read_csv(R/'empty_false_candidates.csv').query('alpha==0.1 and block=="full"')
 for ca,cb,ia,ib in PAIRS:
  pair=ca+'_'+cb
  for state in rat.MAP:
   src=np.load(R/'cache'/f'{ia:03d}_{state}.npz');dev=np.load(R/'cache'/f'{ib:03d}_{state}.npz')
   ref=template(src['x']);prefix=dev['x'][:64]
   dd,g=examine(prefix,ref);gates.append(dict(pair=pair,state=state,**g))
   dd['pair']=pair;dd['state']=state;dd['role']='development_prefix';checks.append(dd)
   for k in [1,2]:
    test=np.load(R/'cache'/f'{ib+k:03d}_{state}.npz');xx=test['x']
    dd,h=examine(xx,ref);dd['pair']=pair;dd['state']=state;dd['role']='target_run'+str(k+1);checks.append(dd)
    # A failed development gate always retains the unaligned baseline.
    corrected=xx.copy();naccept=0
    if g['common_model_pass']:
     for j,a in enumerate(xx):
      t,p,c,e=rat.fit(a,ref)
      if c>=.9 and not e:corrected[j]=rat.apply(a,t,p);naccept+=1
    a=rat.steering(state);th=float(thresholds[(thresholds.capture==ca)&(thresholds.state==state)].threshold.iloc[0])
    for mode,cov in [('unaligned',test['covariance']),('gated',rat.covariance(corrected))]:
     z=ratio(cov,src['covariance'],a,.1);peaks=(z==maximum_filter(z,size=(5,5),mode=('nearest','wrap')))&(z>th)
     inv=invariants(corrected if mode=='gated' else xx)
     scores.append(dict(pair=pair,state=state,target_run=k+1,mode=mode,development_pass=g['common_model_pass'],validation_common_pass=h['common_model_pass'],accepted_packets=naccept if mode=='gated' else 0,total_packets=len(xx),covariance_distance=distance(cov,src['covariance']),spatial_distance=distance(inv['spatial'],src['spatial']),amplitude_distance=distance(inv['amplitude'],src['amplitude']),threshold=th,max_ratio=z.max(),false_peaks=int(peaks.sum())))
  print(pair,'complete',flush=True)
 pd.DataFrame(gates).to_csv(R/'transfer_gates.csv',index=False)
 pd.concat(checks).to_csv(R/'transfer_predictive_checks.csv',index=False)
 df=pd.DataFrame(scores);df.to_csv(R/'transfer_background_scores.csv',index=False)
 sm=df.groupby(['pair','target_run','mode']).agg(states=('state','size'),development_passes=('development_pass','sum'),validation_passes=('validation_common_pass','sum'),false_peaks=('false_peaks','sum'),median_cov_distance=('covariance_distance','median'),median_max_ratio=('max_ratio','median'))
 sm.to_csv(R/'transfer_summary.csv');print(sm.to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
