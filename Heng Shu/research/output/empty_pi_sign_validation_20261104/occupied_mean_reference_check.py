"""Occupied compatibility after finite pi hypotheses; no human coordinates."""
from validate_sign import *
S=O.parent/'static_human_aoa_validation_20261101'
meanrefs=np.load(O/'source_mean_references.npz')
O=O/'occupied_mean';O.mkdir(exist_ok=True)
KEYS=[3,15,24,33]
def main():
 decisions={};rows=[];pred=[];phase=[]
 # Freeze discrete choices before examining the remaining packets.
 for cap,idx in [('E1',78),('E2',81)]:
  for key in KEYS:
   src={s:load(idx,s)['x'] for s in ['G1','G2']}
   tar={s:np.load(S/'packets'/f'{key}_{s}.npz')['x'] for s in ['G1','G2']}
   v,scores=fit_sign(src,tar);decisions[f'{cap}_{key}']=dict(sign=v.tolist(),scores=scores)
 (O/'occupied_sign_choices.json').write_text(json.dumps(decisions,indent=2),encoding='utf8')
 for cap,idx in [('E1',78),('E2',81)]:
  for key in KEYS:
   v=np.array(decisions[f'{cap}_{key}']['sign'])
   for state in ST:
    src=load(idx,state)['x'];ref=meanrefs[cap+'_'+state];raw=rat.unit(np.load(S/'packets'/f'{key}_{state}.npz')['x'])
    for scope,xx in [('prefix_development',raw[:64]),('remaining_validation',raw[64:])]:
     corrected=xx*v[None,:,None]
     d,g=examine(corrected,ref);rows.append(dict(empty_source=cap,occupied_record=key,state=state,scope=scope,**g))
     d['empty_source']=cap;d['occupied_record']=key;d['state']=state;d['scope']=scope;pred.append(d)
     for mode,x in [('raw',xx),('pi_hypothesis',corrected)]:
      a=invariants(src);b=invariants(x);res=abs(np.angle(qmean(x)*qmean(src).conj()))*180/np.pi
      phase.append(dict(empty_source=cap,occupied_record=key,state=state,scope=scope,mode=mode,held_frequency_phase_median_deg=float(np.median(res[:,TEST])),spatial_distance=distance(a['spatial'],b['spatial']),amplitude_distance=distance(a['amplitude'],b['amplitude']),covariance_distance=distance(a['covariance'],b['covariance'])))
   print('occupied',cap,key,flush=True)
 pd.DataFrame(rows).to_csv(O/'occupied_gates_after_pi.csv',index=False);pd.concat(pred).to_csv(O/'occupied_held_predictions.csv',index=False);pd.DataFrame(phase).to_csv(O/'occupied_invariants.csv',index=False)
 print(pd.DataFrame(rows).groupby(['empty_source','scope']).common_model_pass.agg(['sum','count']).to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
