from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent
B=O.parent/'empty_background_validation_20261103'
sys.path.insert(0,str(B))
from transfer_common import rat,template,examine,ratio,invariants,distance,PAIRS
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits
ST=list(rat.MAP);PA=[(0,1),(0,2),(1,2)]
CAND=np.array([[1,b,c] for b,c in itertools.product([1,-1],repeat=2)])
TRAIN=np.arange(30)%3!=2;TEST=~TRAIN
def load(i,s):
 return np.load(B/'cache'/f'{i:03d}_{s}.npz')
def unit(a):return a/np.maximum(abs(a),1e-30)
def qmean(x):return np.array([unit(x[:,i]*x[:,j].conj()).mean(0) for i,j in PA])
def fit_sign(xs,xt):
 delta=np.array([qmean(xt[s][:64])*qmean(xs[s][:64]).conj() for s in ['G1','G2']])
 scores=[]
 for v in CAND:
  factor=np.array([v[i]*v[j] for i,j in PA])
  scores.append(float(np.real(delta[:,:,TRAIN]*factor[None,:,None]).mean()))
 best=int(np.argmax(scores));return CAND[best],scores
def fit():
 frozen={};sources=[]
 for ca,cb,ia,ib in PAIRS:
  xs={s:load(ia,s)['x'] for s in ['G1','G2']};xt={s:load(ib,s)['x'] for s in ['G1','G2']}
  v,scores=fit_sign(xs,xt);frozen[ca+'_'+cb]=dict(sign=v.tolist(),scores=scores,runner_up_margin=float(np.sort(scores)[-1]-np.sort(scores)[-2]))
  for i in [ia,ib]:
   for s in ['G1','G2']:
    p=B/'cache'/f'{i:03d}_{s}.npz';sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 testsource={s:xs[s] for s in ['G1','G2']}
 synthetic={s:testsource[s]*np.array([1,-1,-1])[None,:,None]*np.exp(.7j) for s in ['G1','G2']}
 v,_=fit_sign(testsource,synthetic);assert np.array_equal(v,[1,-1,-1])
 meta=dict(protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest(),pairs=frozen,sources=sources,synthetic_known_sign_recovered=True)
 (O/'frozen_signs.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(frozen,indent=2),flush=True)
def run():
 meta=json.loads((O/'frozen_signs.json').read_text());assert hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest()==meta['protocol_sha256']
 rat.TAU=rat.TAU[:-1]
 th=pd.read_csv(B/'empty_false_candidates.csv').query('alpha==0.1 and block=="full"')
 rows=[];gates=[];predict=[];phase=[];manifest=[];numerics=[]
 for ca,cb,ia,ib in PAIRS:
  pair=ca+'_'+cb;v=np.array(meta['pairs'][pair]['sign'])
  for state in ST:
   src=load(ia,state);source=src['x'];ref=template(source);invsrc=invariants(source)
   steering=rat.steering(state);threshold=float(th[(th.capture==ca)&(th.state==state)].threshold.iloc[0])
   for target_offset in [0,1,2]:
    rec=ib+target_offset;z=load(rec,state);raw=z['x'];corrected=raw*v[None,:,None]
    path=B/'cache'/f'{rec:03d}_{state}.npz';manifest.append(dict(pair=pair,record=rec,state=state,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    scope='development_prefix' if target_offset==0 else 'complete_run_validation'
    xx=corrected[:64] if target_offset==0 else corrected
    detail,gate=examine(xx,ref);gates.append(dict(pair=pair,record=rec,run=target_offset+1,state=state,scope=scope,sign_trained_state=state in ['G1','G2'],**gate))
    detail['pair']=pair;detail['record']=rec;detail['state']=state;predict.append(detail)
    for mode,x in [('raw',raw),('fixed_pi',corrected)]:
     inv=invariants(x);delta=qmean(x)*qmean(source).conj();ph=abs(np.angle(delta))*180/np.pi
     for support,mask in [('development_frequency',TRAIN),('withheld_frequency',TEST)]:
      phase.append(dict(pair=pair,run=target_offset+1,state=state,mode=mode,frequency_role=support,pair_phase_median_deg=float(np.median(ph[:,mask])),pair_phase_p90_deg=float(np.quantile(ph[:,mask],.9)),mean_pair_support=float(abs(delta[:,mask]).mean())))
     if target_offset==0:continue
     score=ratio(inv['covariance'],invsrc['covariance'],steering,.1)
     peakmask=(score==maximum_filter(score,size=(5,5),mode=('nearest','wrap')))&(score>threshold)
     music=rat.music(inv['covariance'],steering)
     rows.append(dict(pair=pair,run=target_offset+1,state=state,mode=mode,amplitude_distance=distance(inv['amplitude'],invsrc['amplitude']),spatial_distance=distance(inv['spatial'],invsrc['spatial']),covariance_distance=distance(inv['covariance'],invsrc['covariance']),threshold=threshold,max_ratio=score.max(),false_peaks=int(peakmask.sum()),states_with_false_peaks=bool(peakmask.any())))
     np.savez_compressed(O/f'{pair}_run{target_offset+1}_{state}_{mode}.npz',angle_deg=rat.ANG,delay_parameter_ns=rat.TAU,background_ratio=score,music=music)
    amp=float(abs(abs(raw)-abs(corrected)).max());c1=rat.covariance(raw);c2=rat.covariance(corrected)
    eig=float(abs(np.linalg.eigvalsh(c1)-np.linalg.eigvalsh(c2)).max());assert amp<1e-12 and eig<1e-10
    numerics.append(dict(pair=pair,run=target_offset+1,state=state,amplitude_max_difference=amp,covariance_eigenvalue_max_difference=eig))
  print('validated',pair,flush=True)
 for name,data in [('comparisons',rows),('common_gates_after_pi',gates),('phase_residuals',phase),('validation_manifest',manifest),('unitary_checks',numerics)]:
  pd.DataFrame(data).to_csv(O/f'{name}.csv',index=False)
 pd.concat(predict).to_csv(O/'held_predictions_after_pi.csv',index=False)
 d=pd.DataFrame(rows);sm=d.groupby(['pair','run','mode']).agg(amplitude_distance=('amplitude_distance','median'),spatial_distance=('spatial_distance','median'),covariance_distance=('covariance_distance','median'),false_peaks=('false_peaks','sum'),states_with_false_peaks=('states_with_false_peaks','sum'))
 sm.to_csv(O/'summary.csv');print(sm.to_string(),flush=True)
 g=pd.DataFrame(gates).groupby(['pair','run']).common_model_pass.agg(['sum','count']);g.to_csv(O/'gate_summary.csv');print(g.to_string(),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if sys.argv[-1]=='fit':fit()
  else:run()
