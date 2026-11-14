from spatial_model import *
KEYS=[0,3,12,15,21,24,30,33]
GROUPS=[('pair_0_3',0,3),('pair_12_15',12,15),('pair_21_24',21,24),('pair_30_33',30,33)]
def clusters(peaks_by_state):
 candidates=[]
 for trial in ANG:
  chosen=[]
  for s in ['G1','G2','G3','G4']:
   vals=[p for p in peaks_by_state.get(s,[]) if p['stable'] and abs(p['angle_parameter_deg']-trial)<=9]
   if vals:
    p=min(vals,key=lambda p:(abs(p['angle_parameter_deg']-trial),-p['score']))
    chosen.append((s,p['angle_parameter_deg'],p['score']))
  if len(chosen)<3:continue
  for _ in range(2):
   center=float(np.median([p[1] for p in chosen]))
   chosen=[p for p in chosen if abs(p[1]-center)<=9]
   if not chosen:break
  if len(chosen)<3:continue
  center=float(np.median([p[1] for p in chosen]))
  candidates.append(dict(center_deg=center,states=','.join(p[0] for p in chosen),state_count=len(chosen),spread_deg=max(p[1] for p in chosen)-min(p[1] for p in chosen),mean_score=float(np.mean([p[2] for p in chosen]))))
 kept=[]
 for c in sorted(candidates,key=lambda p:(-p['state_count'],p['spread_deg'],-p['mean_score'])):
  if not any(abs(c['center_deg']-k['center_deg'])<=9 for k in kept):kept.append(c)
 return kept
def main():
 models={};thresholds=[];background_checks=[];signsrc={'E1':np.ones(3),'E2':np.array(json.loads((P/'frozen_signs.json').read_text())['pairs']['E1_E2']['sign'])}
 canonical={}
 for cap in ['E1','E2']:
  idx=CAP[cap]
  for s in ST:
   x=loadempty(idx,s)*signsrc[cap][None,:,None];dev=loadempty(idx+1,s)*signsrc[cap][None,:,None];canonical[(cap,s)]=x
   m,td,ts=model_from_source(x,dev,s);models[(cap,s)]=(m,td,ts)
   np.savez_compressed(R/f'occupied_model_{cap}_{s}.npz',background=m['background'],distance_threshold=td,score_thresholds=ts)
   stat,pp,_=assess(loadempty(idx+2,s)*signsrc[cap][None,:,None],m,td,ts)
   background_checks.append(dict(source=cap,state=s,**stat))
   thresholds.append(dict(source=cap,state=s,distance_threshold=td,score_threshold=ts[0]))
 pd.DataFrame(background_checks).to_csv(R/'canonical_empty_validation.csv',index=False)
 pd.DataFrame(thresholds).to_csv(R/'occupied_source_thresholds.csv',index=False)
 # Choose only finite per-record signs from prefix; no continuous alignment.
 decisions={}
 for cap in ['E1','E2']:
  for key in KEYS:
   source={s:canonical[(cap,s)] for s in ['G1','G2']}
   target={s:np.load(S/'packets'/f'{key}_{s}.npz')['x'] for s in ['G1','G2']}
   sign,sc=fit_sign(source,target);decisions[f'{cap}_{key}']=dict(sign=sign.tolist(),scores=sc)
 (R/'occupied_frozen_signs.json').write_text(json.dumps(decisions,indent=2))
 freezes=list(R.glob('occupied_model_*.npz'))+[R/'occupied_frozen_signs.json',R/'occupied_source_thresholds.csv']
 (R/'occupied_model_freeze.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in freezes},indent=2))
 rows=[];cand=[];cl=[];sources=[];allclusters={}
 for cap in ['E1','E2']:
  for key in KEYS:
   sign=np.array(decisions[f'{cap}_{key}']['sign']);bystate={}
   for s in ST:
    path=S/'packets'/f'{key}_{s}.npz';z=np.load(path);raw=z['x'][64:]
    sources.append(dict(source=cap,record=key,state=s,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),packets=len(raw)))
    m,td,ts=models[(cap,s)]
    for mode,x in [('raw',raw),('pi_hypothesis',raw*sign[None,:,None])]:
     stat,peaks,arrays=assess(x,m,td,ts);rows.append(dict(source=cap,record=key,state=s,mode=mode,**stat))
     for p in peaks:cand.append(dict(source=cap,record=key,state=s,mode=mode,**p))
     np.savez_compressed(R/'spectra'/f'occupied_{cap}_{key}_{s}_{mode}.npz',**arrays)
     if mode=='pi_hypothesis':bystate[s]=peaks
   cc=clusters(bystate);allclusters[(cap,key)]=cc
   for c in cc:cl.append(dict(source=cap,record=key,**c))
   print('occupied',cap,key,'clusters',len(cc),flush=True)
 repeats=[];repmap={}
 for cap in ['E1','E2']:
  for group,a,b in GROUPS:
   found=[]
   for c in allclusters[(cap,a)]:
    for d in allclusters[(cap,b)]:
     if abs(c['center_deg']-d['center_deg'])<=9:
      rr=dict(source=cap,group=group,record_a=a,record_b=b,center_a=c['center_deg'],center_b=d['center_deg'],repeat_center=(c['center_deg']+d['center_deg'])/2,states_a=c['states'],states_b=d['states'])
      repeats.append(rr);found.append(rr)
   repmap[(cap,group)]=found
 robust=[]
 for group,a,b in GROUPS:
  for c in repmap[('E1',group)]:
   for d in repmap[('E2',group)]:
    if abs(c['repeat_center']-d['repeat_center'])<=9:robust.append(dict(group=group,center_e1=c['repeat_center'],center_e2=d['repeat_center'],human_path_verified=False))
 pd.DataFrame(rows).to_csv(R/'occupied_validation.csv',index=False)
 pd.DataFrame(cand,columns=['source','record','state','mode','angle_parameter_deg','score','temporal_blocks','frequency_halves_consistent','stable']).to_csv(R/'occupied_candidates.csv',index=False)
 pd.DataFrame(cl,columns=['source','record','center_deg','states','state_count','spread_deg','mean_score']).to_csv(R/'occupied_subarray_clusters.csv',index=False)
 pd.DataFrame(repeats,columns=['source','group','record_a','record_b','center_a','center_b','repeat_center','states_a','states_b']).to_csv(R/'occupied_repeat_candidates.csv',index=False)
 pd.DataFrame(robust,columns=['group','center_e1','center_e2','human_path_verified']).to_csv(R/'reference_robust_candidates.csv',index=False)
 pd.DataFrame(sources).to_csv(R/'occupied_sources.csv',index=False)
 summary=pd.DataFrame(rows).groupby(['source','record','mode']).agg(states=('state','size'),change_states=('matrix_change','sum'),raw_peaks=('raw_candidates','sum'),stable_state_peaks=('stable_candidates','sum'))
 summary.to_csv(R/'occupied_summary.csv');print(summary.to_string())
 print('repeat_candidates',len(repeats),'reference_robust',len(robust),flush=True)
 # Do not read physical human coordinates. Freeze outputs before any later interpretation.
 paths=[R/'occupied_validation.csv',R/'occupied_candidates.csv',R/'occupied_subarray_clusters.csv',R/'occupied_repeat_candidates.csv',R/'reference_robust_candidates.csv']
 (R/'frozen_predictions.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
