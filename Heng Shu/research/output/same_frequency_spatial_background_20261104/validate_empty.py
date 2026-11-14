from spatial_model import *
def main():
 check=checks();(R/'numerical_checks.json').write_text(json.dumps(check,indent=2))
 models={};th=[];src=[]
 for cap,idx in CAP.items():
  for s in ST:
   m,td,ts=model_from_source(loadempty(idx,s),loadempty(idx+1,s),s)
   models[(cap,s)]=(m,td,ts)
   np.savez_compressed(R/f'model_{cap}_{s}.npz',background=m['background'],distance_threshold=td,score_thresholds=ts)
   th.append(dict(capture=cap,state=s,distance_threshold=td,all_frequency_threshold=ts[0],lower_half_threshold=ts[1],upper_half_threshold=ts[2]))
   for j in [idx,idx+1]:
    path=B/'cache'/f'{j:03d}_{s}.npz';src.append(dict(capture=cap,state=s,index=j,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 pd.DataFrame(th).to_csv(R/'frozen_thresholds.csv',index=False);pd.DataFrame(src).to_csv(R/'training_sources.csv',index=False)
 (R/'frozen_model_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(R.glob('model_*.npz'))+[R/'frozen_thresholds.csv',R/'PROTOCOL.md']},indent=2))
 signs=json.loads((P/'frozen_signs.json').read_text())['pairs']
 rows=[];candidates=[];valsrc=[]
 cases=[(cap,cap,idx+2,'within',np.ones(3)) for cap,idx in CAP.items()]
 for ca,cb in [('E1','E2'),('E3','E4'),('E1','E3')]:
  for off in [1,2]:
   for mode,v in [('raw',np.ones(3)),('fixed_pi',np.array(signs[ca+'_'+cb]['sign']))]:
    cases.append((ca,cb,CAP[cb]+off,mode,v))
 for ca,cb,idx,mode,v in cases:
  label=f'{ca}_{cb}_{idx}_{mode}'
  for s in ST:
   x=loadempty(idx,s)*v[None,:,None];m,td,ts=models[(ca,s)]
   stat,peaks,arrays=assess(x,m,td,ts)
   rows.append(dict(source=ca,target=cb,index=idx,run=idx-CAP[cb]+1,mode=mode,state=s,**stat))
   for p in peaks:candidates.append(dict(source=ca,target=cb,index=idx,mode=mode,state=s,**p))
   np.savez_compressed(R/'spectra'/f'empty_{label}_{s}.npz',**arrays)
   path=B/'cache'/f'{idx:03d}_{s}.npz';valsrc.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  print(label,flush=True)
 df=pd.DataFrame(rows);df.to_csv(R/'empty_validation.csv',index=False)
 pd.DataFrame(candidates,columns=['source','target','index','mode','state','angle_parameter_deg','score','temporal_blocks','frequency_halves_consistent','stable']).to_csv(R/'empty_false_candidates.csv',index=False)
 pd.DataFrame(valsrc).drop_duplicates().to_csv(R/'validation_sources.csv',index=False)
 sm=df.groupby(['source','target','run','mode']).agg(states=('state','size'),matrix_change_states=('matrix_change','sum'),raw_false_peaks=('raw_candidates','sum'),stable_false_peaks=('stable_candidates','sum'),median_distance=('distance','median'),median_max_score=('max_score','median'))
 sm.to_csv(R/'empty_summary.csv');print(sm.to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
