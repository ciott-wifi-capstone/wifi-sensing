from audit import *
import validate_occupied as vo
def main():
 old=O/'same_frequency_spatial_background_20261104';S=O/'static_human_aoa_validation_20261101'
 mf=pd.read_csv(S/'input_manifest.csv');decisions=json.loads((old/'occupied_frozen_signs.json').read_text())
 groups=[(0,1,2),(12,13,14),(21,22,23),(30,31,32)]
 caches={};sources=[];rows=[];details=[];cl=[];repeat=[];invrows=[];recordrows=[]
 (R/'occupied_packets').mkdir(exist_ok=True);(R/'occupied_spectra').mkdir(exist_ok=True)
 for group in groups:
  first=group[0];session=None
  for idx in group:
   row=mf[mf['index']==idx].iloc[0];path=Path(row.path);assert hashfile(path)==row.sha256
   meta=dec.read(path.parent/'run.json');sessioni=(meta['before']['boot_id'],meta['before']['phy'],meta['before']['ifindex'])
   if session is None:session=sessioni
   assert sessioni==session and meta['before']==meta['after']
   recordrows.append(dict(index=idx,batch_first=first,label=row.label,run=row.run,complete_window=row.complete_window,trailing_s=row.trailing_s,window_frames=row.window_frames,same_batch_epoch=True))
   if idx!=first:
    x,host,nic=dec.decode(path);t=(host-meta['start_ns'])/1e9
    starts,states,guard=dec.align(path.parent/'events.log',meta['start_ns']/1e9)
    ids={s:[] for s in ST}
    for st,si in zip(starts[:-1],states[:-1]):
     if st+.025>meta['requested_seconds'] or .025-2*guard<.005:continue
     a,b=np.searchsorted(t,[st+guard,st+.025-guard]);ids[ST[si]].extend(range(a,b))
   for s in ST:
    if idx==first:
     z=np.load(S/'packets'/f'{idx}_{s}.npz');raw=z['x'];pid=z['packet_id']
    else:
     pid=np.array(ids[s],int)
     if len(pid)>768:pid=pid[np.linspace(0,len(pid)-1,768).astype(int)]
     raw=sm.unit(x[pid].astype(complex))
     np.savez_compressed(R/'occupied_packets'/f'{idx}_{s}.npz',x=raw,packet_id=pid,time_s=t[pid])
    assert len(raw)>128,(idx,s,len(raw))
    caches[(idx,s)]=raw[64:]
    sources.append(dict(index=idx,state=s,raw_path=str(path),raw_sha256=row.sha256,metadata_path=str(path.parent/'run.json'),metadata_sha256=hashfile(path.parent/'run.json'),events_sha256=hashfile(path.parent/'events.log'),available_guarded_packets=len(raw),scored_packets=len(raw)-64))
    if idx!=first:invrows.append(dict(index=idx,batch_first=first,state=s,**metrics(caches[(first,s)],caches[(idx,s)])))
   print('occupied parsed',idx,flush=True)
 pd.DataFrame(sources).to_csv(R/'occupied_within_sources.csv',index=False)
 pd.DataFrame(recordrows).to_csv(R/'occupied_within_records.csv',index=False)
 # Freeze source models and pi decisions reused from prior run1 before scoring later rounds.
 freeze={str(old/f):hashfile(old/f) for f in ['occupied_model_freeze.json','occupied_frozen_signs.json']}
 (R/'within_batch_model_freeze.json').write_text(json.dumps(freeze,indent=2))
 clusters={}
 for first,*rest in groups:
  for cap in ['E1','E2']:
   sign=np.array(decisions[f'{cap}_{first}']['sign'])
   for idx in [first]+rest:
    by={}
    for s in ST:
     m=np.load(old/f'occupied_model_{cap}_{s}.npz');model=sm.prepare(m['background'],s)
     stat,pp,arr=sm.assess(caches[(idx,s)]*sign[None,:,None],model,float(m['distance_threshold']),m['score_thresholds'])
     rows.append(dict(index=idx,batch_first=first,reference=cap,state=s,**stat))
     for p in pp:details.append(dict(index=idx,batch_first=first,reference=cap,state=s,**p))
     np.savez_compressed(R/'occupied_spectra'/f'{cap}_{idx}_{s}.npz',**arr)
     by[s]=pp
    cc=vo.clusters(by);clusters[(cap,idx)]=cc
    for c in cc:cl.append(dict(index=idx,batch_first=first,reference=cap,**c))
   for a in clusters[(cap,first)]:
    followers=[]
    for idx in rest:
     matches=[b for b in clusters[(cap,idx)] if abs(a['center_deg']-b['center_deg'])<=9]
     if matches:followers.append(idx)
    if followers:
     repeat.append(dict(reference=cap,batch_first=first,center_deg=a['center_deg'],matched_later_records=str(followers),all_three_repeat=len(followers)==2,all_three_complete=bool(mf[mf['index'].isin([first]+rest)].complete_window.all()),human_path_verified=False))
 for name,data,cols in [('occupied_within_invariants',invrows,None),('occupied_within_validation',rows,None),('occupied_within_candidates',details,None),('occupied_within_clusters',cl,['index','batch_first','reference','center_deg','states','state_count','spread_deg','mean_score']),('occupied_within_repeats',repeat,['reference','batch_first','center_deg','matched_later_records','all_three_repeat','all_three_complete','human_path_verified'])]:
  pd.DataFrame(data,columns=cols).to_csv(R/f'{name}.csv',index=False)
 outputs=['occupied_within_validation.csv','occupied_within_candidates.csv','occupied_within_clusters.csv','occupied_within_repeats.csv']
 (R/'within_batch_predictions.json').write_text(json.dumps({f:hashfile(R/f) for f in outputs},indent=2))
 print(pd.DataFrame(invrows).groupby(['batch_first','index']).agg(spatial=('spatial_distance','median'),phase=('phase_median_deg','median'),amp=('amplitude_distance','median'),passed=('gate','sum')).to_string())
 print('clusters',len(cl),'repeats',len(repeat))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
