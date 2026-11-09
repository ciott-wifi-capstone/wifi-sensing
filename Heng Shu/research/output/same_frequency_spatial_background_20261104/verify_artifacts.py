from spatial_model import *
import py_compile,platform
def h(p):
 hh=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):hh.update(b)
 return hh.hexdigest()
def main():
 out={};frozen=[]
 for fn in ['frozen_model_hashes.json','occupied_model_freeze.json','frozen_predictions.json']:
  j=json.loads((R/fn).read_text())
  for name,want in j.items():
   assert h(R/name)==want,(fn,name)
   frozen.append(dict(freeze_file=fn,path=name,sha256=want))
 out['frozen_hashes_verified']=len(frozen)
 caches=pd.concat([pd.read_csv(R/f) [['path','sha256']] for f in ['training_sources.csv','validation_sources.csv','occupied_sources.csv']]).drop_duplicates()
 for r in caches.itertuples():
  assert h(r.path)==r.sha256,r.path
  z=np.load(r.path);assert np.isfinite(z['x']).all()
  assert (np.diff(z['packet_id'])>0).all(),r.path
  if 'time_s' in z.files:assert (np.diff(z['time_s'])>=0).all()
 out['unique_cached_state_files_verified']=len(caches)
 em=pd.read_csv(B/'source_manifest.csv')
 im=pd.read_csv(S/'input_manifest.csv'); keys=[0,3,12,15,21,24,30,33]
 original=[]
 for r in em[['index','capture','run','raw_path','raw_sha256','requested_seconds']].drop_duplicates().itertuples():
  original.append(dict(index=r.index,label=r.capture,run=r.run,path=r.raw_path,sha256=r.raw_sha256,duration=r.requested_seconds,role='empty'))
 for r in im[im['index'].isin(keys)].itertuples():
  original.append(dict(index=r.index,label=r.label,run=r.run,path=r.path,sha256=r.sha256,duration=r.duration_s,role='occupied'))
 originals=pd.DataFrame(original)
 for r in originals.itertuples():assert h(r.path)==r.sha256,r.path
 originals.to_csv(R/'original_sources.csv',index=False)
 out['original_CSI_files_verified']=len(originals)
 out['empty_cached_packets_range']=[int(em.packets.min()),int(em.packets.max())]
 ev=pd.read_csv(R/'empty_validation.csv');ov=pd.read_csv(R/'occupied_validation.csv')
 assert len(ev)==96 and len(ov)==192
 assert not ev.duplicated(['source','target','index','mode','state']).any()
 assert not ov.duplicated(['source','record','mode','state']).any()
 for df in [ev,ov]:
  assert np.isfinite(df[['distance','distance_threshold','max_score','score_threshold']].values).all()
  assert (df.stable_candidates<=df.raw_candidates).all()
 assert len(pd.read_csv(R/'occupied_subarray_clusters.csv'))==0
 assert len(pd.read_csv(R/'occupied_repeat_candidates.csv'))==0
 assert len(pd.read_csv(R/'reference_robust_candidates.csv'))==0
 assert (ov[ov['mode']=='pi_hypothesis'].matrix_change).all()
 out['row_counts']={'empty_state_cases':len(ev),'occupied_state_cases':len(ov),'occupied_unique_records':ov.record.nunique(),'empty_G_cluster_cases':len(pd.read_csv(R/'empty_cross_G_clusters.csv'))}
 # Match per-state candidate counts against their detailed evidence rows.
 for stem,df in [('empty',ev),('occupied',ov)]:
  detail=pd.read_csv(R/('empty_false_candidates.csv' if stem=='empty' else 'occupied_candidates.csv'))
  cols=['source','target','index','mode','state'] if stem=='empty' else ['source','record','mode','state']
  cc=detail.groupby(cols).agg(raw=('score','size'),stable=('stable','sum'))
  for row in df.itertuples():
   key=tuple(row._asdict()[c] for c in cols)
   if key in cc.index:raw,stable=cc.loc[key,['raw','stable']]
   else:raw,stable=0,0
   assert (row.raw_candidates,row.stable_candidates)==(raw,stable),(stem,key)
 out['candidate_count_consistency']='passed'
 out['numerics']=checks()
 synth=json.loads((R/'synthetic_end_to_end.json').read_text())
 assert synth['empty_stable_state_peaks']==0
 assert all(x['stable_peaks']==[21.0] for x in synth['states'])
 out['synthetic_positive_control']='four G states recover 21 deg; ideal calibrated simulation only'
 for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
 out['python']=platform.python_version()
 out['status']='passed'
 (R/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 deps=[S/'geometry.json',S/'input_manifest.csv',B/'source_manifest.csv',P/'frozen_signs.json',P/'validate_sign.py']
 (R/'dependency_hashes.json').write_text(json.dumps({str(p):h(p) for p in deps},indent=2),encoding='utf8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
