from audit import *
import py_compile
def main():
 for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
 files={}
 for csv in ['source_hashes.csv','batch_provenance_hashes.csv']:
  for q in pd.read_csv(R/csv).itertuples():
   assert hashfile(q.path)==q.sha256
   files[q.path]=q.sha256
 for q in pd.read_csv(R/'occupied_within_sources.csv').itertuples():
  assert hashfile(q.raw_path)==q.raw_sha256
  assert hashfile(q.metadata_path)==q.metadata_sha256
 # Every input state chain list agrees with confirmed geometry.
 seen=set();eventcount=0
 for p in files:
  pp=Path(p)
  if pp.name=='run.json':
   evt=pp.parent/'events.log'
   if evt in seen:continue
   seen.add(evt)
   ev=re.findall(r'READY (G[1-4]|C[12]) chains=([\d,]+) t=(\d+) seq_idx=(\d)/6',evt.read_text())
   for s,chain,t,seq in ev:
    assert list(map(int,chain.split(',')))==sm.MAP[s] and int(seq)==ST.index(s)
   eventcount+=len(ev)
 for name,hashkey,modelfile in [('phase_model_freeze.json','model_hash','frozen_phase_models.npz'),('gain_model_freeze.json','model_hash','frozen_gain_phase_models.npz')]:
  assert json.loads((R/name).read_text())[hashkey]==hashfile(R/modelfile)
 for name in ['within_batch_predictions.json','occupied_baseline_references.json','within_batch_model_freeze.json']:
  for p,h in json.loads((R/name).read_text()).items():
   path=Path(p);path=path if path.is_absolute() else R/path
   assert hashfile(path)==h,(name,path)
 br=pd.read_csv(R/'baseline_reconstruction.csv');assert len(br)==36 and br.packet_ids_match.all() and br.normalized_CSI_max_error.max()<1e-12
 dc=pd.read_csv(R/'independent_decoder_checks.csv');assert len(dc)==384 and dc.max_CSI_difference.max()==0
 wi=pd.read_csv(R/'occupied_within_validation.csv');assert len(wi)==144
 assert not wi.duplicated(['index','reference','state']).any()
 assert len(pd.read_csv(R/'occupied_within_clusters.csv'))==0
 assert len(pd.read_csv(R/'occupied_within_repeats.csv'))==0
 wr=pd.read_csv(R/'occupied_within_records.csv');assert len(wr)==12 and wr.complete_window.sum()==11
 gt=np.array([0,7.,-9.]);gp=np.array([0,.35,-.7]);known=np.exp(1j*(gp[:,None]+2*np.pi*F[None]*gt[:,None]*1e-9))
 src={s:load(78,s) for s in ST};tar={s:src[s]/known[None] for s in ST}
 estimated,par=phase_fit(src,tar,True)
 correction_error=float(abs(estimated-known).max());assert correction_error<1e-6,correction_error
 held=[sm.dist(sm.spatial(src[s]),sm.spatial(tar[s]*estimated[None])) for s in ['G3','G4','C1','C2']]
 assert max(held)<1e-6
 out=dict(status='passed',unique_provenance_files=len(files),run_event_logs=len(seen),READY_events_checked=eventcount,baseline_state_caches_reconstructed=len(br),independent_bit_decoded_packets=len(dc),exact_decoder_max_error=0,within_batch_records=len(wr),complete_within_batch_records=int(wr.complete_window.sum()),within_state_scores=len(wi),new_continuous_occupied_correction=False,synthetic_known_phase_slope_max_error=correction_error,synthetic_held_state_covariance_max_error=max(held))
 (R/'verification.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
