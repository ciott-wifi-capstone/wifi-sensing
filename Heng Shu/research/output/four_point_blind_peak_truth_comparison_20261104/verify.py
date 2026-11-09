from export_blind import *
import py_compile
checks={}
for name,want in json.loads((A/'within_batch_predictions.json').read_text()).items():
 assert sha(A/name)==want
for name,want in json.loads((R/'blind_predictions_frozen.json').read_text()).items():
 assert sha(R/name)==want
sources=pd.read_csv(R/'spectrum_sources.csv')
for v in sources.itertuples(index=False):assert sha(v.path)==v.sha256
checks['frozen_hashes_and_144_spectrum_sources']=len(sources)
b=pd.read_csv(R/'blind_predictions.csv');c=pd.read_csv(R/'all_frozen_candidates.csv')
d=pd.read_csv(R/'blind_peak_truth_comparison.csv');gt=pd.read_csv(R/'theoretical_angles.csv')
assert len(b)==len(d)==144 and len(gt)==24
for v in b.itertuples(index=False):
 z=np.load(A/'occupied_spectra'/f'{v.reference}_{v.index}_{v.state}.npz')
 ix=int(np.argmax(z['scores'][0]))
 assert v.spectrum_top1_deg==float(z['angles'][ix])
 assert np.isclose(v.spectrum_top1_score,z['scores'][0][ix])
 q=c[(c['index']==v.index)&(c.reference==v.reference)&(c.state==v.state)&c.stable].sort_values(['score','angle_parameter_deg'],ascending=[False,True])
 assert (not len(q) and np.isnan(v.blind_stable_top1_deg)) or (len(q) and v.blind_stable_top1_deg==q.angle_parameter_deg.iloc[0])
checks['all_blind_maxima_and_stable_choices_recomputed']=144
assert np.allclose(gt.theory_subarray_deg,np.degrees(np.arctan2(-gt.target_y-gt.subarray_center_y,3.65-gt.target_x)))
for y in [1.,2.]:
 for s,t in [('G1','G4'),('G2','G3'),('C1','C2')]:
  a=gt[(gt.target_y==y)&(gt.state==s)].theory_subarray_deg.iloc[0]
  b0=gt[(gt.target_y==-y)&(gt.state==t)].theory_subarray_deg.iloc[0]
  assert np.isclose(a,-b0)
checks['geometry_formula_and_mirror_symmetry']=True
for k in ['spectrum_top1','blind_stable_top1']:
 assert np.allclose(d[k+'_absolute_difference_deg'],abs(d[k+'_deg']-d.theory_subarray_deg),equal_nan=True)
for ref,q in d.groupby('reference'):
 assert len(q)==72 and q.complete_window.sum()==66
 assert set(q[~q.complete_window]['index'])=={2}
checks['coverage']={'E1':72,'E2':72,'complete_per_reference':66,'partial_record':2}
assert not b.human_path_validated.any()
for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
checks['python_compile']=True
checks['status']='PASS'
(R/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
hashes={p.name:sha(p) for p in sorted(R.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'}
(R/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
