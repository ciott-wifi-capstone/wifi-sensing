from validate_sign import *
import py_compile
meta=json.loads((O/'frozen_signs.json').read_text())
assert hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest()==meta['protocol_sha256']
for r in meta['sources']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
val=pd.read_csv(O/'validation_manifest.csv')
for r in val.itertuples():assert hashlib.sha256(Path(r.path).read_bytes()).hexdigest()==r.sha256
for path in O.glob('*.py'):py_compile.compile(str(path),doraise=True)
d=pd.read_csv(O/'comparisons.csv');assert len(d)==72
g=pd.read_csv(O/'common_gates_after_pi.csv');assert len(g)==54
u=pd.read_csv(O/'unitary_checks.csv');assert u.amplitude_max_difference.max()<1e-12
assert u.covariance_eigenvalue_max_difference.max()<1e-10
h=pd.read_csv(O/'occupied_gates_after_pi.csv');hm=pd.read_csv(O/'occupied_mean/occupied_gates_after_pi.csv')
assert len(h)==96 and len(hm)==96
s0=json.loads((O/'occupied_sign_choices.json').read_text());s1=json.loads((O/'occupied_mean/occupied_sign_choices.json').read_text())
assert s0==s1
source_paths=[B/'audit_inputs.py',B/'transfer_common.py',B/'validate_background.py',B/'empty_false_candidates.csv',Path(rat.__file__),Path(rat.__file__).with_name('config.json')]
checks=dict(training_sources_verified=len(meta['sources']),validation_cache_references_verified=len(val),empty_comparison_rows=len(d),state_gate_rows=len(g),amplitude_invariance_max=float(u.amplitude_max_difference.max()),covariance_eigenvalue_invariance_max=float(u.covariance_eigenvalue_max_difference.max()),occupied_rows_per_reference=len(h),mean_control_uses_identical_discrete_signs=True,spectrum_fc_hz=rat.FC,frequencies_hz=rat.F.tolist(),physical_human_path_verified=False,visual_review='Both PNG figures inspected for labels, units, roles and cropping.',dependencies={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
(O/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
artifacts={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_hashes.json'}
(O/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2),encoding='utf8')
print({k:v for k,v in checks.items() if k not in ['frequencies_hz','dependencies']})
