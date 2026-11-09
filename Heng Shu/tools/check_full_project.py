"""Run source-only research self-tests in a disposable copy, not the originals.
No capture processes, radio configuration, Internet calls or real data are used.
"""
from pathlib import Path
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
JOBS=[
 ('uniform','noncoherent_uniform_20261106','import predict_uniform as m; m.synthetic()'),
 ('coherent12','coherent_array_validation_20261106','import validate_coherent as m; m.synthetic()'),
 ('pcd','noncoherent_bc_followup_20261106','import pcd_kernel as m; print(m.mathematical_checks())'),
 ('empty_noise_scale','empty_scale_validation_20261106','import validate_empty_models as m; m.math_checks()'),
 ('g1_weighted','g1_chain2_weighting_20261106','import analyze_g1 as m; m.synthetic()'),
 ('independent_phase_calibration','independent_phase_calibration_20261106','import check_calibration as m; print(m.self_test())'),
 ('ratnam_synthetic','ratnam_phase_timing_validation','import test_synthetic as m; m.run()'),
 ('ratnam_extended','ratnam_phase_timing_validation','import test_extended as m; m.main()'),
 ('ratnam_fades','ratnam_phase_timing_validation','import test_fades as m; m.main()'),
 ('empty_reference_preservation','empty_reference_static_recovery_20261024','import test_preservation as m; m.main()'),
 ('baseline_and_graph','static_human_aoa_validation_20261101','import check_models as m; m.run()'),
]
def main():
 if not __debug__:raise SystemExit("Do not use Python -O: research integrity checks require assertions.")
 ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path);ap.add_argument('--only',choices=[x[0] for x in JOBS]);ap.add_argument('--timeout',type=int,default=180);a=ap.parse_args()
 syntax=[]
 for path in sorted(ROOT.rglob('*.py')):
  if any(x in path.parts for x in ['outputs','.venv','__pycache__']):continue
  ast.parse(path.read_text('utf-8-sig'));syntax.append(path.relative_to(ROOT).as_posix())
 rows=[]
 with tempfile.TemporaryDirectory(prefix='wifi-capstone-check-') as tmp:
  temp=Path(tmp);work=temp/'research';shutil.copytree(ROOT/'research',work)
  for directory in list((work/'output').rglob('*')):
   if directory.is_dir():(directory/'results').mkdir(exist_ok=True)
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WIFI_DATA_ROOT=str(temp/'no-real-data'),WIFI_TARGET_MAC='')
  for name,folder,code in JOBS:
   if a.only and a.only!=name:continue
   try:
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=work/'output'/folder,env=env,capture_output=True,text=True,encoding='utf8',timeout=a.timeout)
    row={'check':name,'passed':r.returncode==0,'exit_code':r.returncode}
    if r.returncode:row['error']=r.stderr[-2400:]
   except subprocess.TimeoutExpired:row={'check':name,'passed':False,'error':'test timeout; no physical conclusion'}
   rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 result={'syntax_files':len(syntax),'checks':rows,'all_pass':all(x['passed'] for x in rows),'physical_validation':False,'new_measurements':False}
 if a.report:
  a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['all_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
