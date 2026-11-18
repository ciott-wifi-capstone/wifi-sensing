"""Freeze protocol, inventory and recoverable source versions before estimation."""
from pathlib import Path
import hashlib,json,shutil,zipfile,csv,xml.etree.ElementTree as ET
import pandas as pd
R=Path(__file__).resolve().parent;ROOT=R.parents[1];TH=ROOT/'output/thesis_algorithm_draft_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def freeze(p,x):
 if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==x,f'Frozen file differs: {p}'
 else:dump(p,x)
def main():
 for d in ['provenance','provenance/ratnam','provenance/office','backups','results','packets','figures','review']:(R/d).mkdir(exist_ok=True,parents=True)
 assert TH.resolve().is_relative_to(ROOT.resolve())
 backup=R/'backups/local_thesis_before_20261101.zip'
 if not backup.exists():
  with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
   for p in TH.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(TH))
 with zipfile.ZipFile(backup) as z:assert z.testzip() is None
 online=R/'backups/overleaf_before_20261101.zip'
 if not online.exists():shutil.copy2(Path.home()/'Downloads/New_Draft_Thesis (5).zip',online)
 with zipfile.ZipFile(online) as z:
  assert z.testzip() is None
  rows=[]
  for n in z.namelist():
   if n.endswith('/'):continue
   local=TH/'overleaf_upload'/n
   a=z.read(n);b=local.read_bytes() if local.exists() else None
   equal=a==b
   if b is not None and n.endswith(('.tex','.bib')):equal=a.decode('utf-8-sig').replace('\r\n','\n')==b.decode('utf-8-sig').replace('\r\n','\n')
   rows.append(dict(online_file=n,online_sha256=hashlib.sha256(a).hexdigest(),local_export_exists=local.exists(),content_equal=equal))
  pd.DataFrame(rows).to_csv(R/'results/online_preedit_comparison.csv',index=False)
 sources=[(ROOT/'output/ratnam_phase_timing_validation/core.py','ratnam/core.py'),(ROOT/'output/ratnam_phase_timing_validation/config.json','ratnam/config.json'),(ROOT/'output/ratnam_phase_timing_validation/provenance/analyze.py','decoder_ht40.py'),(ROOT/'output/office_static_recovery_20261028/snapshots/decoder.py','decoder_ht20.py'),(ROOT/'output/office_static_recovery_20261028/analyze.py','office/analyze.py'),(ROOT/'output/office_static_recovery_20261028/shared_reference.py','office/shared_reference.py'),(ROOT/'output/ratnam_phase_timing_validation/input_manifest.csv','previous_input_manifest.csv'),(ROOT/'output/static_analysis_20261018/geometry_model_checks.py','geometry_model_checks.py'),(Path.home()/'.codex/attachments/c80ec31c-b6c4-4b3d-916a-0d7c406e2e2c/goal-objective.md','goal-objective.md'),(Path.home()/'.codex/attachments/c7c973b3-c218-4a53-9a06-02a31803d696/已粘贴的文本.txt','confirmed_plan.txt')]
 sr=[]
 for src,dst in sources:
  dest=R/'provenance'/dst
  if dest.exists():assert sha(dest)==sha(src)
  else:shutil.copy2(src,dest)
  sr.append(dict(source=str(src),snapshot=dst,sha256=sha(src)))
 dump(R/'provenance/source_snapshots.json',sr)
 m=pd.read_csv(R/'provenance/previous_input_manifest.csv');primary=m[m['index']<39].copy();assert len(primary)==39 and primary.sha256.nunique()==39
 dev=set([0,1,2,12,13,14,21,22,23,30,31,32]);primary['role_current']=['development' if i in dev else 'retrospective_evaluation' for i in primary['index']]
 primary['previously_processed']=True;primary['hardware_session_relation']='within_file_confirmed; between_batches_not_assumed'
 primary['bounded_stage']=['development' if i in [0,12,21,30] else 'evaluation' if i in [3,15,24,33] else 'not_scheduled' for i in primary['index']]
 primary.to_csv(R/'input_manifest.csv',index=False)
 hashes=[]
 for row in primary.itertuples():
  p=Path(row.path);h=sha(p);assert h==row.sha256
  hashes.append(dict(index=row.index,file=str(p),sha256=h,kind='raw',matches_previous=True))
  for f in ['run.json','events.log']:
   p=Path(row.path).parent/f;hashes.append(dict(index=row.index,file=str(p),sha256=sha(p),kind=f,matches_previous='not_compared'))
 of=ROOT/'office_broadside_raw6state_20260907'
 for e in json.loads((of/'delivery_manifest.json').read_text())['files']:
  p=of/e['path'];h=sha(p);assert h==e['sha256'];hashes.append(dict(index='office',file=str(p),sha256=h,kind='office',matches_previous=True))
 pd.DataFrame(hashes).to_csv(R/'results/input_integrity.csv',index=False)
 config={'version':1,'date':'2026-11-01','max_packets_per_state':768,'prefix_packets':64,'minimum_packets':128,'development':[0,12,21,30,'office_R1'],'evaluation':[3,15,24,33,'office_R2'],'never_unseen_test':True,'gate_delay_p90_ns':5.,'gate_phase_p90_deg':15.,'gate_prediction_corr':.9,'packet_correlation':.9,'prediction_stride':16,'prediction_masks':['leave_chain_1','leave_chain_2','leave_chain_3','leave_lower_frequency_block','leave_middle_frequency_block','leave_upper_frequency_block'],'time_blocks':4,'music_q_baseline':3,'music_q_sensitivity':[2,3,4],'aoa_mae_target_deg':8.,'coverage_target':.8,'all_four_points_required':True,'human_path_requires':['hardware_angle_reference_valid','human_association_valid','unambiguous_output'],'cross_record_reference':'Only use verified same-scene same-session Office R1; primary D unavailable pending session/scene validation. Never select a per-position reference by test label.','cross_state_phase_gate_p90_deg':15.,'cross_state_delay_gate_p90_ns':5.,'cross_state_prediction_corr':.9,'no_joint_array_until_all_gates_pass':True,'no_gt_in_estimators':True,'raw_baseline_preserved':True,'synthetic_weak_human_distortion_limit':.05,'geometry_source':'User confirmed global TX=(0,0), RX=(3.65,0); Office distances unknown','output_status':'diagnostic_only_until_hardware_and_human_path_gates_pass'}
 freeze(R/'config.json',config);(R/'config.sha256').write_text(sha(R/'config.json'))
 geo={'global_tx':[0,0],'global_rx':[3.65,0],'rxlocal_transform':'(3.65-X,-Y)','array_y_local_m':[(6.5-m)*.028 for m in range(1,13)],'global_array_xy_m':[[3.65,-(6.5-m)*.028] for m in range(1,13)],'state_map':{'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]},'office_tx_rx_distance_m':None,'office_human_xy':None,'old_geometry_not_overwritten':True}
 freeze(R/'geometry.json',geo)
 with zipfile.ZipFile(ROOT/'thesis_template.docx') as z:
  tree=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
  texts=[''.join(el.itertext()) for el in []]
  texts=[''.join(t.text or '' for t in para.findall('.//w:t',ns)) for para in tree.findall('.//w:p',ns)]
  (R/'review/template_text.txt').write_text('\n'.join(texts),encoding='utf-8')
 dump(R/'results/setup_audit.json',{'raw_primary_files_verified':39,'office_manifest_verified':True,'local_backup':str(backup),'local_backup_sha256':sha(backup),'online_backup':str(online),'online_backup_sha256':sha(online),'online_editing_observed':True,'target_project':'6ad575f3284122a9b7bf48d2','protocol_frozen_before_analysis':True})
 print(pd.DataFrame(rows).to_string(index=False));print('SETUP COMPLETE: 39 primary files and Office source hashes verified.')
if __name__=='__main__':main()
