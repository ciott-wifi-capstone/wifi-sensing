from audit import *
def main():
 selected=[]
 for i in range(78,90):
  q=MAN[MAN['index']==i].iloc[0]
  selected.append((i,q.capture,Path(q.raw_path),'empty'))
 mm=pd.read_csv(O/'static_human_aoa_validation_20261101/input_manifest.csv')
 for q in mm[mm['index'].isin([0,1,2,3,12,13,14,15,21,22,23,24,30,31,32,33])].itertuples():
  selected.append((q.index,q.label,Path(q.path),'occupied'))
 rows=[];evidence=[];sources=[];checks=[];excerpt=[]
 for i,label,p,role in selected:
  batch=p.parents[2];ops=batch/'operations.log';meta=dec.read(p.parent/'run.json')
  man=json.loads((batch/'manifest.json').read_text(encoding='utf8'))
  expected={v['path']:v for v in man['files']}
  relevant=[ops,batch/'rx_kernel.log',batch/'radio_prepared.json',p.parent/'run.json',p,batch/'runtime_before.json']
  for f in relevant:
   hh=hashfile(f);entry=expected.get(f.relative_to(batch).as_posix())
   ok=entry is not None and entry['sha256']==hh and entry['bytes']==f.stat().st_size
   assert ok,f
   sources.append(dict(index=i,path=str(f),sha256=hh,manifest_verified=ok))
  lines=ops.read_text().splitlines();commands=[]
  for n,line in enumerate(lines,1):
   if line.startswith('{"command"'):
    try:c=json.loads(line)
    except json.JSONDecodeError:continue
    if c['command'][0] in ['rmmod','modprobe','array_prepare_for_picoscenes']:
     commands.append(c);evidence.append(dict(index=i,label=label,source=str(ops),line=n,command=' '.join(c['command']),returncode=c['returncode']))
  reloadok=all(any(c['command'][:2]==cmd and c['returncode']==0 for c in commands) for cmd in [['rmmod','iwldvm'],['rmmod','iwlwifi'],['modprobe','iwlwifi'],['modprobe','iwldvm']])
  kern=(batch/'rx_kernel.log').read_text()
  fw=re.search(r'^\d+,\d+,(\d+),[^;]*;.*loaded firmware version ([^\n]+)',kern,re.M)
  stream=re.search(r'^\d+,\d+,(\d+),[^;]*;.*RxS data starts streaming',kern,re.M)
  before=meta['before'];phy=before['phy'].split('/')[-1]
  rows.append(dict(index=i,label=label,role=role,batch=batch.name,run=meta['label'],driver_epoch=before['boot_id']+'_'+phy+'_if'+before['ifindex'],boot_id=before['boot_id'],phy=phy,ifindex=before['ifindex'],successful_batch_driver_reload=reloadok,within_run_identity_unchanged=meta['identity_unchanged'],firmware_loaded=fw[2] if fw else '',firmware_load_kernel_us=int(fw[1]) if fw else np.nan,first_batch_stream_kernel_us=int(stream[1]) if stream else np.nan,run_metadata_scope=meta['radio_initialization']))
 df=pd.DataFrame(rows);df.to_csv(R/'revised_session_manifest.csv',index=False)
 pd.DataFrame(evidence).to_csv(R/'driver_reload_evidence.csv',index=False)
 pd.DataFrame(sources).drop_duplicates(['path']).to_csv(R/'batch_provenance_hashes.csv',index=False)
 ep=df[df.role=='empty'].groupby('label').driver_epoch.nunique()
 assert (ep==1).all()
 assert df.driver_epoch.nunique()==12
 assert df.successful_batch_driver_reload.all()
 e1=df[(df.label=='E1')&(df.run=='run1')].iloc[0]
 e3=df[(df.label=='E3')&(df.run=='run1')].iloc[0]
 mt=pd.read_csv(R/'metadata_audit.csv')
 t1=mt.query('capture=="E1" and run=="run1"').iloc[0]
 t3=mt.query('capture=="E3" and run=="run1"').iloc[0]
 comparison=dict(same_OS_boot_id=e1.boot_id==e3.boot_id,driver_epoch_same=e1.driver_epoch==e3.driver_epoch,first_stream_kernel_gap_s=float((e3.first_batch_stream_kernel_us-e1.first_batch_stream_kernel_us)/1e6),first_ready_ESP_gap_s=float((t3.esp_first_us-t1.esp_first_us)/1e6),source='per-batch operations plus kernel log; filename calendar dates not used',interpretation='same reported scene and physical hardware; different evidenced receiver driver-initialization epochs, not an uninterrupted RF calibration session')
 (R/'E1_E3_session_evidence.json').write_text(json.dumps(comparison,indent=2))
 note="""# Evidence update: automatic driver reload before each batch

The user-reported facts remain: same scene, no manual reboot, no rewiring/movement. The filename dates are not reliable chronology.
New evidence from manifest-verified operations.log and rx_kernel.log shows successful iwldvm/iwlwifi unload+reload at batch preparation, firmware loading, and new monitor interfaces. run.json's 'none; logger restarts only' refers to within-batch run acquisition, not batch preparation.
Therefore E1/E3 cannot be assumed to share an uninterrupted receiver phase/calibration epoch. This does not establish that driver reload explains all amplitude, channel or phase differences. Controls E1/E2 and E3/E4 also reload, yet finite pi correction transfers much better.
No raw log has been edited; revised_session_manifest.csv distinguishes OS boot, receiver-driver epoch, scene/user report, and actual validated transfer.
"""
 (R/'SESSION_EVIDENCE_UPDATE.md').write_text(note,encoding='utf8')
 print(df[['index','label','run','phy','ifindex','successful_batch_driver_reload']].to_string(index=False))
 print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
