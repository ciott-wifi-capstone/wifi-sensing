"""First campaign re-audit: hash raw inputs, reuse saved state-wise MUSIC.
No GT-based peak selection, phase fitting, or parameter tuning.
"""
import os
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
S=R.parent/'static_analysis_20261018/geometry_update_20261019'
T=R.parent/'thesis_algorithm_draft_20261019/evidence'
groups=['old+1','old-1','old+2','old-2'];states=['G1','G2','G3','G4','C1','C2']
dirs={
 'old+1':(Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/'1m _(1)/1m _'),
 'old-1':(Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/'1m _(1)/1m-2'),
 'old+2':(Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/'1m _(1)/45 good'),
 'old-2':(Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/'1m _(1)/45 bad'),
 'kong':(Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))/'buchong shuju/kong')}
inp=pd.read_csv(S/'inputs.csv');old=inp[inp.analysis_group.isin(groups)]
hashmap={r.sha256:(i,r.analysis_group) for i,r in inp.iterrows()}
inventory=[]
for label,root in dirs.items():
 for p in sorted(root.rglob('*.csi')):
  h=hashlib.sha256(p.read_bytes()).hexdigest();match=hashmap.get(h)
  inventory.append(dict(folder=label,path=str(p),bytes=p.stat().st_size,sha256=h,cached_index=match[0] if match else None,cached_group=match[1] if match else 'UNMATCHED',use_as_empty=label=='kong' and bool(match) and match[1].startswith('empty')))
inv=pd.DataFrame(inventory);inv.to_csv(R/'source_inventory.csv',index=False,encoding='utf-8-sig')
sessions=[]
for v in inventory:
 p=Path(v['path']).parent/'run.json';j=json.loads(p.read_text(encoding='utf8'));b=j['before']
 sessions.append(dict(group=v['cached_group'],index=v['cached_index'],run=str(p),metadata_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),boot_id=b.get('boot_id'),ifindex=b.get('ifindex'),phy=b.get('phy'),radio=b.get('radio'),identity_unchanged=j.get('identity_unchanged')))
sessions=pd.DataFrame(sessions);sessions.to_csv(R/'hardware_sessions.csv',index=False,encoding='utf-8-sig')
assert inv.cached_index.notna().all(), 'Unmatched input: do not silently reuse cached spectra'
for g in groups:assert inv[inv.folder==g].cached_group.eq(g).all()
assert len(old)==39
st=pd.read_csv(S/'state_results.csv');st=st[st.group.isin(groups+['empty Oct03','empty Oct18'])].copy()
st.to_csv(R/'run_state_peaks.csv',index=False,encoding='utf-8-sig')
summary=st.groupby(['group','state']).agg(runs=('index','size'),angle_median=('angle_search_deg','median'),angle_min=('angle_search_deg','min'),angle_max=('angle_search_deg','max'),raw_delay_median_ns=('raw_delay_ns','median'),raw_delay_min_ns=('raw_delay_ns','min'),raw_delay_max_ns=('raw_delay_ns','max'))
summary.to_csv(R/'peak_summary.csv')
qa=old.groupby('analysis_group').agg(runs=('path','size'),batches=('batch','nunique'),complete_10s=('complete_window','sum'),median_pps=('window_pps','median'))
qa.to_csv(R/'capture_quality.csv')
old[~old.complete_window].to_csv(R/'incomplete_runs.csv',index=False)
mo=pd.read_csv(S/'model_order_sensitivity.csv');mo=mo[mo.group.isin(groups)]
spread=mo.groupby(['index','group','state']).angle_deg.agg(lambda x:x.max()-x.min()).reset_index(name='angle_span_deg')
spread.to_csv(R/'model_order_spread.csv',index=False)
cl=pd.read_csv(S/'closure_per_run.csv');cl=cl[cl.group.isin(groups+['empty Oct03','empty Oct18'])];cl.to_csv(R/'reference_per_run.csv',index=False)
bg=pd.read_csv(T/'background_diagnostics.csv');bg=bg[bg.group.isin(groups+['empty Oct03','empty Oct18'])];bg.to_csv(R/'background_existing.csv',index=False)
geometry=[]
for g,y in zip(groups,[1.,-1.,2.,-2.]):
 x=1.64;rxrange=np.hypot(3.65-x,y);txrange=np.hypot(x,y);length=rxrange+txrange
 geometry.append(dict(group=g,global_x=x,global_y=y,rx_local_x=3.65-x,rx_local_y=-y,rx_angle_deg=np.rad2deg(np.arctan2(-y,3.65-x)),rx_range_m=rxrange,total_path_m=length,total_delay_ns=length/299792458*1e9,excess_length_m=length-3.65,excess_delay_ns=(length-3.65)/299792458*1e9))
geo=pd.DataFrame(geometry);geo.to_csv(R/'corrected_theory.csv',index=False)
(R/'geometry_correction.json').write_text(json.dumps(dict(confirmed='2026-10-20 user: TX on left, RX on right',global_TX=[0,0],global_RX=[3.65,0],points='retain supplied drawing x=1.64, y=+/-1,+/-2',local_transform='x_rx=3.65-x_global; y_rx=-y_global',orientation_assumption='retain earlier confirmation: numbering 1->12 left-to-right viewed from behind RX facing TX; positive local angle towards observer left',note='saved spectra use local array geometry; unchanged numerical spectra are reinterpreted in corrected global coordinates, not relabelled as calibrated estimates',mapping='45 good=+2 and 45 bad=-2 retained from prior explicit user mapping'),indent=2),encoding='utf8')
spec=np.load(S/'spectra.npz');ang=spec['angle_deg'];tau=spec['delay_ns']
def median_spec(g,state):
 arr=[spec[f'{i:03d}_{state}'] for i in inp[inp.analysis_group==g].index if f'{i:03d}_{state}' in spec]
 p=np.median(arr,axis=0);return 10*np.log10(np.maximum(p/p.max(),1e-6))
fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for ax,g in zip(axs.flat,groups+['empty Oct03','empty Oct18']):
 im=ax.pcolormesh(tau,ang,median_spec(g,'G1'),cmap='viridis',vmin=-15,vmax=0,shading='auto',rasterized=True)
 ax.set(title=g+' | G1',xlabel='Uncalibrated delay (ns)',ylabel='Local angle parameter (deg)')
fig.colorbar(im,ax=axs,label='Per-run normalised, median MUSIC (dB)',shrink=.7)
fig.suptitle('State-wise 2D MUSIC: first campaign and separate-date empty references\nNo truth-selected peaks; local angle axis, not calibrated human AoA',fontsize=13)
fig.savefig(R/'g1_spectra.png',dpi=170);fig.savefig(R/'g1_spectra.pdf');plt.close(fig)
fig,axs=plt.subplots(4,6,figsize=(18,12),layout='constrained')
for row,g in zip(axs,groups):
 for ax,state in zip(row,states):
  im=ax.pcolormesh(tau,ang,median_spec(g,state),vmin=-15,vmax=0,cmap='viridis',shading='auto',rasterized=True)
  ax.set(title=f'{g}: {state}',xticks=[-400,0,400],yticks=[-90,0,90])
fig.colorbar(im,ax=axs,label='Normalised MUSIC (dB)',shrink=.5)
fig.supxlabel('Uncalibrated delay (ns)');fig.supylabel('Local angle parameter (deg)')
fig.suptitle('All 39 occupied runs retained; six states processed separately; q=3')
fig.savefig(R/'all_states.png',dpi=150);fig.savefig(R/'all_states.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for j,g in enumerate(groups):
 sub=st[(st.group==g)&st.state.str.startswith('G')]
 for k,state in enumerate(states[:4]):
  v=sub[sub.state==state];axs[0].scatter(np.full(len(v),j)+(k-1.5)*.16,v.angle_search_deg,s=22,color=f'C{k}',label=state if j==0 else None,alpha=.75)
 axs[0].plot(j,geo.iloc[j].rx_angle_deg,'kx',ms=11,mew=2)
axs[0].set(xticks=range(4),xticklabels=groups,ylabel='Local angle (deg)',title='Raw strongest peaks; black x = geometry only')
axs[0].legend(ncols=4,fontsize=8);axs[0].grid(axis='y',alpha=.2)
gg=groups+['empty Oct03','empty Oct18']
for j,g in enumerate(gg):
 v=bg[(bg.group==g)&(bg.role!='background_fit')].whitened_shape
 axs[1].scatter(np.full(len(v),j),v,s=12,alpha=.4)
axs[1].set(xticks=range(6),xticklabels=gg,ylabel='Whitened covariance shape deviation',title='Date-matched background diagnostic; not detections');axs[1].tick_params(axis='x',rotation=30)
fig.savefig(R/'peak_and_background.png',dpi=170);fig.savefig(R/'peak_and_background.pdf');plt.close(fig)
audit=dict(raw_files=len(inv),all_hashes_match_cached=True,occupied_files=len(old),kong_files=int((inv.folder=='kong').sum()),kong_true_empty=int(inv.use_as_empty.sum()),kong_occupied_A=int(((inv.folder=='kong')&(inv.cached_group=='A')).sum()),old_state_entries=int(st.group.isin(groups).sum()),old_state_status=st[st.group.isin(groups)].status.value_counts().to_dict(),old_model_order_cases=len(mo),old_model_order_span_median=float(spread.angle_span_deg.median()),old_model_order_over15_fraction=float((spread.angle_span_deg>15).mean()),quality=qa.reset_index().to_dict('records'),theory=geometry,reference=cl.groupby('group')[['residual','coherence']].median().reset_index().to_dict('records'),background=bg[bg.role!='background_fit'].groupby('group').whitened_shape.median().to_dict(),scope='fresh raw-file hashes + recomputation of summaries and figures from matching saved spectra/covariances; no new calibrated estimator; no GT-driven fitting')
(R/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
sources=[]
for base,names in [(S,['inputs.csv','state_results.csv','spectra.npz','model_order_sensitivity.csv','closure_per_run.csv']),(T,['background_diagnostics.csv','background_split.csv'])]:
 for name in names:
  p=base/name;sources.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
pd.DataFrame(sources).to_csv(R/'cached_result_sources.csv',index=False)
print(json.dumps(audit,indent=2))
