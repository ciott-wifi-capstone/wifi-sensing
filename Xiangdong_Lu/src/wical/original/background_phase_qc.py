#!/usr/bin/env python3
"""WiCAL HT40 empty-room cross-session phase-reference feasibility study.

NOT an absolute hardware phase calibration, not person-path isolation, not location estimation.
Read-only ZIP input. Never use A-D ground truth for fitting.

Usage: python background_phase_qc.py --zip '/path/to/buchong shuju-...zip' --out output_dir
Requires Python 3, numpy, matplotlib and per_subcarrier_calibration.py alongside this file.
"""
import argparse, csv, hashlib, json, re, sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
if __package__:
    from .per_subcarrier_calibration import MAP, EDGES, phase_r, solve, wrap
else:
    from per_subcarrier_calibration import MAP, EDGES, phase_r, solve, wrap

PAIRS = [(1,0),(2,1),(2,0)]
STATES = ('G1','G2','G3','G4','C1','C2')
BIT = (np.arange(30)[:,None]*51+3+np.arange(3)[None,:]*16).ravel()

def decode(raw):
    """PicoScenes fixed-size Intel 5300 CSI:v3 records; 30x3 int8 complex CSI."""
    payload=raw[:,168:360]
    def bits(pos):
        pos=np.asarray(pos); idx=pos//8; shift=pos%8
        out=((payload[:,idx].astype(np.uint16)|(payload[:,idx+1].astype(np.uint16)<<8))>>shift)&255
        return out.astype(np.uint8).view(np.int8).astype(np.float32)
    c=(bits(BIT)+1j*bits(BIT+8)).reshape(-1,30,3)
    ant=raw[:,161]
    perms=np.stack([ant&3,(ant>>2)&3,(ant>>4)&3],axis=1)
    valid=np.all(np.sort(perms,axis=1)==np.arange(3),axis=1)
    result=np.zeros_like(c)
    for rx in range(3):
        result[np.flatnonzero(valid),:,perms[valid,rx]] = c[valid,:,rx]
    return result,valid

def events(z,pref):
    s=z.read(pref+'events.log').decode('utf-8')
    records=[]
    for ln in s.splitlines():
        m=re.match(r'(\d+\.\d+)\s+READY\s+(G\d|C\d)\s+chains=.*\s+t=(\d+)',ln)
        if m: records.append((int(round(float(m[1])*1e9)),m[2],int(m[3])))
    if len(records)<8: raise ValueError('Insufficient switch events '+pref)
    host=np.array([r[0] for r in records],np.int64)
    dev=np.array([r[2] for r in records],np.int64)
    offset=int(np.median(host-dev*1000))
    starts=dev*1000+offset
    jitter_ms=float(np.std((host-starts)/1e6))
    return starts,np.array([r[1] for r in records]),jitter_ms

def read_run(z,sess,run,step=10):
    pref=sess+f'/capture/{run}/'
    csi=next((f for f in z.namelist() if f.startswith(pref) and f.endswith('.csi')),None)
    if csi is None: raise ValueError('Missing CSI '+pref)
    data=z.read(csi)
    if len(data)%385: raise ValueError('Unexpected record size '+csi)
    raw=np.frombuffer(data,dtype=np.uint8).reshape(-1,385)
    if not np.all(raw[::max(1,len(raw)//100),0]==125): raise ValueError('Unexpected record format '+csi)
    cfg=json.loads(z.read(pref+'run.json'))
    t=raw[:,37:45].copy().view('<u8').reshape(-1).astype(np.int64)
    starts,states,jitter=events(z,pref)
    ids=np.searchsorted(starts,t,side='right')-1
    valid=(ids>=0)&(ids<len(starts)-1)&(t>=cfg['window']['start_ns'])&(t<cfg['window']['end_ns'])
    ms=np.zeros(len(t),float); ix=np.flatnonzero(valid)
    ms[ix]=(t[ix]-starts[ids[ix]])/1e6
    valid &= (ms>=3)&(ms<=21)
    n_by_state={s:int(np.sum(valid & (states[np.maximum(ids,0)]==s))) for s in STATES}
    selected=np.flatnonzero(valid)[::step]
    h,permvalid=decode(raw[selected]); selected=selected[permvalid]; h=h[permvalid]
    gs=states[ids[selected]]
    zsum=np.zeros((18,30),np.complex128); counts=np.zeros((18,30),np.int64)
    for e,(s,a,b) in enumerate(EDGES):
        c=h[gs==s]; aa=c[:,:,MAP[s].index(a)]; bb=c[:,:,MAP[s].index(b)]
        good=(np.abs(aa)>0)&(np.abs(bb)>0)
        u=np.zeros(aa.shape,np.complex128)
        u[good]=(aa[good]/np.abs(aa[good]))*np.conj(bb[good]/np.abs(bb[good]))
        zsum[e]=np.sum(u,axis=0); counts[e]=np.sum(good,axis=0)
    return dict(z=zsum,n=counts, counts=n_by_state, jitter=jitter,
                raw_frames=len(raw),analyzed_packets=len(selected),duration=cfg['requested_seconds'])

def aggregate(v):
    return (sum((r['z'] for r in v),np.zeros((18,30),complex)),
            sum((r['n'] for r in v),np.zeros((18,30),int)))

def error_summary(train,hold):
    tz,tn=aggregate(train); ph,r=phase_r(tz,tn); offsets,diag=solve(ph,r,tn)
    hz,hn=aggregate(hold); hp,hr=phase_r(hz,hn)
    rec=[]
    for e,(s,a,b) in enumerate(EDGES):
        valid=(hn[e]>=30)&(hr[e]>=.5)&np.isfinite(hp[e])&np.isfinite(offsets[a-1])&np.isfinite(offsets[b-1])
        res=wrap(hp[e]-offsets[a-1]+offsets[b-1]); good=np.abs(res[valid])
        rec.append(dict(state=s,pair=f'{a}-{b}',n_subcarriers=int(valid.sum()),mean_abs_residual_deg=float(good.mean()) if len(good) else None,
                        p95_abs_residual_deg=float(np.percentile(good,95)) if len(good) else None))
    return offsets,rec

def savecsv(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True,type=Path); ap.add_argument('--out',required=True,type=Path); ap.add_argument('--step',type=int,default=10)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    z=ZipFile(args.zip); names=z.namelist()
    sessions=sorted({'/'.join(n.split('/')[:3]) for n in names
       if re.fullmatch(r'buchong shuju/(A|B|C|D|kong)/[^/]+/capture/run1/run.json',n)})
    assert len(sessions)==17, ('Unexpected session count',len(sessions))
    all_data={}; runs=[]
    for i,sess in enumerate(sessions):
        records=[]; label=sess.split('/')[1]
        for run in ('run1','run2','run3'):
            v=read_run(z,sess,run,args.step); records.append(v)
            runs.append(dict(point=label,session=sess.split('/')[-1],run=run,duration_s=v['duration'],total_frames=v['raw_frames'],sampled_frames=v['analyzed_packets'],host_switch_jitter_ms=round(v['jitter'],3),**{f'{s}_valid_packets':v['counts'][s] for s in STATES}))
        all_data[sess]=records
        print(f'{i+1}/{len(sessions)} {label}: {sess.split("/")[-1]}',flush=True)
    savecsv(args.out/'run_quality.csv',runs)
    background=[s for s in sessions if s.split('/')[1]=='kong']
    assert len(background)==5
    checks=[]; pergroup=[]; within=[]
    for held in background:
        train=[r for s in background if s!=held for r in all_data[s]]
        offsets,edges=error_summary(train,all_data[held])
        for v in edges: checks.append(dict(heldout=held.split('/')[-1],**v))
        for st in STATES:
            rr=[v for v in edges if v['state']==st and v['mean_abs_residual_deg'] is not None]
            pergroup.append(dict(heldout=held.split('/')[-1],state=st,available_pairs=len(rr),valid_subcarrier_edges=sum(x['n_subcarriers'] for x in rr),mean_abs_phase_residual_deg=float(np.mean([x['mean_abs_residual_deg'] for x in rr])) if rr else None))
        # Within this one empty-room recording batch, train only Run1, hold out Run2/3.
        # The date-prefix label is not used to define separate experimental days.
        session_runs=all_data[held]
        off_session,_=error_summary([session_runs[0]],[session_runs[1],session_runs[2]])
        _,one_edges=error_summary([session_runs[0]],[session_runs[1],session_runs[2]])
        vals=[e['mean_abs_residual_deg'] for e in one_edges if e['state'].startswith('G') and e['mean_abs_residual_deg'] is not None]
        within.append(dict(session=held.split('/')[-1],train_run='run1',heldout_runs='run2,run3',finite_offsets=int(np.isfinite(off_session).sum()),testable_pair_tones=sum(e['n_subcarriers'] for e in one_edges if e['state'].startswith('G')),mean_abs_phase_residual_deg=float(np.mean(vals)) if vals else None))
    savecsv(args.out/'holdout_background_pair_residual.csv',checks)
    savecsv(args.out/'holdout_background_summary.csv',pergroup)
    savecsv(args.out/'within_background_session_validation.csv',within)
    fullz,fulln=aggregate([r for s in background for r in all_data[s]]); ph,r=phase_r(fullz,fulln)
    final_offsets,train_diag=solve(ph,r,fulln)
    valid_count=int(np.isfinite(final_offsets).sum()); held=[x['mean_abs_phase_residual_deg'] for x in pergroup if x['state'].startswith('G') and x['mean_abs_phase_residual_deg'] is not None]
    passed_coverage=sum(x['valid_subcarrier_edges'] for x in pergroup if x['state'].startswith('G')) >= 5*4*3*30*.8
    ok=valid_count==360 and passed_coverage and len(held)==20 and max(held)<=5
    with open(args.out/'exploratory_background_offsets.json','w') as f:
        json.dump(dict(status='EXPLORATORY / NOT A VERIFIED HARDWARE CALIBRATION',reference_hypothesis='Tx-Rx direct path AoA 0 deg in empty scene; dominant observed phase might contain unresolved multipath',derived_from='kong only, five background sessions',freq_center_MHz=5230,
              relative_offsets_deg=[[float(x) if np.isfinite(x) else None for x in row] for row in final_offsets],
              test_all_antennas_subcarriers_identifiable=valid_count==360,
              finite_offsets=valid_count,total_offsets=360,cross_session_pass_under_strict_exploratory_5deg_check=ok,
              note='Human A-D positions not used to train. Never treat these offsets as proven direct-path phase; no person AoA is produced.'),f,indent=2,allow_nan=True)
    # Avoid creating a misleading AoA accuracy table: only write QC and background cross-session residuals.
    summary=dict(independent_sessions=len(sessions),unique_runs=len(runs),kong_sessions=len(background),kong_runs=15,
        all_kong_sessions_included_despite_mislabeled_dates=True,
        background_0deg_reference_geometry='Tx=(3.65,0); Rx=(0,0); broadside along +x',
        finite_phase_offsets=valid_count,total_offsets=360,
        background_heldout_valid_pair_subcarrier_bins=sum(x['valid_subcarrier_edges'] for x in pergroup if x['state'].startswith('G')),
        background_heldout_total_pair_subcarrier_bins=5*4*3*30,
        background_cross_session_mean_abs_phase_residual_deg_on_testable_subset=float(np.mean(held)) if held else None,
        background_cross_session_max_group_mean_abs_phase_residual_deg=float(max(held)) if held else None,
        within_session_run1_calibration_checks=within,
        preliminary_strict_5deg_stability_gate_passed=ok,
        warning='Residual is PHASE DEGREES, not AoA accuracy. No person echo isolation or person AoA confirmed. Background-derived offsets mix hardware and propagation effects.')
    (args.out/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    try:
        import matplotlib
    except ImportError:
        print('matplotlib unavailable: numerical outputs completed; extended analysis generates SVG figures.')
        print(json.dumps(summary,indent=2,ensure_ascii=False))
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    labels=['A','B','C','D','kong']
    for g in ('G1','G2','G3','G4'):
        axs[0].plot(labels,[np.mean([rr[f'{g}_valid_packets'] for rr in runs if rr['point']==s]) for s in labels],marker='o',label=g)
    axs[0].set_title('Safe-window CSI packets per run'); axs[0].set_ylabel('Mean packets/run'); axs[0].legend()
    for g in ('G1','G2','G3','G4'):
        v=[e['mean_abs_phase_residual_deg'] for e in pergroup if e['state']==g]
        axs[1].plot(range(1,6),v,marker='o',label=g)
    axs[1].axhline(5,ls='--',color='gray',label='exploratory 5° phase QC')
    axs[1].set_xticks(range(1,6)); axs[1].set_xlabel('Held-out empty-room session'); axs[1].set_ylabel('Mean abs residual PHASE (deg)'); axs[1].set_title('Leave-one-empty-session-out phase check'); axs[1].legend(fontsize=8)
    fig.suptitle('CSI QC and background reference consistency — NOT person AoA / position')
    fig.tight_layout(); fig.savefig(args.out/'qc_background_cross_session.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.3))
    vals=[e['mean_abs_phase_residual_deg'] for e in within]
    ax.bar(range(1,6),vals)
    ax.axhline(5,ls='--',color='gray',label='5° exploratory reference threshold')
    ax.set_xticks(range(1,6),[f'K{i+1}' for i in range(5)])
    ax.set_ylabel('Mean absolute relative-PHASE residual (deg)')
    ax.set_title('Within empty-room batch: Run1 fit; Runs2+3 checks')
    for i,e in enumerate(within,1):ax.text(i,e['mean_abs_phase_residual_deg']+.15,f"{e['mean_abs_phase_residual_deg']:.1f}° / {e['finite_offsets']}/360",ha='center',fontsize=9)
    ax.set_ylim(0,max(vals)*1.25);ax.legend(loc='upper left')
    fig.tight_layout();fig.savefig(args.out/'within_session_phase_check.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
