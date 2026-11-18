"""Reproducible WiCAL static-scene diagnostics. Python 3 + numpy only.
No target coordinates are used in fitting. No angular/position estimate is fabricated.
Run: python extended_analysis.py --zip RAW.zip --out RESULTS
"""
import argparse, csv, hashlib, json, re, itertools
from pathlib import Path
from zipfile import ZipFile
from decimal import Decimal
import numpy as np
if __package__:
    from .background_phase_qc import decode, STATES, error_summary, aggregate
    from .per_subcarrier_calibration import MAP, EDGES, phase_r, wrap, solve, clean
else:
    from background_phase_qc import decode, STATES, error_summary, aggregate
    from per_subcarrier_calibration import MAP, EDGES, phase_r, wrap, solve, clean

def savecsv(p, rows):
    if not rows: return
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def savejson(p, data): p.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def meanabs(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if len(x) else np.nan

def scalar_decode(frame):
    """Independent bit-by-bit implementation of Intel read_bfee.c loop."""
    payload=frame[168:360]; vals=np.zeros((30,3),complex); bit=0
    for k in range(30):
        bit+=3
        for j in range(3):
            out=[]
            for shift in (0,8):
                v=sum(int((payload[(bit+shift+l)//8]>>((bit+shift+l)%8))&1)<<l for l in range(8))
                out.append(v if v<128 else v-256)
            vals[k,j]=complex(*out); bit+=16
    perm=[int(frame[161]>>s)&3 for s in (0,2,4)]
    result=np.zeros_like(vals)
    for j in range(3): result[:,perm[j]]=vals[:,j]
    return result

def read(z,sess,run):
    pref=sess+'/capture/'+run+'/'
    name=next(n for n in z.namelist() if n.startswith(pref) and n.endswith('.csi'))
    b=z.read(name); assert len(b)%385==0
    raw=np.frombuffer(b,np.uint8).reshape(-1,385)
    # Validate every fixed-layout frame, not only the first byte of a sample.
    assert np.all(raw[:,:4]==np.array([125,1,0,0]))
    assert np.all(raw[:,4:11]==np.array([21,3,21,32,1,0,3]))
    for start,word in [(16,b'RxSBasic\x00'),(72,b'ExtraInfo\x00'),(123,b'CSI\x00')]:
        assert np.all(raw[:,start:start+len(word)]==np.frombuffer(word,np.uint8))
    assert np.all(raw[:,25:27]==[4,0]) and np.all(raw[:,127:129]==[3,0])
    assert np.all(raw[:,129:134]==[0,83,1,40,0])
    assert np.all(raw[:,154:159]==[30,0,1,3,0])
    assert np.all(raw[:,164:168]==[192,0,0,0])
    h,ok=decode(raw)
    for i in np.linspace(0,len(raw)-1,3).astype(int):
        if ok[i]: assert np.array_equal(h[i],scalar_decode(raw[i]))
    cfg=json.loads(z.read(pref+'run.json'))
    meta=json.loads(z.read(pref+'metadata_qc.json'))
    assert not any(meta['mismatches'].values())
    t=raw[:,37:45].copy().view('<u8').ravel().astype(np.int64)
    assert np.all(np.diff(t)>0)
    rec=[]
    for ln in z.read(pref+'events.log').decode().splitlines():
        m=re.match(r'(\d+\.\d+)\s+READY\s+(G\d|C\d)\s+chains=([\d,]+)\s+t=(\d+)',ln)
        if m:
            assert list(map(int,m[3].split(',')))==MAP[m[2]]
            rec.append((int(Decimal(m[1])*10**9),m[2],int(m[4])))
    host=np.array([r[0] for r in rec],np.int64); dev=np.array([r[2] for r in rec],np.int64)*1000
    offset=int(np.median(host-dev)); starts=dev+offset
    resid=(host-starts)/1e6
    # OLS slope is a timing diagnostic only; never used to optimize person results.
    slope=float(np.polyfit((dev-dev[0])/1e9,resid,1)[0])
    boot=cfg['before']['boot_id']
    q=dict(point=sess.split('/')[1],session=sess.split('/')[-1],run=run,total_frames=len(raw),
           invalid_permutation=int((~ok).sum()),host_event_residual_std_ms=float(resid.std()),
           host_event_residual_span_ms=float(np.ptp(resid)),host_event_slope_ms_per_s=slope,
           actual_pps=cfg['host_pps'],boot_id=boot,ifindex=cfg['before']['ifindex'],
           csi_sha256=sha(b),csi_file=name)
    return dict(h=h,t=t,ok=ok,starts=starts,states=np.array([r[1] for r in rec]),cfg=cfg,q=q)

def statistics(d,lo=3,hi=21,shift=0,full=False):
    starts=d['starts']+int(shift*1e6); t=d['t']
    ids=np.searchsorted(starts,t,side='right')-1; safe=np.clip(ids,0,len(starts)-1)
    age=(t-starts[safe])/1e6
    valid=(ids>=0)&(ids<len(starts)-1)&d['ok']&(age>=lo)&(age<=hi)
    valid&=(t>=d['cfg']['window']['start_ns'])&(t<d['cfg']['window']['end_ns'])
    gs=d['states'][safe]
    zz=np.zeros((18,30),complex); nn=np.zeros((18,30),int)
    cov=np.full((6,30,3,3),np.nan+0j); counts={}
    for si,s in enumerate(STATES):
        hh=d['h'][valid&(gs==s)]; counts[s]=len(hh)
        power=np.sum(abs(hh)**2,axis=2)
        v=np.divide(hh,np.sqrt(power[:,:,None]),out=np.zeros_like(hh),where=power[:,:,None]>0)
        if len(hh):
            cov[si]=np.einsum('nka,nkb->kab',v,v.conj())/np.maximum((power>0).sum(axis=0),1)[:,None,None]
        for e,(st,a,b) in enumerate(EDGES):
            if st!=s: continue
            aa=hh[:,:,MAP[s].index(a)]; bb=hh[:,:,MAP[s].index(b)]
            good=(abs(aa)>0)&(abs(bb)>0)
            unit=np.divide(aa*bb.conj(),abs(aa)*abs(bb),out=np.zeros_like(aa),where=good)
            zz[e]=unit.sum(axis=0); nn[e]=good.sum(axis=0)
    result=dict(z=zz,n=nn,counts=counts,cov=cov,q=d['q'])
    adjacent=[]
    if full:
        e1=EDGES.index(('C1',6,8)); e2=EDGES.index(('C2',6,8))
        cache={}
        for j,s in enumerate(d['states']):
            if s not in ('C1','C2'):continue
            hh=d['h'][valid&(ids==j)]
            if not len(hh):continue
            aa=hh[:,:,2]; bb=hh[:,:,1]; good=(abs(aa)>0)&(abs(bb)>0)
            u=np.divide(aa*bb.conj(),abs(aa)*abs(bb),out=np.zeros_like(aa),where=good)
            p,r=phase_r(u.sum(axis=0),good.sum(axis=0))
            cache[j]=(p,r,good.sum(axis=0))
        for j in cache:
            if d['states'][j]!='C1' or j+1 not in cache or d['states'][j+1]!='C2':continue
            p1,r1,n1=cache[j]; p2,r2,n2=cache[j+1]
            mask=(n1>=3)&(n2>=3)&(r1>=.5)&(r2>=.5)
            for k in np.flatnonzero(mask):
                adjacent.append(dict(**{x:d['q'][x] for x in ('point','session','run')},occurrence=j,tone=k,
                                     C2_minus_C1_phase_deg=float(wrap(p2[k]-p1[k])),R_C1=r1[k],R_C2=r2[k]))
    return result,adjacent

def distance(a,b):
    return float(np.sqrt(np.nanmean(np.sum(abs(a-b)**2,axis=(-2,-1)))))

def svg_heatmap(path, title, matrix, rowlabels, collabels, vmax=180, footer='NA = insufficient usable observations. Phase degrees are not AoA error.'):
    from html import escape
    a=np.asarray(matrix); cw=65; ch=25; left=245; top=75; width=max(800,left+cw*a.shape[1]+25); height=top+ch*a.shape[0]+65
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',f'<text x="20" y="26" font-family="Arial" font-size="17">{escape(title)}</text>']
    for j,label in enumerate(collabels): parts.append(f'<text x="{left+j*cw+8}" y="60" font-family="Arial" font-size="12">{escape(str(label))}</text>')
    for i,label in enumerate(rowlabels):
        parts.append(f'<text x="10" y="{top+i*ch+17}" font-family="Arial" font-size="11">{escape(str(label))}</text>')
        for j,value in enumerate(a[i]):
            v=min(1,max(0,float(value)/vmax)) if np.isfinite(value) else None
            color=f'rgb(255,{int(245-170*v)},{int(235-195*v)})' if v is not None else '#ddd'
            x=left+j*cw;y=top+i*ch
            parts.append(f'<rect x="{x}" y="{y}" width="{cw-1}" height="{ch-1}" fill="{color}"/>')
            label=f'{value:.1f}' if np.isfinite(value) else 'NA'
            parts.append(f'<text x="{x+6}" y="{y+17}" font-family="Arial" font-size="12">{label}</text>')
    parts.append('<text x="20" y="'+str(height-20)+'" font-family="Arial" font-size="12">'+escape(footer)+'</text></svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    z=ZipFile(args.zip);names=z.namelist()
    sessions=sorted({'/'.join(n.split('/')[:3]) for n in names if re.fullmatch(r'buchong shuju/(A|B|C|D|kong)/[^/]+/capture/run1/run.json',n)})
    assert len(sessions)==17
    # Audit all per-session manifest entries and identify nested duplicate captures.
    manifest=[]
    for sess in sessions:
        m=json.loads(z.read(sess+'/manifest.json'))
        for entry in m['files']:
            n=sess+'/'+entry['path'];b=z.read(n)
            manifest.append(dict(session=sess,file=entry['path'],size_ok=len(b)==entry['bytes'],sha256_ok=sha(b)==entry['sha256']))
    savecsv(out/'manifest_verification.csv',manifest)
    duplicate=[]; known={}
    for n in names:
        if n.endswith('.csi'):
            digest=sha(z.read(n));duplicate.append(dict(file=n,sha256=digest,duplicate_of=known.get(digest,'')));known.setdefault(digest,n)
    savecsv(out/'csi_duplicate_audit.csv',duplicate)
    all_data={};qc=[];adj=[];tone=[];sensitivity=[];covrows=[]
    for index,sess in enumerate(sessions):
        all_data[sess]=[]
        for run in ('run1','run2','run3'):
            d=read(z,sess,run);v,aa=statistics(d,full=True);adj+=aa;all_data[sess].append(v)
            qc.append(dict(**d['q'],**{s+'_valid_packets':v['counts'][s] for s in STATES}))
            p,r=phase_r(v['z'],v['n'])
            for e,(s,a,b) in enumerate(EDGES):
                for k in range(30):tone.append(dict(**{x:d['q'][x] for x in ('point','session','run')},state=s,pair=f'{a}-{b}',tone=k,phase_deg=p[e,k],R=r[e,k],n=v['n'][e,k]))
            for lo,hi,shift in [(3,21,0),(6,18,0),(9,15,0),(3,21,-2),(3,21,2)]:
                vv=v if (lo,hi,shift)==(3,21,0) else statistics(d,lo,hi,shift)[0]
                pp,rr=phase_r(vv['z'],vv['n'])
                for s in STATES:
                    es=[e for e,x in enumerate(EDGES) if x[0]==s];mask=(vv['n'][es]>=30)&(v['n'][es]>=30)&(rr[es]>=.5)&(r[es]>=.5)
                    delta=wrap(pp[es]-p[es]);values=abs(delta[mask])
                    sensitivity.append(dict(**{x:d['q'][x] for x in ('point','session','run')},state=s,window_lo_ms=lo,window_hi_ms=hi,anchor_shift_ms=shift,valid_packets=vv['counts'][s],comparable_pair_tones=int(mask.sum()),mean_abs_change_phase_deg=float(values.mean()) if len(values) else np.nan))
        print(f'{index+1}/17 full-packet analysis {sess}',flush=True)
    savecsv(out/'run_quality_full.csv',qc);savecsv(out/'edge_tone_statistics.csv',tone);savecsv(out/'shared_pair_adjacent.csv',adj);savecsv(out/'timing_sensitivity.csv',sensitivity)
    shared=[]
    for q in qc:
        vals=np.array([x['C2_minus_C1_phase_deg'] for x in adj if x['session']==q['session'] and x['run']==q['run']])
        shared.append(dict(**{x:q[x] for x in ('point','session','run')},testable_adjacent_tones=len(vals),mean_abs_phase_deg=meanabs(vals),p95_abs_phase_deg=float(np.percentile(abs(vals),95)) if len(vals) else np.nan))
    savecsv(out/'shared_pair_summary.csv',shared)
    bg=[s for s in sessions if '/kong/' in s];bgcal=[];within=[];drift=[];null=[];changes=[]
    bgcov={s:np.nanmean([r['cov'] for r in all_data[s]],axis=0) for s in bg}
    for held in bg:
        off,edges=error_summary([r for s in bg if s!=held for r in all_data[s]],all_data[held])
        for e in edges:bgcal.append(dict(heldout=held.split('/')[-1],**e))
        off,edges=error_summary(all_data[held][:1],all_data[held][1:])
        vals=[e['mean_abs_residual_deg'] for e in edges if e['state'].startswith('G') and e['n_subcarriers']]
        within.append(dict(session=held.split('/')[-1],finite_offsets=int(np.isfinite(off).sum()),testable_G_pair_tones=sum(e['n_subcarriers'] for e in edges if e['state'].startswith('G')),mean_abs_phase_deg=float(np.mean(vals)) if vals else np.nan))
        template=np.nanmean([bgcov[s] for s in bg if s!=held],axis=0)
        for si,s in enumerate(STATES):null.append(dict(session=held.split('/')[-1],state=s,covariance_distance=distance(bgcov[held][si],template[si])))
    for a,b in itertools.combinations(bg,2):
        za,na=aggregate(all_data[a]);zb,nb=aggregate(all_data[b]);pa,ra=phase_r(za,na);pb,rb=phase_r(zb,nb)
        for e,(s,x,y) in enumerate(EDGES):
            mask=(na[e]>=30)&(nb[e]>=30)&(ra[e]>=.5)&(rb[e]>=.5)
            delta=wrap(pb[e]-pa[e]);dd=delta[mask]
            rotation=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(dd)))))) if len(dd) else np.nan
            residual=wrap(dd-rotation)
            slope=np.nan
            if mask.sum()>=10:
                slope=float(np.polyfit(np.flatnonzero(mask),np.degrees(np.unwrap(np.radians(dd))),1)[0])
            drift.append(dict(session_a=a.split('/')[-1],session_b=b.split('/')[-1],state=s,pair=f'{x}-{y}',testable_tones=int(mask.sum()),mean_abs_raw_phase_deg=meanabs(dd),fitted_rotation_deg=rotation,mean_abs_after_rotation_deg=meanabs(residual),slope_deg_per_tone_index=slope))
    fulltemplate=np.nanmean(list(bgcov.values()),axis=0)
    for sess in sessions:
        if sess in bg:continue
        # A-D labels are retained only for stratified replication, never geometric fitting.
        for si,s in enumerate(STATES):
            m=np.nanmean([r['cov'][si] for r in all_data[sess]],axis=0)
            dist=distance(m,fulltemplate[si]);threshold=max(x['covariance_distance'] for x in null if x['state']==s)
            diff=m-fulltemplate[si]
            ev=np.linalg.eigvalsh(np.nan_to_num(diff))
            rank1=float(np.sum(np.maximum(ev[:,-1],0)**2)/max(np.sum(ev**2),1e-30))
            changes.append(dict(point=sess.split('/')[1],session=sess.split('/')[-1],state=s,covariance_distance=dist,background_LOSO_max=threshold,exceeds_background_max=dist>threshold,positive_rank1_energy_fraction=rank1,
                                interpretation='scene_change_only_not_identified_human_path'))
    savecsv(out/'background_holdout_edges_full.csv',bgcal);savecsv(out/'within_background_full.csv',within);savecsv(out/'background_pairwise_drift.csv',drift);savecsv(out/'background_covariance_null.csv',null);savecsv(out/'human_background_covariance.csv',changes)
    zsum,nsum=aggregate([r for s in bg for r in all_data[s]]);p,r=phase_r(zsum,nsum);off,_=solve(p,r,nsum)
    np.savez_compressed(out/'run_statistics.npz',phase_sums=np.array([r['z'] for s in sessions for r in all_data[s]]),phase_counts=np.array([r['n'] for s in sessions for r in all_data[s]]),normalized_covariance=np.array([r['cov'] for s in sessions for r in all_data[s]]),session=np.repeat([s.split('/')[-1] for s in sessions],3),point=np.repeat([s.split('/')[1] for s in sessions],3),run=np.tile(['run1','run2','run3'],17))
    # Geometry quoted from supplied README; not independently verified metadata.
    gt={'A':(.8,.6),'B':(2.8,1.2),'C':(.8,-1.2),'D':(2.4,-1.6)}
    estimates=[]
    for label,(x,y) in gt.items():
        for method in ['G1','G2','G3','G4','Fusion']:
            estimates.append(dict(point=label,method=method,ground_truth_x_m=x,ground_truth_y_m=y,ground_truth_angle_deg=float(np.degrees(np.arctan2(y,x))),estimated_angle_deg='',estimated_x_m='',estimated_y_m='',angle_error_deg='',position_error_m='',status='NOT_IDENTIFIABLE',reason='cross_batch_reference_and_person_path_not_validated; element_geometry_unverified'))
    savecsv(out/'aoa_fusion_status.csv',estimates)
    valid=sum(e['n_subcarriers'] for e in bgcal if e['state'].startswith('G'))
    summary=dict(sessions=len(sessions),runs=len(qc),raw_zip_sha256=sha(args.zip.read_bytes()),manifest_entries=len(manifest),manifest_failures=sum(not x['size_ok'] or not x['sha256_ok'] for x in manifest),raw_csi_files=len(duplicate),duplicate_csi_files=sum(bool(x['duplicate_of']) for x in duplicate),full_decoded_frames=sum(q['total_frames'] for q in qc),invalid_permutations=sum(q['invalid_permutation'] for q in qc),finite_background_offsets=int(np.isfinite(off).sum()),background_G_holdout_testable=valid,background_G_holdout_possible=1800,within_background=within,shared_pair_run_mean_abs_median_deg=float(np.nanmedian([s['mean_abs_phase_deg'] for s in shared])),shared_pair_run_p95_median_deg=float(np.nanmedian([s['p95_abs_phase_deg'] for s in shared])),timing_std_ms_range=[min(q['host_event_residual_std_ms'] for q in qc),max(q['host_event_residual_std_ms'] for q in qc)],unique_boot_ids=len(set(q['boot_id'] for q in qc)),covariance_scene_change_flags=sum(x['exceeds_background_max'] for x in changes),covariance_scene_change_tests=len(changes),AoA_Fusion_validated=False,ground_truth_used_for_fit=False,scalar_decoder_self_checks_passed=True,limitations=['Fixed-layout decoder checks structural metadata; not cross-checked against official PicoScenes executable parser.','Host-device offset has unknown serial latency; sensitivity is not independent synchronization validation.','Covariance removes common per-packet phase and total tone power but retains chain-dependent hardware drift.','Five background sessions yield a descriptive empirical null, not a calibrated significance test.','Array element coordinates, orientation, and hardware phase-only reference not established by acquisition metadata.','Ground truth coordinates inherited from supplied README and not independently confirmed.'])
    savejson(out/'summary.json',summary)
    rows=[]
    for sess in sessions:rows.append([np.mean([q[s+'_valid_packets'] for q in qc if q['session']==sess.split('/')[-1]]) for s in STATES])
    labels=[s.split('/')[1]+' '+s.split('/')[-1].split('_')[1] for s in sessions]
    svg_heatmap(out/'01_packet_coverage.svg','Full valid packets per run (mean over 3 runs)',rows,labels,STATES,vmax=max(map(max,rows)),footer='Packet counts; no downsampling. Color scale spans observed counts.')
    rows=[[np.nanmedian([q['mean_abs_raw_phase_deg'] for q in drift if q['session_a']==a.split('/')[-1] and q['session_b']==b.split('/')[-1] and q['state']==s]) for s in STATES] for a,b in itertools.combinations(bg,2)]
    svg_heatmap(out/'02_background_drift.svg','Background batch differences: median pair phase magnitude (deg)',rows,[f'K{bg.index(a)+1} vs K{bg.index(b)+1}' for a,b in itertools.combinations(bg,2)],STATES)
    rows=[[np.nanmedian([q['mean_abs_phase_deg'] for q in shared if q['session']==sess.split('/')[-1]])] for sess in sessions]
    svg_heatmap(out/'03_shared_pair.svg','Adjacent C1 / C2 shared pair (6,8): median run mean |phase difference|',rows,labels,['deg'])
    rows=[[100*x['covariance_distance']/x['background_LOSO_max'] for x in changes if x['session']==sess.split('/')[-1]] for sess in sessions if sess not in bg]
    svg_heatmap(out/'04_scene_change.svg','Scene covariance distance / largest held-out background distance (%)',rows,[labels[sessions.index(sess)] for sess in sessions if sess not in bg],STATES,vmax=200,footer='100% = empirical background maximum; not a significance threshold or AoA error.')
    print(json.dumps(clean(summary),indent=2),flush=True)

if __name__=='__main__':main()
