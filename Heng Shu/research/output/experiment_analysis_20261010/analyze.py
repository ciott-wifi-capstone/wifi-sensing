from pathlib import Path
import json, re, hashlib, csv
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STATES=['G1','G2','G3','G4','C1','C2']

def write_csv(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def decode(path):
    raw=np.fromfile(path,dtype=np.uint8)
    assert len(raw)%215==0, 'Unexpected framing; do not silently continue'
    a=raw.reshape(-1,215);h=a[:,3:23];p=a[:,23:]
    assert np.all(a[:,0]==0) and np.all(a[:,1]==213) and np.all(a[:,2]==187)
    assert np.all(h[:,8]==3) and np.all(h[:,9]==1)
    assert np.all(h[:,16]==192) and np.all(h[:,17]==0)
    n=len(a);c=np.empty((n,3,30),dtype=np.complex64)
    for k in range(30):
        for j in range(3):
            bit=k*51+3+j*16;byte=bit//8;shift=bit%8
            u=p[:,byte].astype(np.uint16)|(p[:,byte+1].astype(np.uint16)<<8)
            v=p[:,byte+1].astype(np.uint16)|(p[:,byte+2].astype(np.uint16)<<8)
            c[:,j,k]=((u>>shift)&255).astype(np.uint8).view(np.int8).astype(float)+1j*((v>>shift)&255).astype(np.uint8).view(np.int8)
    # Independent scalar bit extraction check on first/middle/last records.
    for i in [0,n//2,n-1]:
        bits=int.from_bytes(p[i].tobytes(),'little')
        for k in range(30):
            for j in range(3):
                b=k*51+3+j*16;x=(bits>>b)&255;y=(bits>>(b+8))&255
                assert c[i,j,k]==complex(x-256 if x>127 else x,y-256 if y>127 else y)
    perm=np.stack([(h[:,15]>>(2*j))&3 for j in range(3)],axis=1)
    assert np.all(np.sort(perm,axis=1)==[0,1,2])
    for order in np.unique(perm,axis=0):
        mask=np.all(perm==order,axis=1);c[mask]=c[mask][:,np.argsort(order),:]
    nic=sum(h[:,j].astype(np.int64)<<(8*j) for j in range(4))
    dn=(np.diff(nic)+2**32)%2**32
    count=h[:,4].astype(np.int64)+(h[:,5].astype(np.int64)<<8)
    dc=np.diff(count)%65536
    return c,h,np.r_[0,np.cumsum(dn)]/1e6,dn,dc,perm

def host_times(folder,date):
    # Files contain Sydney wall-clock time, in September UTC+10 (no DST).
    lines=(folder/'session_time.txt').read_text().splitlines()
    tod=np.array([int(s[:2])*3600+int(s[3:5])*60+float(s[6:]) for s in lines])
    epoch=datetime.fromisoformat(date).replace(tzinfo=timezone(timedelta(hours=10))).timestamp()
    return epoch+tod+np.r_[0,np.cumsum(np.diff(tod)<-43200)]*86400

def events(path):
    txt=path.read_text();issued=float(re.search(r'([\d.]+)\s+RUN_ISSUED',txt)[1])
    ev=re.findall(r'([\d.]+)\s+READY (G[1-4]|C[12]) chains=([\d,]+) t=(\d+) seq_idx=(\d)/6',txt)
    host=np.array([float(x[0]) for x in ev]);dev=np.array([int(x[3]) for x in ev],dtype=np.int64)
    state=np.array([STATES.index(x[1]) for x in ev]);assert np.all(state==np.arange(len(ev))%6)
    delta=(dev-dev[0])/1e6
    lo=issued+.0001;hi=np.min(host-delta+delta*100e-6)
    assert hi>=lo
    anchor=(hi+lo)/2
    guard=(hi-lo)/2+delta[-1]*100e-6+.0001
    return issued,anchor+delta,state,guard,{'anchor':float(anchor),'anchor_width_ms':float((hi-lo)*1000),'guard_ms':float(guard*1000),'valid_ms':float((.025-2*guard)*1000),'state_period_ms':float(np.median(np.diff(dev))/1000),'cycle_period_ms':float(np.median(dev[6:]-dev[:-6])/1000),'usb_latency_p95_ms':float(np.percentile(host-anchor-delta,95)*1000),'events':len(ev)}

def verify_files():
    rows=[]
    for folder,manifest,key,pathkey in [('D03_D04_raw_data_20260917','CONTENTS.json','source_files','archive_path'),('office_broadside_raw6state_20260907','delivery_manifest.json','files','path')]:
        d=json.loads((ROOT/folder/manifest).read_text())
        for e in d[key]:
            p=ROOT/e[pathkey] if pathkey=='archive_path' else ROOT/folder/e[pathkey]
            ok=p.exists() and p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
            rows.append({'file':str(p.relative_to(ROOT)),'pass':bool(ok)})
    write_csv('file_integrity.csv',rows)
    return {'checked':len(rows),'passed':sum(r['pass'] for r in rows),'failed':[r['file'] for r in rows if not r['pass']]}

def main():
    integrity=verify_files();print('integrity',integrity,flush=True)
    folders=[('Office','empty',ROOT/'office_broadside_raw6state_20260907','2026-09-07')]
    for dataset in ['D03','D04']:
        for folder in sorted((ROOT/'D03_D04_raw_data_20260917'/dataset/'native').iterdir()):
            scene='walking' if 'walking_r34' in folder.name else 'empty'
            folders.append((dataset,scene,folder,'2026-09-13'))
    files=[];runs=[];bursts=[];profile_means={};sensitivity=[]
    for dataset,scene,folder,date in folders:
        print('decode',dataset,scene,flush=True)
        c,h,nic,dn,dc,perm=decode(folder/'session.dat');host=host_times(folder,date)
        assert len(host)==len(c)
        summary=json.loads((folder/'summary.json').read_text())
        assert len(c)==summary['logger_packets']
        info={'dataset':dataset,'scene':scene,'records':len(c),'span_s':float(nic[-1]),'pps':float((len(c)-1)/nic[-1]),'max_nic_gap_ms':float(dn.max()/1000),'max_host_gap_ms':float(np.diff(host).max()*1000),'bfee_nonunit_steps':int(np.sum(dc!=1)),'nic_nonpositive_steps':int(np.sum(dn==0)),'rssi_zero_values':int(np.sum(h[:,10:13]==0)),'agc_values':np.unique(h[:,14]).tolist(),'permutations':np.unique(perm,axis=0).tolist(),'rate_flags':np.unique(h[:,18].astype(int)+256*h[:,19]).tolist()}
        files.append(info)
        for epath in sorted(folder.glob('run*_events.log')):
            native=int(re.search(r'run(\d+)',epath.name)[1]);logical=native+(2 if scene=='walking' else 0)
            issued,starts,states,guard,align=events(epath)
            sel=(host>=issued)&(host<issued+30)
            run={'dataset':dataset,'scene':scene,'run':logical,'raw_30s_packets':int(sel.sum()),**align}
            features=[]; nvalid=[]; vectors={s:[] for s in STATES};central_features=[]
            # Exclude the last event: STOP truncates its nominal 25-ms window.
            for ei in range(len(starts)-1):
                start=starts[ei];s=STATES[states[ei]]
                if start+.025>issued+30:continue
                i,j=np.searchsorted(host,[start+guard,start+.025-guard])
                if j-i<10:continue
                x=c[i:j];power=np.abs(x)**2
                # Per-packet unit vector removes common amplitude and phase via products.
                products=np.stack([x[:,0]*x[:,1].conj(),x[:,1]*x[:,2].conj(),x[:,2]*x[:,0].conj()],axis=1)
                u=products/(np.abs(products)+1e-12)
                good=np.all(np.abs(products)>0,axis=0)
                mean_u=u.mean(axis=0);rho=np.abs(mean_u)
                phase_sd=np.sqrt(-2*np.log(np.clip(rho,1e-9,1)))*180/np.pi
                phase_stat=float(np.median(phase_sd[good]))
                cv=float(np.median(power.std(axis=0)/(power.mean(axis=0)+1e-12)))
                shape=power/(np.mean(power,axis=(1,2),keepdims=True)+1e-12)
                shape_cv=float(np.median(shape.std(axis=0)/(shape.mean(axis=0)+1e-12)))
                row={'dataset':dataset,'scene':scene,'run':logical,'state':s,'event':ei,'time_s':float(start-issued),'packets':j-i,'duration_ms':float((nic[j-1]-nic[i])*1000),'phase_sd_deg':phase_stat,'power_cv':cv,'normalized_power_cv':shape_cv,'agc_changes':int(np.sum(np.diff(h[i:j,14])!=0))}
                bursts.append(row);features.append(row);nvalid.append(j-i);vectors[s].append(mean_u)
                # Narrower central 10 ms is an alignment sensitivity check.
                ci,cj=np.searchsorted(host,[start+.0075,start+.0175]);cu=u[max(ci-i,0):min(cj-i,len(u))]
                if len(cu)>8:
                    cs=np.sqrt(-2*np.log(np.clip(np.abs(cu.mean(axis=0)),1e-9,1)))*180/np.pi
                    central_features.append(float(np.median(cs[good])))
            run.update({'complete_used_bursts':len(features),'packets_per_burst_min':min(nvalid),'packets_per_burst_median':float(np.median(nvalid)),'packets_per_burst_max':max(nvalid),'used_packets':int(sum(nvalid)), 'phase_sd_deg':float(np.median([r['phase_sd_deg'] for r in features])),'normalized_power_cv':float(np.median([r['normalized_power_cv'] for r in features])),'central10ms_phase_sd_deg':float(np.median(central_features))})
            for s in STATES:
                a=np.array(vectors[s]);diff=np.angle(a[1:]*a[:-1].conj())*180/np.pi
                run[s+'_successive_burst_phase_change_deg']=float(np.median(np.abs(diff)))
                profile_means[(dataset,scene,logical,s)]=a.mean(axis=0)
            runs.append(run);print(run,flush=True)
        del c
    write_csv('bursts.csv',bursts);write_csv('runs.csv',runs)
    out={'integrity':integrity,'files':files,'runs':runs,'method':'Independent Intel5300 decode; permutation corrected; no absolute power scaling/calibration/CPA/AoA/ToF/DFS. Guarded per-burst common-phase-invariant diagnostics. Temporal samples are correlated; descriptive, not classification accuracy.'}
    (OUT/'results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item()),encoding='utf-8')
    # Comparison plots: all complete guarded bursts, no concatenated raw time series.
    labels=['Office empty','D03 empty','D03 walking','D04 empty','D04 walking']
    keys=[('Office','empty'),('D03','empty'),('D03','walking'),('D04','empty'),('D04','walking')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,col,title in zip(axes,['phase_sd_deg','normalized_power_cv'],['Within-burst relative-phase variation','Within-burst normalized power variation']):
        values=[[r[col] for r in bursts if (r['dataset'],r['scene'])==key] for key in keys]
        ax.boxplot(values,labels=labels,showfliers=False,whis=(10,90));ax.set_title(title);ax.tick_params(axis='x',rotation=23);ax.grid(axis='y',alpha=.25)
        ax.set_ylabel('Median circular SD over pairs/tones (degrees)' if col=='phase_sd_deg' else 'Median coefficient of variation')
    fig.suptitle('Raw CSI diagnostics | guarded state bursts | boxes: 25-75%, whiskers: 10-90%')
    fig.savefig(OUT/'comparison.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharey=True)
    for di,dataset in enumerate(['D03','D04']):
        for si,scene in enumerate(['empty','walking']):
            ax=axes[di,si]
            for runnum in ([1,2] if scene=='empty' else [3,4]):
                vals=[r for r in bursts if r['dataset']==dataset and r['scene']==scene and r['run']==runnum]
                bins=np.arange(30);med=[np.median([r['phase_sd_deg'] for r in vals if int(r['time_s'])==b]) for b in bins]
                ax.plot(bins+.5,med,label=f'R{runnum}',lw=1.5)
            ax.set_title(f'{dataset} {scene}');ax.set_xlabel('Time since run start (s)');ax.set_ylabel('Relative-phase SD (degrees)');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('One-second medians of within-burst diagnostics (not a Doppler spectrogram)')
    fig.savefig(OUT/'time_comparison.png',dpi=190);plt.close(fig)
    print(json.dumps({'files':files,'integrity':integrity},indent=2),flush=True)

if __name__=='__main__':main()
