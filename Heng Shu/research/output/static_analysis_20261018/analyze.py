import os
from pathlib import Path
import json, struct, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
DESK=Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))
STATES=['G1','G2','G3','G4','C1','C2']
COORD={'old+1':(1.64,1),'old-1':(1.64,-1),'old+2':(1.64,2),'old-2':(1.64,-2),'A':(.8,.6),'B':(2.8,1.2),'C':(.8,-1.2),'D':(2.4,-1.6),'kong':None}
SOURCES=[(k,DESK/'1m _(1)'/v) for k,v in [('old+1','1m _'),('old-1','1m-2'),('old+2','45 good'),('old-2','45 bad')]]+[(k,DESK/'buchong shuju'/k) for k in ['A','B','C','D','kong']]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def decode(p):
    # Narrow, fail-closed parser for observed frame v1 / RxBasic v4 / CSI v3.
    # Format reference: https://github.com/wifisensing/RXS-Parsing-Core
    raw=p.read_bytes(); pos=0; payload=[]; ant=[]; host=[]; nic=[]
    while pos<len(raw):
        size,magic,ver,ns=struct.unpack_from('<IIHB',raw,pos)
        assert magic==0x20150315 and ver==1 and pos+size+4<=len(raw)
        end=pos+size+4; s=pos+11; seen=set()
        for _ in range(ns):
            length,n=struct.unpack_from('<IB',raw,s); name=raw[s+5:s+5+n].rstrip(b'\0').decode(); version=struct.unpack_from('<H',raw,s+5+n)[0]; b=s+7+n
            assert s+4+length<=end
            if name=='RxSBasic':
                assert version==4 and struct.unpack_from('<H',raw,b)[0]==0x5300
                nic.append(struct.unpack_from('<Q',raw,b+2)[0]);host.append(struct.unpack_from('<Q',raw,b+10)[0])
                assert struct.unpack_from('<hhh',raw,b+18)==(5230,5240,40)
                assert raw[b+24]==1 and struct.unpack_from('<H',raw,b+25)[0]==40
                assert tuple(raw[b+30:b+33])==(1,0,3)
                seen.add(name)
            if name=='CSI':
                assert version==3
                assert struct.unpack_from('<H',raw,b)[0]==0x5300 and raw[b+2]==1
                assert struct.unpack_from('<H',raw,b+3)[0]==40
                assert struct.unpack_from('<H',raw,b+25)[0]==30
                assert tuple(raw[b+27:b+30])==(1,3,0)
                assert struct.unpack_from('<H',raw,b+30)[0]==1
                assert struct.unpack_from('<I',raw,b+35)[0]==192
                ant.append(raw[b+32]);payload.append(raw[b+39:b+231]);seen.add(name)
            s+=length+4
        assert seen=={'CSI','RxSBasic'}
        pos=end
    assert pos==len(raw)
    a=np.frombuffer(b''.join(payload),np.uint8).reshape(-1,192)
    c=np.empty((len(a),3,30),np.complex64)
    for k in range(30):
        for j in range(3):
            bit=51*k+3+16*j; byte,shift=divmod(bit,8)
            u=a[:,byte].astype(np.uint16)|(a[:,byte+1].astype(np.uint16)<<8)
            v=a[:,byte+1].astype(np.uint16)|(a[:,byte+2].astype(np.uint16)<<8)
            c[:,j,k]=((u>>shift)&255).astype(np.uint8).view(np.int8).astype(float)+1j*((v>>shift)&255).astype(np.uint8).view(np.int8)
    for i in [0,len(a)//2,len(a)-1]:
        bits=int.from_bytes(a[i].tobytes(),'little')
        for k in range(30):
            for j in range(3):
                bit=51*k+3+16*j;x=(bits>>bit)&255;y=(bits>>(bit+8))&255
                assert c[i,j,k]==complex(x-256 if x>127 else x,y-256 if y>127 else y)
    ant=np.asarray(ant);perm=np.stack([(ant>>(2*j))&3 for j in range(3)],1)
    assert np.all(np.sort(perm,axis=1)==[0,1,2])
    for order in np.unique(perm,axis=0):
        m=np.all(perm==order,axis=1);c[m]=c[m][:,np.argsort(order),:]
    return c,np.array(host,dtype=np.int64),np.array(nic,dtype=np.int64)

def align(p,start):
    ev=re.findall(r'([\d.]+)\s+READY (G[1-4]|C[12]) chains=[\d,]+ t=(\d+) seq_idx=(\d)/6',p.read_text())
    h=np.array([float(x[0]) for x in ev])-start;d=np.array([int(x[2]) for x in ev],dtype=np.int64)
    state=np.array([STATES.index(x[1]) for x in ev]);assert np.all(state==np.arange(len(ev))%6)
    delta=(d-d[0])/1e6
    # Host serial arrival is an upper bound, RUN_ISSUED + settling a lower bound.
    # Allow 100 ppm oscillator drift; use guarded interiors, not host READY times.
    lo=.0001;hi=float(np.min(h-delta+delta*100e-6))
    if hi<lo:return None
    anchor=(lo+hi)/2;guard=(hi-lo)/2+delta[-1]*100e-6+.0001
    return anchor+delta,state,guard

def main():
    runs=[];bursts=[];integrity=[];features=[];featureids=[];batchrows=[]
    for label,root in SOURCES:
        for manifest in sorted(root.rglob('manifest.json')):
            batch=manifest.parent
            for e in read(manifest)['files']:
                p=batch/e['path'];ok=p.is_file() and p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
                integrity.append(dict(label=label,batch=batch.name,file=str(p),ok=ok))
            status=read(batch/'acquisition_summary.json')['status']
            batchrows.append(dict(label=label,batch=batch.name,status=status,path=str(batch)))
        for p in sorted(root.rglob('*.csi')):
            r=read(p.parent/'run.json');q=read(p.parent/'metadata_qc.json');batch=p.parent.parent.parent
            c,host,nic=decode(p);assert len(c)==q['complete_frames']==r['frames']
            t=(host-r['start_ns'])/1e9;mask=(t>=0)&(host<r['end_ns']);duration=r['requested_seconds']
            assert int(mask.sum())==r['window']['records']
            row=dict(label=label,batch=batch.name,run=p.parent.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frames=len(c),window_frames=int(mask.sum()),duration_s=duration,window_pps=float(mask.sum()/duration),leading_s=float(t[mask][0]) if mask.any() else duration,trailing_s=float(duration-t[mask][-1]) if mask.any() else duration,max_gap_ms=float(np.diff(host).max()/1e6),failure=r.get('failure'),identity_unchanged=r.get('identity_unchanged'),start_ns=r['start_ns'],phase_sd_deg=np.nan,shape_cv=np.nan,guard_ms=np.nan,bursts=0)
            alignment=align(p.parent/'events.log',r['start_ns']/1e9)
            profiles={s:[] for s in STATES};bs=[]
            if alignment:
                starts,states,guard=alignment;row['guard_ms']=guard*1000
                for ei in range(len(starts)-1):
                    s=STATES[states[ei]];start=starts[ei]
                    if start+.025>duration or .025-2*guard<.005:continue
                    i,j=np.searchsorted(t,[start+guard,start+.025-guard])
                    if j-i<5:continue
                    x=c[i:j];power=np.abs(x)**2;shape=power/(power.mean(axis=(1,2),keepdims=True)+1e-12)
                    prod=np.stack([x[:,0]*x[:,1].conj(),x[:,1]*x[:,2].conj(),x[:,2]*x[:,0].conj()],1)
                    good=np.all(np.abs(prod)>0,axis=0);u=prod/(np.abs(prod)+1e-12)
                    phase=float(np.median(np.sqrt(-2*np.log(np.clip(np.abs(u.mean(0))[good],1e-9,1)))*180/np.pi)) if good.any() else np.nan
                    shape_cv=float(np.median(shape.std(0)/(shape.mean(0)+1e-12)))
                    z=dict(label=label,batch=batch.name,run=p.parent.name,state=s,time_s=start,packets=j-i,phase_sd_deg=phase,shape_cv=shape_cv)
                    bursts.append(z);bs.append(z);profiles[s].append(shape.mean(0))
                if bs:
                    row.update(phase_sd_deg=float(np.nanmedian([b['phase_sd_deg'] for b in bs])),shape_cv=float(np.nanmedian([b['shape_cv'] for b in bs])),bursts=len(bs))
                if all(profiles[s] for s in STATES):
                    features.append(np.stack([np.mean(profiles[s],axis=0) for s in STATES]));featureids.append(len(runs))
            runs.append(row);print(label,batch.name,p.parent.name,len(c),round(row['window_pps'],1),row['bursts'],flush=True)
    csv('runs.csv',runs);csv('bursts.csv',bursts);csv('integrity.csv',integrity);csv('batches.csv',batchrows)
    np.savez_compressed(OUT/'power_profiles.npz',profiles=np.array(features),run_indices=np.array(featureids))
    df=pd.DataFrame(runs);bf=pd.DataFrame(batchrows)
    group=df.groupby('label',sort=False).agg(runs=('run','size'),frames=('frames','sum'),formal_frames=('window_frames','sum'),duration=('duration_s','median'),pps=('window_pps','median'),phase_sd=('phase_sd_deg','median'),shape_cv=('shape_cv','median'),bad_trailing=('trailing_s',lambda s:int((s>.1).sum())))
    group['batches']=bf.groupby('label').size();group['failed_batches']=bf.groupby('label')['status'].apply(lambda s:int((s!='PASS').sum()));group.to_csv(OUT/'groups.csv',encoding='utf-8-sig')
    geometry=[]
    for k,xy in COORD.items():
        if xy:
            x,y=xy;length=np.hypot(x,y)+np.hypot(x-3.65,y)
            geometry.append(dict(label=k,x=x,y=y,bistatic_m=length,excess_m=length-3.65,excess_ns=(length-3.65)/299792458*1e9,bearing_deg=np.degrees(np.arctan2(y,x))))
    csv('geometry_theory_only.csv',geometry)
    (OUT/'metadata.json').write_text(json.dumps(dict(coordinates=COORD,geometry_source='User-confirmed uploaded diagram; theoretical horizontal point-scatterer approximation',method='Full raw PicoScenes frame validation and Intel5300 signed8 decode; standard 2-bit antenna permutation; per-packet normalized power and intra-burst inter-chain circular phase dispersion; NO calibrated AoA/ToF or position estimate',alignment='RUN_ISSUED/READY causal bounds with assumed 100ppm oscillator allowance; guarded interiors >=5ms, >=5 packets. Not external clock calibration.',format_reference='https://github.com/wifisensing/RXS-Parsing-Core',integrity_checked=len(integrity),integrity_pass=sum(x['ok'] for x in integrity)),ensure_ascii=False,indent=2),encoding='utf8')
    plt.rcParams.update({'font.size':10})
    labels=list(group.index);fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    for ax,col,title in zip(axes,['window_pps','phase_sd_deg','shape_cv'],['Received packets / scheduled second','Within-burst phase SD (degrees)','Within-burst normalized power CV']):
        ax.boxplot([df[df.label==s][col].dropna() for s in labels],tick_labels=labels,showfliers=True)
        ax.set_title(title);ax.tick_params(axis='x',rotation=50);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Run-level diagnostics: old points vs new points and empty room (kong)');fig.tight_layout();fig.savefig(OUT/'quality.png',dpi=180);plt.close(fig)
    fs=np.array(features);fdf=df.iloc[featureids].reset_index(drop=True)
    # RMS difference across 6 states x 3 chains x 30 tones; dimensionless, not metres.
    flat=fs.reshape(len(fs),-1);distance=np.sqrt(np.mean((flat[:,None]-flat[None,:])**2,axis=2))
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(distance,cmap='viridis',vmin=0);centers=[];pos=0
    for label in labels:
        count=int((fdf.label==label).sum());centers.append(pos+(count-1)/2);pos+=count;ax.axhline(pos-.5,color='white',lw=.6);ax.axvline(pos-.5,color='white',lw=.6)
    ax.set_xticks(centers,labels,rotation=45);ax.set_yticks(centers,labels);ax.set_title('Normalized CSI power-profile RMS distance\nAll usable runs; blocks may include batch / day effects');fig.colorbar(im,ax=ax,label='Dimensionless RMS distance');fig.tight_layout();fig.savefig(OUT/'profile_distance.png',dpi=180);plt.close(fig)
    pairs=[]
    for i in range(len(fs)):
        for j in range(i):
            a,b=fdf.iloc[i],fdf.iloc[j]
            typ='same_point_same_batch' if a.label==b.label and a.batch==b.batch else 'same_point_other_batch' if a.label==b.label else 'different_point'
            pairs.append(dict(label1=a.label,label2=b.label,batch1=a.batch,batch2=b.batch,type=typ,rms=float(distance[i,j])))
    csv('profile_pair_distances.csv',pairs)
    print(group.to_string());print('integrity',len(integrity),sum(x['ok'] for x in integrity),flush=True)

if __name__=='__main__':main()
