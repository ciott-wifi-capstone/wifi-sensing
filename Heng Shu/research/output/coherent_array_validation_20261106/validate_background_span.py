"""Offline paired-empty subspace test. Truth-free, all cached complex covariances."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
import validate_coherent as c
ROOT=Path(__file__).resolve().parent

def projector(r1,r3,kind):
    m=r1.shape[-1];eye=np.eye(m)
    if kind=='pooled_rank1':
        _,v=np.linalg.eigh(c.herm((r1+r3)/2));h=v[:,:,-1:]
        return eye-h@h.conj().swapaxes(-1,-2),np.ones(30,int)
    _,u=np.linalg.eigh(c.herm(r1));_,v=np.linalg.eigh(c.herm(r3))
    h=np.stack([u[:,:,-1],v[:,:,-1]],axis=-1)
    w,s,_=np.linalg.svd(h,full_matrices=False)
    second=s[:,1]/np.maximum(s[:,0],1e-15)>.05
    take=np.stack([np.ones(30,bool),second],axis=1)
    return eye-(w*take[:,None,:])@w.conj().swapaxes(-1,-2),1+second.astype(int)

def profile(B,nb,test,train,y):
    m=len(y);e,u=np.linalg.eigh(c.herm(B@test@B))
    floor=np.max(np.linalg.eigvalsh(c.herm(B[None]@train@B[None]))[...,-1],axis=0)
    use=(m-nb>=2)&(e[:,-1]>floor+1e-10)
    h=u[:,:,-1:];N=B-h@h.conj().swapaxes(-1,-2)
    a=np.exp(2j*np.pi/c.base.C*c.F[:,None,None]*y[None,:,None]*np.sin(np.radians(c.ANG))[None,None,:])
    num=np.maximum(np.einsum('kma,kmn,kna->ka',a.conj(),N,a).real,0)
    den=np.einsum('kma,kmn,kna->ka',a.conj(),B,a).real
    costs=[]
    for ids in (np.arange(30),np.arange(15),np.arange(15,30)):
        good=ids[use[ids]];v=np.full(len(c.ANG),np.nan)
        if len(good)*2>=len(ids):
            ns=num[good].sum(0);ds=den[good].sum(0)
            np.divide(ns,ds,out=v,where=ds>=.10*m*len(good))
        costs.append(v)
    power=float(np.mean(np.trace(B@test@B,axis1=1,axis2=2).real/np.maximum(m-nb,1)))
    return np.array(costs),power,int(use.sum())

def main():
    registry=json.loads((ROOT/'prediction_freeze.json').read_text())
    assert all(c.base.sha(ROOT/k)==v for k,v in registry.items())
    rows=[];saved={}
    for site in 'ABCDE':
        z=np.load(ROOT/f'{site}_complex_covariances.npz')
        for arm in ('G2','tree6','tree8'):
            raw=z[arm];y=c.base.Y[3:6] if arm=='G2' else c.base.Y
            cal=c.calibration(raw[0,:8].mean(0),y)
            for calname in ('raw_phase','R1_empirical'):
                d=raw if calname=='raw_phase' else np.array([c.transform(a,cal) for a in raw])
                r1=d[0,:8].mean(0);r3=d[2,:12].mean(0)
                train=np.array([d[0,:4].mean(0),d[0,4:8].mean(0),d[2,:6].mean(0),d[2,6:12].mean(0)])
                for model in ('pooled_rank1','two_empty_span'):
                    B,nb=projector(r1,r3,model)
                    views={};powers={};nt={}
                    blocks={'predict':d[1,:12].mean(0),'held':d[1,12:].mean(0),
                        'all_R2':d[1].mean(0),'negative_R1':d[0,8:].mean(0),'negative_R3':d[2,12:].mean(0)}
                    for block,data in blocks.items():
                        v,p,n=profile(B,nb,data,train,y)
                        views[block]=v;powers[block]=p;nt[block]=n
                        saved[f'{site}_{arm}_{calname}_{model}_{block}']=v
                    a=c.peak(views['predict'][0]);h=[c.peak(v) for v in views['held']]
                    neg=[c.peak(views[b][0]) for b in ('negative_R1','negative_R3')]
                    consistent=np.isfinite(h+[a]).all() and max(abs(np.array(h)-a))<=9
                    empty_same=any(np.isfinite(x) and abs(x-a)<=9 for x in neg)
                    ratio=powers['predict']/max(powers['negative_R1'],powers['negative_R3'],1e-15)
                    gate=bool(np.isfinite(a) and abs(a)<87 and consistent and not empty_same and ratio>1)
                    rows.append(dict(point=site,arm=arm,calibration=calname,background=model,prediction=a,
                        all_R2_peak=c.peak(views['all_R2'][0]),held_R2=h[0],held_low=h[1],held_high=h[2],
                        background_two_rank_tones=int((nb==2).sum()),valid_noise_dimension_tones=int((len(y)-nb>=2).sum()),
                        usable_prediction_tones=nt['predict'],negative_R1_angle=neg[0],negative_R3_angle=neg[1],
                        foreground_negative_ratio=ratio,spectral_gate=gate,physical_calibration_verified=False,physical_release=False))
        print(site,'paired-empty span diagnostic complete',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'blind_background_span.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(ROOT/'background_span_profiles.npz',angle=c.ANG,**saved)
    names=['additional_protocol.json','validate_background_span.py','blind_background_span.csv','background_span_profiles.npz']
    (ROOT/'background_span_freeze.json').write_text(json.dumps({n:c.base.sha(ROOT/n) for n in names},indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
