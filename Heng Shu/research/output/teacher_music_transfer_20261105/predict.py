"""Fixed, truth-free transfer of background subtraction and spectral product to spatial MUSIC."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
sys.path.insert(0,str(OUT/"kong1_kong1a_20261105"))
import analyze_captures as base

LATEST=OUT/"paired_be_abcde_20261105"
INPUTS={"A":OUT/"new_a_paired_20261105","B":LATEST/"B","C":OUT/"real_cd_paired_20261105"/"C",
        "D":OUT/"real_cd_paired_20261105"/"D","E":LATEST/"E","B_old":OUT/"paired_bc_20261105"/"B",
        "C_old":OUT/"paired_bc_20261105"/"C"}
ANG=np.arange(-90.,91.)
STATES=base.STATES
BANDS=["full","low","high"]
STEER={s:np.exp(2j*np.pi/base.C*base.F[:,None,None]*base.Y[np.array(base.MAP[s])-1][None,:,None]*
                   np.sin(np.deg2rad(ANG))[None,None,:]) for s in STATES}


def save(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding="utf-8-sig")


def input_covariances(site,folder):
    cov={}; audit=[]
    for row in pd.read_csv(folder/"raw_runs.csv").itertuples():
        source=Path(row.source)
        assert base.sha(source)==row.sha256 and row.continuity_gate
        raw,host,nic=base.decoder.decode(source)
        addresses,_=base.frame_details(source)
        assert len(raw)==len(addresses)==row.frames and np.all(np.diff(host)>0) and np.all(np.diff(nic)>0)
        times=(host-row.start_ns)/1e9
        target=(addresses==base.APPARENT_TA)&(times>=0)&(times<10)
        assert target.sum()==row.target_frames and row.duration_s==10
        log=source.parent/"events.log"
        records=base.EVENT.findall(log.read_text(encoding="utf8"))
        assert all([int(n) for n in chain.split(",")]==base.MAP[s] and int(seq)==STATES.index(s)
                   for _,s,chain,_,seq in records)
        aligned,order,_=base.decoder.align(log,row.start_ns/1e9)
        serial=np.array([float(e[0])-row.start_ns/1e9 for e in records])
        for clock,starts in [("aligned",aligned),("serial",serial)]:
            idx=np.searchsorted(starts,times,side="right")-1
            valid_idx=np.maximum(idx,0)
            dt=times-starts[valid_idx]
            inside=target&(idx>=0)&(dt>=.003)&(dt<.021)
            total=0
            for j,state in enumerate(STATES):
                select=inside&(order[valid_idx]==j)
                x=raw[select].astype(np.complex128)
                t=times[select]
                assert len(x)>0 and np.isfinite(x).all()
                total+=len(x)
                masks={"full":np.ones(len(t),bool),"first5":t<5}
                if clock=="aligned":
                    masks.update({f"Q{i+1}":(t>=i*2.5)&(t<(i+1)*2.5) for i in range(4)})
                for block,mask in masks.items():
                    assert mask.any()
                    # unit packet mean-square power across 3 chains and 30 tones
                    cov[clock,row.run,block,state]=90*base.spatial(x[mask])
                audit.append(dict(point=site,folder=source.parents[2].name,run=row.run,sha256=row.sha256,phy=row.phy,
                                  ifindex=row.ifindex,raw_frames=row.frames,formal_target=row.target_frames,clock=clock,
                                  state=state,selected=len(x),valid_visits=np.unique(idx[select]).size))
            assert total==inside.sum()
    return cov,audit


def rotation(cfirst,state):
    q=base.relative(cfirst)
    los=base.los_response(state)
    return np.stack([np.ones(len(base.F),complex),
                     np.exp(1j*np.angle(q[0]*los[1].conj())),
                     np.exp(1j*np.angle(q[1]*los[2].conj()))]).T


def music(matrix,steer,difference):
    hermitian=(matrix+matrix.conj().transpose(0,2,1))/2
    values,vectors=np.linalg.eigh(hermitian)
    tol=1e-10*np.maximum(1.,abs(values).max(axis=1))
    positive=values[:,-1]>tol
    unique=(values[:,-1]-values[:,-2])>tol
    valid=positive&unique
    # Delta may be indefinite; take the leading positive contrast direction only.
    # This is a change-direction model, not a covariance of the person.
    norms=np.sum(abs(steer)**2,axis=1)
    projection=np.einsum("ki,kij->kj",vectors[:,:,-1].conj(),steer)
    noise=np.clip(1-abs(projection)**2/norms,0,1)
    curves=[]
    for band in [np.arange(30),np.arange(15),np.arange(15,30)]:
        tones=band[valid[band]]
        if len(tones)==0:
            curves.append(np.full(len(ANG),np.nan))
        else:
            p=1/np.maximum(noise[tones].mean(axis=0),1e-12)
            curves.append(p/p.max())
    return np.array(curves),dict(usable_tones=int(valid.sum()),negative_tones=int((values[:,0]<-tol).sum()),
                                 multiple_positive_tones=int((values[:,1]>tol).sum()),difference=difference)


def each_state(c1,ct,phase,state,model):
    a=STEER[state]
    d=ct-c1
    if model=="raw":
        matrix=ct
    elif model=="delta":
        matrix=d
    elif model=="phase_delta":
        matrix=d*phase.conj()[:,:,None]*phase[:,None,:]
    elif model=="white":
        loaded=.9*c1+.1*np.trace(c1,axis1=1,axis2=2).real[:,None,None]/3*np.eye(3)[None]
        values,u=np.linalg.eigh(loaded)
        w=(u/np.sqrt(np.maximum(values,1e-12))[:,None,:])@u.conj().transpose(0,2,1)
        matrix=w@ct@w.conj().transpose(0,2,1)
        a=w@a
    else:
        raise ValueError(model)
    return music(matrix,a,model in ("delta","phase_delta"))


def beam_contrast(d,state):
    a=STEER[state]
    projection=np.einsum("kid,kij,kjd->kd",a.conj(),d,a,optimize=True).real/3
    return np.maximum(projection,0).mean(axis=0)


def coarse_keep(gcurve):
    ids=np.arange(0,len(ANG),10)
    v=gcurve[ids]
    if not np.isfinite(v).all():
        return np.zeros(len(ANG),bool)
    local=np.flatnonzero(v==maximum_filter1d(v,3,mode="nearest"))
    chosen=[]
    for i in local[np.argsort(-v[local],kind="stable")]:
        pos=ANG[ids[i]]
        if all(abs(pos-j)>=20 for j in chosen):
            chosen.append(pos)
        if len(chosen)==2:break
    return np.any(np.array([abs(ANG-j)<=15 for j in chosen]),axis=0)


def fusions(profiles):
    groups=np.stack([profiles[s] for s in STATES])
    four=groups[:4]
    out={s:profiles[s] for s in STATES}
    for name,x in [("G4",four),("S6",groups)]:
        out[name+"_mean"]=x.mean(axis=0)
        out[name+"_product"]=np.exp(np.log(np.maximum(x,1e-12)).mean(axis=0))
    mask=np.stack([coarse_keep(v) for v in out["G4_mean"]])
    out["S6_product_coarse"]=np.where(mask,out["S6_product"],np.nan)
    return out,mask


def largest(curve):
    return ANG[np.nanargmax(curve)] if np.isfinite(curve).any() else np.nan


def check_synthetic():
    for s in [*STATES]:
        a=STEER[s][:,:,np.flatnonzero(ANG==-27)[0]]
        c1=np.tile(np.eye(3,dtype=complex)[None],(30,1,1))
        ct=c1+.35*a[:,:,None]*a.conj()[:,None,:]
        for model in ["raw","delta","white"]:
            curves,_=each_state(c1,ct,np.ones((30,3)),s,model)
            assert largest(curves[0])==-27
        z,_=each_state(c1,c1,np.ones((30,3)),s,"delta")
        assert np.isnan(z).all() and np.all(beam_contrast(c1-c1,s)==0)


def analyze():
    check_synthetic()
    runs=[]; peaks=[]; eigen=[]; powers=[]; drifts=[]; curves={}
    for site,folder in INPUTS.items():
        cov,audit=input_covariances(site,folder)
        runs.extend(audit)
        for clock in ["aligned","serial"]:
            rotations={s:rotation(cov[clock,1,"first5",s],s) for s in STATES}
            for s in STATES:
                pre=base.relative(cov[clock,1,"full",s]);post=base.relative(cov[clock,3,"full",s])
                keep=(abs(pre)>=.5)&(abs(post)>=.5)
                z=np.abs(np.rad2deg(np.angle(post*pre.conj())))
                drifts.append(dict(point=site,clock=clock,state=s,supported=keep.mean(),
                                   full_empty_pair_drift_p95=np.quantile(z[keep],.95) if keep.any() else np.nan))
            # Swap reference once, as a sensitivity only; the primary R3 is untouched.
            for ref,r in [(1,2),(1,3),(3,2)]:
                blocks=["full"]
                if clock=="aligned" and ref==1:
                    blocks+=["Q1","Q2","Q3","Q4"]
                for block in blocks:
                    c0={s:cov[clock,ref,"full",s] for s in STATES}
                    test={s:cov[clock,r,block,s] for s in STATES}
                    beams={s:beam_contrast(test[s]-c0[s],s) for s in STATES}
                    if block=="full":
                        for sn,b in [("G3",beams["G3"]),("S6",np.mean(list(beams.values()),axis=0))]:
                            powers.append(dict(point=site,clock=clock,reference=ref,test=r,scope=sn,
                                               contrast_peak_power=b.max(),contrast_peak_deg=largest(b)))
                    models=["delta"] if block!="full" or ref==3 else ["raw","delta","white","phase_delta"]
                    for model in models:
                        profiles={}
                        for s in STATES:
                            profiles[s],record=each_state(c0[s],test[s],rotations[s],s,model)
                            if block=="full":eigen.append(dict(point=site,clock=clock,reference=ref,test=r,model=model,state=s,**record))
                        fs,mask=fusions(profiles)
                        for label,x in fs.items():
                            if block!="full" and label not in ("S6_product","S6_product_coarse","G3"):continue
                            for i,band in enumerate(BANDS):
                                peaks.append(dict(point=site,clock=clock,reference=ref,test=r,block=block,model=model,
                                     group=label,band=band,angle=largest(x[i]),domain_grid_count=mask[i].sum() if label=="S6_product_coarse" else len(ANG)))
                            if clock=="aligned" and block=="full" and ref==1 and label in ("G3","G4_mean","S6_product","S6_product_coarse"):
                                curves[f"{site}_{model}_{label}_R{r}"]=x[0]
        print(site, "done",flush=True)
    save("packet_audit.csv",runs)
    save("blind_peaks.csv",peaks)
    save("eigen_diagnostics.csv",eigen)
    save("contrast_strength.csv",powers)
    save("phase_drift.csv",drifts)
    np.savez_compressed(ROOT/"spectra.npz",angles=ANG,**curves)
    files=["protocol.json","predict.py","packet_audit.csv","blind_peaks.csv","eigen_diagnostics.csv",
           "contrast_strength.csv","phase_drift.csv","spectra.npz"]
    (ROOT/"blind_registry.json").write_text(json.dumps({n:base.sha(ROOT/n) for n in files},indent=2),encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(limits=1):
        analyze()

