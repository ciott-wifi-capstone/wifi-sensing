"""Bounded, truth-free MUSIC and matched-packet array comparisons."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
TRANSFER=ROOT.parent/"teacher_music_transfer_20261105"
sys.path.insert(0,str(TRANSFER))
import predict as previous
base=previous.base
STATES=base.STATES
GROUPS=STATES[:4]
ANG=previous.ANG
BANDS=("full","low","high")
RADII=np.array([np.inf,.75,1.5,3.,6.])
I=np.eye(3,dtype=complex)
SETS=[(s,(s,)) for s in STATES]
for size in (2,3,4):
    SETS.extend(("+".join(g),g) for g in combinations(GROUPS,size))
SETS.append(("six_states",tuple(STATES)))
METHODS=("positive_q1","background_null","signed_rank")


def save(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding="utf-8-sig")


def read_packets(site,folder):
    packets={}; audit=[]
    for row in pd.read_csv(folder/"raw_runs.csv").itertuples():
        source=Path(row.source)
        assert base.sha(source)==row.sha256 and row.continuity_gate
        raw,host,nic=base.decoder.decode(source)
        ta,_=base.frame_details(source)
        assert len(raw)==len(ta)==row.frames and np.all(np.diff(host)>0) and np.all(np.diff(nic)>0)
        log=source.parent/"events.log"
        starts,order,_=base.decoder.align(log,row.start_ns/1e9)
        t=(host-row.start_ns)/1e9
        idx=np.searchsorted(starts,t,side="right")-1
        valid=np.maximum(idx,0)
        dt=t-starts[valid]
        target=(ta==base.APPARENT_TA)&(t>=0)&(t<10)
        assert target.sum()==row.target_frames
        keep=target&(idx>=0)&(dt>=.003)&(dt<.021)
        for j,s in enumerate(STATES):
            selected=keep&(order[valid]==j)
            quarters=[]
            for q in range(4):
                x=raw[selected&(t>=2.5*q)&(t<2.5*(q+1))].astype(complex)
                assert len(x)>0
                quarters.append(x)
            packets[row.run,s]=quarters
            audit.append(dict(point=site,run=row.run,state=s,sha256=row.sha256,
                              selected=sum(len(x) for x in quarters),quarter_counts="/".join(str(len(x)) for x in quarters)))
    return packets,audit


def moments(packets,n,quota,policy):
    result={}
    for key,quarters in packets.items():
        blocks=[];sizes=[]
        for x in quarters:
            if policy=="equal":
                count=quota//n
                # Half-bin positions spread the fixed budget through the recorded quarter.
                ids=np.floor((np.arange(count)+.5)*len(x)/count).astype(int)
                assert len(np.unique(ids))==count and ids[-1]<len(x)
                x=x[ids]
            blocks.append(90*base.spatial(x))
            sizes.append(len(x))
        for q,c in enumerate(blocks,1):
            result[*key,f"Q{q}"]=c
        result[*key,"full"]=np.average(blocks,axis=0,weights=sizes)
    return result


def steering_outers():
    out={}
    for s in STATES:
        y=base.Y[np.array(base.MAP[s])-1]
        sin=np.sin(np.radians(ANG))
        paths=[-y[:,None]*sin]
        for r in RADII[1:]:
            paths.append(np.sqrt(r*r+y[:,None]**2-2*r*y[:,None]*sin)-r)
        phase=-2j*np.pi/base.C*base.F[:,None,None,None]*np.stack(paths)[None]
        a=np.exp(phase).transpose(0,1,3,2)
        assert np.allclose(a[:,0].transpose(0,2,1),previous.STEER[s])
        out[s]=a[:,:,:,:,None]*a[:,:,:,None,:].conj()
    return out


def outer(vec):
    return vec[:,:,None]*vec[:,None,:].conj()


def hermitian(a):
    return (a+a.conj().swapaxes(-1,-2))/2


def fit(ref,quarters):
    eig,v=np.linalg.eigh(hermitian(ref))
    B=I-outer(v[:,:,-1])
    ref_dom=eig[:,-1]/np.maximum(eig.sum(1),1e-12)
    spread=np.max(np.abs(np.linalg.eigvalsh(hermitian(quarters-ref))),axis=(0,2))
    residual_floor=np.max(np.linalg.eigvalsh(hermitian(B@quarters@B))[...,-1],axis=0)
    return B,ref_dom,spread,residual_floor


def projectors(method,ref,test,background):
    B,dominance,spread,residual_floor=background
    if method=="background_null":
        val,vec=np.linalg.eigh(hermitian(B@test@B))
        u=vec[:,:,-1]
        use=(dominance>=.70)&(val[:,-1]>residual_floor+1e-10)&(val[:,-1]-val[:,-2]>1e-10)
        return B-outer(u),B,use,np.ones(30,int)
    val,vec=np.linalg.eigh(hermitian(test-ref))
    eye=np.broadcast_to(I,ref.shape)
    if method=="positive_q1":
        tol=1e-10*np.maximum(1,abs(val).max(axis=1))
        use=(val[:,-1]>tol)&(val[:,-1]-val[:,-2]>tol)
        return eye-outer(vec[:,:,-1]),eye,use,np.ones(30,int)
    selected=abs(val)>np.maximum(spread[:,None],1e-10)
    q=selected.sum(1)
    signal=(vec*selected[:,None,:])@vec.conj().transpose(0,2,1)
    use=(q>=1)&(q<=2)
    loss=(eye-signal)*(3/np.maximum(3-q,1))[:,None,None]
    return loss,eye,use,q


def state_cost(outers,N,B,use,near):
    # frequency x range x angle; tr(M a a^H) is the MUSIC projection energy.
    steering=outers if near else outers[:,:1]
    def response(m):
        return np.einsum("kij,krtji->krt",m,steering,optimize=True).real
    num=response(N)
    den=response(B)
    assert np.min(num[use])>=-1e-6 if use.any() else True
    arrays=[]
    for indices in (np.arange(30),np.arange(15),np.arange(15,30)):
        valid=indices[use[indices]]
        k=len(valid)
        if 2*k<len(indices):
            arrays.append(None)
        else:
            arrays.append((np.maximum(num[valid],0).sum(0),den[valid].sum(0),k))
    return arrays


def joint(costs,states,band,spherical=False):
    selected=[costs[s][band] for s in states]
    if any(z is None for z in selected):
        return np.nan,np.nan,np.full(len(ANG),np.nan)
    num=sum(z[0] for z in selected);den=sum(z[1] for z in selected)
    count=sum(z[2] for z in selected)
    cost=np.divide(num,den,out=np.full_like(num,np.inf),where=(den>=.10*3*count))
    view=cost if spherical else cost[:1]
    if not np.isfinite(view).any():
        return np.nan,np.nan,np.full(len(ANG),np.nan)
    i,j=np.unravel_index(np.argmin(view),view.shape)
    return ANG[j],RADII[i],np.min(view,axis=0)


def synthetic(outers):
    # A coherent static mixture: the reflection shares an RF source with the background.
    for s in STATES:
        a0=previous.STEER[s][:,:,np.where(ANG==4)[0][0]]
        a1=previous.STEER[s][:,:,np.where(ANG==-28)[0][0]]
        ref=outer(a0)+.02*I
        target=outer(a0+(.68+.45j)*a1)+.02*I
        fitdata=fit(ref,np.repeat(ref[None],4,axis=0))
        for method in ("background_null","signed_rank"):
            N,B,use,_=projectors(method,ref,target,fitdata)
            assert use.all()
            cost=state_cost(outers[s],N,B,use,False)
            # Some isolated triplets admit an alternate exact solution. Verify the planted
            # direction is a noise-subspace minimum, not uniqueness of the solution.
            v=joint({s:cost},[s],0)[2]
            assert v[np.where(ANG==-28)[0][0]]<1e-8
            same=projectors(method,ref,ref,fitdata)
            assert not same[2].any()
    longamb=previous.STEER["C2"]
    i,j=[np.where(ANG==a)[0][0] for a in (48,-18)]
    rho=np.abs(np.sum(longamb[:,:,i].conj()*longamb[:,:,j],axis=1)/3)
    assert np.median(rho)>.98


def run():
    outers=steering_outers()
    synthetic(outers)
    results=[];sources=[];budgets=[];support=[];strength=[];stored={}
    prior_audit=pd.read_csv(TRANSFER/"packet_audit.csv").query('clock=="aligned"')
    previous_peaks=pd.read_csv(TRANSFER/"blind_peaks.csv")
    ref_registry=json.loads((TRANSFER/"blind_registry.json").read_text(encoding="utf8"))
    assert all(base.sha(TRANSFER/name)==value for name,value in ref_registry.items())
    for site,folder in previous.INPUTS.items():
        packets,audit=read_packets(site,folder)
        sources.extend(audit)
        first=pd.DataFrame(audit).sort_values(["run","state"])
        earlier=prior_audit.query('point==@site').sort_values(["run","state"])
        assert np.array_equal(first.selected.to_numpy(),earlier.selected.to_numpy())
        quota=12*(min(len(x) for qq in packets.values() for x in qq)//12)
        assert quota>0
        for policy,n in [("all",0),*[("equal",n) for n in (1,2,3,4,6)]]:
            config=[(label,states) for label,states in SETS if policy=="all" or len(states)==n]
            if not config:continue
            mom=moments(packets,n or 1,quota,policy)
            for label,states in config:
                for r in (1,2,3):
                    num=sum(sum(len(x) for x in packets[r,s]) for s in states) if policy=="all" else 4*quota
                    budgets.append(dict(point=site,budget=policy,config=label,states="/".join(states),run=r,total_packets=num,
                                        unique_antennas=len(set(a for s in states for a in base.MAP[s]))))
            for ref,r in ((1,2),(1,3),(3,2)):
                fitdata={}
                for s in STATES:
                    parts=np.stack([mom[ref,s,f"Q{q}"] for q in range(1,5)])
                    fitdata[s]=fit(mom[ref,s,"full"],parts)
                blocks=("full","Q1","Q2","Q3","Q4") if ref==1 else ("full",)
                for block in blocks:
                    near=(block=="full" and policy=="all" and ref==1)
                    for mode in METHODS:
                        cs={}
                        for s in STATES:
                            N,B,use,q=projectors(mode,mom[ref,s,"full"],mom[r,s,block],fitdata[s])
                            cs[s]=state_cost(outers[s],N,B,use,near and mode=="background_null")
                            if block=="full":
                                support.append(dict(point=site,budget=policy,states_sharing_budget=n,ref=ref,test=r,method=mode,
                                                    state=s,used_tones=use.sum(),q0=(q==0).sum(),q1=(q==1).sum(),q2=(q==2).sum(),q3=(q==3).sum()))
                        for label,states in config:
                            for band in range(3):
                                angle,radius,profile=joint(cs,states,band)
                                results.append(dict(point=site,budget=policy,config=label,method=mode,ref=ref,test=r,block=block,
                                                    band=BANDS[band],angle=angle))
                                if mode=="positive_q1" and policy=="all" and len(states)==1 and block=="full" and ref==1:
                                    prev=previous_peaks[(previous_peaks.point==site)&(previous_peaks.clock=="aligned")&
                                      (previous_peaks.reference==1)&(previous_peaks.test==r)&(previous_peaks.block=="full")&
                                      (previous_peaks.model=="delta")&(previous_peaks.group==label)&(previous_peaks.band==BANDS[band])].angle.item()
                                    assert np.isclose(angle,prev,equal_nan=True)
                                if near and mode=="background_null" and label in ("G1+G2+G3+G4","six_states") and band==0:
                                    phi,rad,cost=joint(cs,states,band,spherical=True)
                                    results.append(dict(point=site,budget=policy,config=label,method="profile_radius",ref=ref,test=r,
                                                        block=block,band="full",angle=phi,nuisance_radius_m=rad))
                                if ref==1 and block=="full" and band==0 and label in ("G2","G3","C1","G1+G2+G3+G4","six_states"):
                                    stored[f"{site}_{policy}_{label}_{mode}_R{r}"]=profile
                    if block=="full":
                        power={s:previous.beam_contrast(mom[r,s,block]-mom[ref,s,"full"],s) for s in STATES}
                        for label,states in config:
                            intensity=np.stack([power[s] for s in states]).mean(axis=0).max()
                            strength.append(dict(point=site,budget=policy,config=label,ref=ref,test=r,intensity=intensity))
        print(site,"done",flush=True)
    save("blind_angles.csv",results)
    save("packet_sources.csv",sources);save("packet_budgets.csv",budgets)
    save("tone_support.csv",support);save("contrast_power.csv",strength)
    np.savez_compressed(ROOT/"pooled_residuals.npz",angles=ANG,**stored)
    files=["protocol.json","predict_subarrays.py","blind_angles.csv","packet_sources.csv","packet_budgets.csv",
           "tone_support.csv","contrast_power.csv","pooled_residuals.npz"]
    (ROOT/"blind_registry.json").write_text(json.dumps({name:base.sha(ROOT/name) for name in files},indent=2),encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(limits=1):run()

