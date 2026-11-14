"""Blind shortlist, local branch votes and empty controls. No survey coordinates."""
from hashlib import sha256
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"music_subarray_gain_20261105"
sys.path.insert(0,str(PRIOR))
import predict_subarrays as core
ANG=core.ANG
BANDS=core.BANDS
POINTS=("A","B","C","D","E")
ROLES=("P","S","L1","L2","H1","H2")
ARMS={"G2_splits":("G2",)*6,"G_only_12":("G2","G3","G2","G2","G4","G1"),
      "short_long_12":("G2","G3","C1","C2","G4","G1")}
MODELS=("background_null","positive_q1")
DB_FLOOR=1e-12


def save(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding="utf-8-sig")


def packet_moments(site,packets):
    quota=12*(min(len(x) for parts in packets.values() for x in parts)//12)
    assert quota>=12
    moments={};ids={};full_g2={};budget_audit=[]
    required={(state,j) for states in ARMS.values() for j,state in enumerate(states)}
    for r in (1,2,3):
        for state,j in sorted(required):
            cov=[];ident=[]
            for q,x in enumerate(packets[r,state]):
                all_ids=np.floor((np.arange(quota)+.5)*len(x)/quota).astype(int)
                ix=all_ids[j::6]; assert len(np.unique(ix))==quota//6 and len(np.unique(all_ids))==quota
                ident.extend((state,q,int(i)) for i in ix)
                cov.append(90*core.base.spatial(x[ix]))
            ids[r,state,j]=ident
            moments[r,state,j]={"full":np.mean(cov,axis=0),**{f"Q{k+1}":v for k,v in enumerate(cov)}}
        full_g2[r]={key:np.mean([moments[r,"G2",j][key] for j in range(6)],axis=0) for key in ("full","Q1","Q2","Q3","Q4")}
        for arm,states in ARMS.items():
            used=set()
            for j,state in enumerate(states):
                part=ids[r,state,j]
                assert not used.intersection(part)
                used.update(part)
                budget_audit.append(dict(point=site,arm=arm,run=r,role=ROLES[j],state=state,packets=len(part),
                    selected_ordinals_sha256=sha256(json.dumps(part,separators=(",",":")).encode()).hexdigest()))
            assert len(used)==4*quota
    return moments,full_g2,budget_audit


class Spectrum:
    def __init__(self,profile):
        self.cost=np.asarray(profile)
        a=self.cost
        finite=np.isfinite(a)
        ids=[]
        if finite.any():
            for i in find_peaks(np.where(finite,-a,-np.inf))[0]:
                if i>0 and i<len(a)-1 and finite[i-1:i+2].all():
                    ids.append(i)
            if finite[:2].all() and a[0]<a[1]:ids.append(0)
            if finite[-2:].all() and a[-1]<a[-2]:ids.append(len(a)-1)
        self.peaks=[]
        if ids:
            z=max(float(a[finite].min()),DB_FLOOR)
            for i in sorted(ids,key=lambda j:(a[j],j)):
                loss=10*np.log10(max(a[i],DB_FLOOR)/z)
                if loss<=6+1e-10:
                    self.peaks.append((float(ANG[i]),float(loss)))

    def local(self,direction):
        return any(abs(p-direction)<=9 for p,_ in self.peaks)

    def neighbourhood(self,direction):
        v=self.cost[(abs(ANG-direction)<=9)&np.isfinite(self.cost)]
        return float(max(np.min(v),DB_FLOOR)) if len(v) else np.nan


def shortlist(spec):
    out=[]
    for angle,_ in spec.peaks:
        if all(abs(angle-v)>=20 for v in out):
            out.append(angle)
    return out[:2],len(out)


def preference(spec,a,b):
    x=spec.neighbourhood(a);y=spec.neighbourhood(b)
    return float(10*np.log10(y/x)) if np.isfinite(x) and np.isfinite(y) else np.nan


def supports(dataset,angle):
    views=[dataset["full",b] for b in BANDS]
    quarters=[dataset[f"Q{i}","full"] for i in range(1,5)]
    local=[v.local(angle) for v in views]
    quarters_yes=sum(v.local(angle) for v in quarters)
    checkable=all(np.isfinite(v.neighbourhood(angle)) for v in views) and sum(np.isfinite(v.neighbourhood(angle)) for v in quarters)>=3
    return bool(all(local) and quarters_yes>=3),bool(checkable)


def vote(dataset,seeds,i,tol,full_only=False):
    if len(seeds)==1:
        return dataset["full","full"].local(seeds[0]) if full_only else supports(dataset,seeds[0])[0]
    def margin(block,band):
        return preference(dataset[block,band],seeds[i],seeds[1-i])
    m=margin("full","full")
    full=np.isfinite(m) and m>1e-9 and m+1e-10>=tol
    if full_only:return bool(full)
    return bool(full and all(margin("full",b)>1e-9 for b in ("low","high"))
                and sum(margin(f"Q{q}","full")>1e-9 for q in range(1,5))>=3)


def full_pref(dataset,seeds):
    if len(seeds)!=2:return np.nan
    return preference(dataset["full","full"],*seeds)


def assess(seeds,trials,ratio,tol):
    foreground,empty,reverse=trials
    margins=[full_pref(foreground[s],seeds) for s in ("L1","L2")]
    mu=float(np.mean(margins)) if len(seeds)==2 and np.isfinite(margins).all() else np.nan
    preferred=seeds[0] if len(seeds)==1 else (seeds[int(mu<0)] if np.isfinite(mu) and abs(mu)>1e-9 else np.nan)
    details=[]
    for i,angle in enumerate(seeds):
        short,short_ok=supports(foreground["P"],angle)
        l1=vote(foreground["L1"],seeds,i,tol);l2=vote(foreground["L2"],seeds,i,tol)
        corroboration=[]
        neg_triples=[];complete_triples=[]
        for long,held in (("L1","H1"),("L2","H2")):
            corroboration.append(supports(foreground[long],angle)[0] and supports(foreground[held],angle)[0])
            negs=[supports(empty[s],angle) for s in ("P",long,held)]
            neg_triples.append(all(u[0] for u in negs))
            complete_triples.append(all(u[1] for u in negs))
        s_ok=np.isfinite(foreground["S"]["full","full"].neighbourhood(angle))
        if len(seeds)==2:
            s_ok=s_ok and np.isfinite(full_pref(foreground["S"],seeds)) and not vote(foreground["S"],seeds,1-i,tol)
        rev_ok=reverse["P"]["full","full"].local(angle) and all(vote(reverse[s],seeds,i,tol,True) for s in ("L1","L2"))
        reasons=[]
        if abs(angle)>=87:reasons.append("边界")
        if not short:reasons.append("短基线跨频段/分块不稳")
        if not(l1 and l2):reasons.append("两判别状态未同时支持")
        if not any(corroboration):reasons.append("留出复核未通过")
        if not s_ok:reasons.append("辅助短基线反对或不可评")
        if not ratio>1:reasons.append("强度未超过空场")
        if any(neg_triples):reasons.append("空场也支持邻近峰")
        if not any(complete_triples):reasons.append("空场对照不完整")
        if not rev_ok:reasons.append("换参考未复现")
        details.append(dict(seed=angle,short_supported=short,short_evaluable=short_ok,L1_vote=l1,L2_vote=l2,
            L1_H1_corroborate=corroboration[0],L2_H2_corroborate=corroboration[1],aux_short_ok=s_ok,
            reverse_ok=rev_ok,empty_stable_near=any(neg_triples),empty_checkable=any(complete_triples),
            passes=not reasons,reason="、".join(reasons)))
    passed=[r["seed"] for r in details if r["passes"]]
    assert len(passed)<=1
    if passed:status="单解"
    elif len(seeds)==2:status="双候选未决"
    elif len(seeds)==1:status="单候选未确认"
    else:status="无候选"
    row=dict(count=len(seeds),seed1=seeds[0] if seeds else np.nan,seed2=seeds[1] if len(seeds)>1 else np.nan,
             L1_margin_to_first_db=margins[0],L2_margin_to_first_db=margins[1],combined_margin_db=mu,
             provisional=preferred,selected=passed[0] if passed else np.nan,decision=status,
             strength_ratio=ratio)
    return row,details


def profiles(data,state,model,outers,ref,test,blocks):
    prior=data[ref]["full"]
    training=core.fit(prior,np.stack([data[ref][f"Q{i}"] for i in range(1,5)]))
    views={}
    for block in blocks:
        N,B,use,_=core.projectors(model,prior,data[test][block],training)
        cs=core.state_cost(outers[state],N,B,use,False)
        for i,b in enumerate(BANDS):
            cost=core.joint({state:cs},[state],i)[2]
            views[block,b]=Spectrum(cost)
    return views


def synthetic():
    def pattern(peak):
        v=.10+.85*(np.cos(np.radians((ANG-peak)*3))-1)**2
        return Spectrum(v)
    def paired(a,b,second):
        v=np.minimum(.12+.002*(ANG-a)**2,second+.002*(ANG-b)**2)
        return Spectrum(v)
    def views(spec,partial=False):
        return {(block,band):spec for block in (("full",) if partial else ("full","Q1","Q2","Q3","Q4")) for band in BANDS}
    a,b=-26.,34.
    short=paired(a,b,.17)
    assert shortlist(short)[0]==[a,b]
    pro_b=paired(b,a,.58)
    pro_a=paired(a,b,.58)
    neutral=paired(a,b,.12)
    null=views(pattern(63))
    empty={s:null for s in ROLES}
    fg={s:views(pro_b) for s in ROLES};fg["P"]=views(short)
    rev=fg
    row,_=assess([a,b],(fg,empty,rev),2.,1.)
    assert row["decision"]=="单解" and row["selected"]==b
    split=dict(fg,L2=views(pro_a));row,_=assess([a,b],(split,empty,rev),2.,1.)
    assert row["decision"]=="双候选未决"
    tie=dict(fg,L1=views(neutral));row,_=assess([a,b],(tie,empty,rev),2.,1.)
    assert row["decision"]=="双候选未决"
    row,_=assess([a,b],(fg,fg,rev),2.,1.)
    assert row["decision"]=="双候选未决"
    unavailable=Spectrum(np.full(len(ANG),np.inf))
    assert shortlist(unavailable)==([],0) and not np.isfinite(preference(unavailable,a,b))
    g=core.base.MAP
    assert set(g["G4"]).isdisjoint(g["G2"]+g["C1"])
    assert set(g["G1"]).isdisjoint(g["G2"]+g["C2"])
    assert set(g["C1"])&set(g["C2"])=={6,8}


def run():
    registry=json.loads((PRIOR/"blind_registry.json").read_text(encoding="utf8"))
    assert all(core.base.sha(PRIOR/k)==v for k,v in registry.items())
    synthetic()
    outers=core.steering_outers()
    inputs=[];alloc=[];selections=[];gates=[];diagnostics=[];baseline=[];saved={}
    for site in POINTS:
        packets,records=core.read_packets(site,core.previous.INPUTS[site])
        inputs.extend(records)
        m,full_g2,audit=packet_moments(site,packets);alloc.extend(audit)
        for model in MODELS:
            cache={};powers={}
            sets=[(1,2),(1,3),(3,2)]
            for state,j in sorted({(a,j) for states in ARMS.values() for j,a in enumerate(states)}):
                data={r:m[r,state,j] for r in (1,2,3)}
                for ref,test in sets:
                    blocks=("full","Q1","Q2","Q3","Q4") if ref==1 else ("full",)
                    cache[state,j,ref,test]=profiles(data,state,model,outers,ref,test,blocks)
                    powers[state,j,ref,test]=core.previous.beam_contrast(data[test]["full"]-data[ref]["full"],state)
            for ref,test in sets:
                g2=profiles(full_g2,"G2",model,outers,ref,test,("full","Q1","Q2","Q3","Q4") if ref==1 else ("full",))
                v=g2["full","full"]; idx=np.nanargmin(v.cost) if np.isfinite(v.cost).any() else None
                cs,n=shortlist(v)
                baseline.append(dict(point=site,model=model,ref=ref,test=test,global_peak=ANG[idx] if idx is not None else np.nan,
                    seed1=cs[0] if cs else np.nan,seed2=cs[1] if len(cs)>1 else np.nan,total_separated=n))
                saved[f"{site}_{model}_G2_all_ref{ref}_r{test}"]=v.cost
            for arm,states in ARMS.items():
                trials=[{role:cache[s,j,ref,test] for j,(s,role) in enumerate(zip(states,ROLES))} for ref,test in sets]
                fg,empty,reverse=trials
                proposal=fg["P"]["full","full"];seeds,n=shortlist(proposal)
                ratio_vals=[np.mean([powers[s,j,1,r] for j,s in enumerate(states)],axis=0).max() for r in (2,3)]
                ratio=ratio_vals[0]/ratio_vals[1] if ratio_vals[1]>0 else np.nan
                for margin in (0.,1.,3.):
                    row,details=assess(seeds,trials,ratio,margin)
                    key=dict(point=site,model=model,arm=arm,threshold_db=margin)
                    selections.append(dict(**key,total_separated=n,**row))
                    gates.extend(dict(**key,**rec) for rec in details)
                for role,state in zip(ROLES,states):
                    for dataset,(ref,test) in zip(trials,sets):
                        for (block,band),spec in dataset[role].items():
                            m1=full_pref({("full","full"):spec},seeds)
                            info={f"near_seed{i+1}":spec.local(a) for i,a in enumerate(seeds)}
                            diagnostics.append(dict(point=site,model=model,arm=arm,role=role,state=state,
                                ref=ref,test=test,block=block,band=band,margin_to_first_db=m1,
                                local_peaks="/".join(f"{a:g}" for a,_ in spec.peaks),**info))
                        saved[f"{site}_{model}_{arm}_{role}_ref{ref}_r{test}"]=dataset[role]["full","full"].cost
        print(site,"complete",flush=True)
    save("input_files.csv",inputs);save("packet_allocations.csv",alloc);save("blind_decisions.csv",selections)
    save("branch_checks.csv",gates);save("role_spectra_checks.csv",diagnostics);save("G2_all_baseline.csv",baseline)
    np.savez_compressed(ROOT/"full_profiles.npz",angle=ANG,**saved)
    names=["protocol.json","predict_branches.py","input_files.csv","packet_allocations.csv","blind_decisions.csv",
           "branch_checks.csv","role_spectra_checks.csv","G2_all_baseline.csv","full_profiles.npz"]
    (ROOT/"blind_registry.json").write_text(json.dumps({n:core.base.sha(ROOT/n) for n in names},indent=2),encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(limits=1):run()

