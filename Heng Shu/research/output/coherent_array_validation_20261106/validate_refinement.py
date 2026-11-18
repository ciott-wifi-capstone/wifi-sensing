"""Frozen temporal split, local noise-residual refinement; no survey truth."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'short_long_branch_20261106'))
import predict_branches as old
ANG=old.ANG;ROLES=old.ROLES

def fuse(profiles,weights):
    ix=np.where(np.asarray(weights)>0)[0]
    if not len(ix):return np.full(len(ANG),np.nan)
    a=np.asarray(profiles)[ix];w=np.asarray(weights)[ix]
    good=np.isfinite(a).all(0)
    out=np.full(len(ANG),np.nan)
    out[good]=np.average(np.log(np.maximum(a[:,good],1e-12)),axis=0,weights=w)
    return out

def argmin(cost,center=None,radius=None):
    good=np.isfinite(cost)
    if center is not None:good=good&(abs(ANG-center)<=radius)
    if not good.any():return np.nan
    ids=np.where(good)[0];return float(ANG[ids[np.argmin(cost[ids])]])

def local_weights(datasets,center,radius):
    weights=[];rows=[]
    for role in ROLES:
        ds=datasets[role]
        full=fuse([ds['Q1','full'].cost,ds['Q2','full'].cost],[1,1])
        a=argmin(full,center,radius)
        angles=[argmin(ds[q,b].cost,center,radius) for q,b in
                [('Q1','full'),('Q2','full'),('Q1','low'),('Q1','high'),('Q2','low'),('Q2','high')]]
        spread=max(angles)-min(angles) if np.isfinite(angles).all() else np.inf
        interior=np.isfinite(a) and abs(a-center)<radius and abs(a)<87
        eligible=bool(interior and spread<=6)
        w=1/(1+(spread/3)**2) if eligible else 0.
        weights.append(w);rows.append(dict(role=role,local_peak=a,spread_deg=spread,eligible=eligible,weight=w))
    # The common short proposal remains available even if additional roles are rejected.
    # It is not automatically released: independent frequency/time gates still apply.
    weights[0]=max(weights[0],1.)
    return np.array(weights),rows

def main():
    registry=json.loads((old.ROOT/'blind_registry.json').read_text(encoding='utf8'))
    assert all(old.core.base.sha(old.ROOT/k)==v for k,v in registry.items())
    outs=[];weightrows=[];profiles={};outers=old.core.steering_outers()
    for site in 'ABCDE':
        packets,_=old.core.read_packets(site,old.core.previous.INPUTS[site])
        m,_,_=old.packet_moments(site,packets)
        cache={};power={}
        for s,j in sorted({(s,j) for states in old.ARMS.values() for j,s in enumerate(states)}):
            data={r:m[r,s,j] for r in (1,2,3)}
            for ref,test in [(1,2),(1,3),(3,2)]:
                cache[s,j,ref,test]=old.profiles(data,s,'background_null',outers,ref,test,('full','Q1','Q2','Q3','Q4'))
                power[s,j,ref,test]=old.core.previous.beam_contrast(data[test]['full']-data[ref]['full'],s)
        proposal=cache['G2',0,1,2]
        seed=argmin(fuse([proposal['Q1','full'].cost,proposal['Q2','full'].cost],[1,1]))
        for arm,states in old.ARMS.items():
            trials=[{role:cache[s,j,ref,test] for j,(role,s) in enumerate(zip(ROLES,states))}
                    for ref,test in [(1,2),(1,3),(3,2)]]
            fg,empty,reverse=trials
            for radius in (9,18):
                w,audit=local_weights(fg,seed,radius)
                weightrows.extend(dict(point=site,arm=arm,radius=radius,**r) for r in audit)
                for mode,ww in [('equal',np.ones(6)),('reliability',w)]:
                    def spectrum(ds,blocks,band):
                        vectors=[]
                        for role in ROLES:
                            vectors.append(fuse([ds[role][b,band].cost for b in blocks],np.ones(len(blocks))))
                        # Inputs above are log costs. Average without a second logarithm.
                        ix=np.where(ww>0)[0];v=np.array(vectors)[ix];valid=np.isfinite(v).all(0)
                        out=np.full(len(ANG),np.nan)
                        out[valid]=np.average(v[:,valid],axis=0,weights=ww[ix]);return out
                    pred=spectrum(fg,['Q1','Q2'],'full');angle=argmin(pred,seed,radius)
                    held=spectrum(fg,['Q3','Q4'],'full')
                    hp=argmin(held,seed,radius)
                    hp_global=argmin(held)
                    halves=[argmin(spectrum(fg,['Q3','Q4'],band)) for band in ('low','high')]
                    qs=[argmin(spectrum(fg,[q],'full')) for q in ('Q3','Q4')]
                    neg=[argmin(spectrum(empty,['full'],b)) for b in old.BANDS]
                    rev=argmin(spectrum(reverse,['Q3','Q4'],'full'),seed,radius)
                    checkable=np.isfinite(neg).all()
                    empty_near=checkable and max(abs(np.array(neg)-angle))<=9
                    onedge=not np.isfinite(angle) or abs(angle-seed)>=radius or abs(angle)>=87
                    stable=np.isfinite(halves+qs+[rev]).all() and max(abs(np.array(halves+qs+[rev])-angle))<=9
                    beams=[np.average(np.array([power[s,j,1,r] for j,s in enumerate(states)]),axis=0,weights=ww).max() for r in (2,3)]
                    ratio=beams[0]/max(beams[1],1e-15)
                    statistical=not onedge and stable and checkable and not empty_near and ratio>1
                    outs.append(dict(point=site,arm=arm,radius=radius,weighting=mode,seed=seed,prediction=angle,
                        held_local=hp,held_global=hp_global,held_low=halves[0],held_high=halves[1],held_Q3=qs[0],held_Q4=qs[1],
                        reverse_held_local=rev,empty_peak=neg[0],empty_evaluable=checkable,empty_same_direction=empty_near,
                        search_boundary=onedge,stable=stable,strength_ratio=ratio,spectral_gate=bool(statistical),
                        active_roles=int((ww>0).sum()),long_roles_weight=float(ww[2]+ww[3]),weights='/'.join(f'{x:.6f}' for x in ww),
                        human_path_validated=False,physical_release=False))
                    profiles[f'{site}_{arm}_{radius}_{mode}_predict']=pred
                    profiles[f'{site}_{arm}_{radius}_{mode}_held']=held
        print(site,'local refinement complete',flush=True)
    pd.DataFrame(outs).to_csv(ROOT/'blind_refinement.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(weightrows).to_csv(ROOT/'refinement_weights.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(ROOT/'refinement_profiles.npz',angle=ANG,**profiles)
    names=['validate_refinement.py','blind_refinement.csv','refinement_weights.csv','refinement_profiles.npz']
    (ROOT/'refinement_freeze.json').write_text(json.dumps({n:old.core.base.sha(ROOT/n) for n in names},indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
