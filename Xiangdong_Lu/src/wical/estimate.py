"""GT-blind independent-state AoA-only SAGE and multi-baseline profiles.

No target coordinates are imported. Unit-area beamspace powers retain all
grating lobes. Each tone/state has its own complex amplitude (profiled out).
The unchanged SAGE core checks the plane-wave one-path profile, not a new
coherent 12-channel SAGE or a multi-person estimator.
"""
import sys,time,argparse
import numpy as np
if __package__:
    from .common import *
    from .original.sage_adapter import load_core
else:
    from common import *
    sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent/'original'))
    from sage_adapter import load_core
PLANE=np.array([steering(g) for g in STATES]);NEAR=np.array([steering(g,True) for g in STATES])
METHODS=list(STATES)+['F4','J6_product','J6_consensus','J6_robust','F4_near','J6_near','G3_eq6','F4_eq4','G3_eq4','J6_product_eq6','J6_robust_eq6','J6_near_eq6']
POLICY=dict(date='2026-11-01 Australia/Sydney',title='Multi-Baseline Joint AoA Estimation Using a 12-Element Switched Antenna Array',
    physical_map=MAP,physical_y='(antenna_index-6.5)*0.028 metres',center_frequency_hz=FC,tones_hz=FREQ,plane_grid_deg=ANGLES,near_nuisance_ranges_m=RANGES,
    sign='exp(+j k y sin(theta)); inherited antenna-index orientation, independently unconfirmed; no sign reversal by GT',
    model='same inherited covariance-reference-vector AoA-only one-path per-tone SAGE profile; independent complex tone/state gains; no cross-state coherent samples',
    branches=['kong_phase','identity'],old_batch='identity only, no cross-batch calibration or background subtraction; descriptive channel directions',
    training='first 3 chronological new kong sessions; last 2 kong sessions untouched; training displays and thresholds exclude entire own session',
    phase='per-state/per-tone training-kong principal eigenvector / spherical nominal Tx steering; effective scene plus hardware, not pure hardware calibration',
    static_suppression='new: subtract session-balanced training-kong covariance before choosing reference column (chain 1 zero-based); old: no subtraction possible',
    normalization='each state spectrum divided by its mean over 181 plane-wave angles; positive floor 0.05 only inside log fusion',
    coarse='union: top 4 peaks of F4 plus top 2 peaks per G1-G4, min 8 degree separation; include +/-15 degrees of each, near scan +/-20 degrees; endpoints retained',
    product='exp(mean(log(max(unit-area-state-power,0.05)))) for six states, restricted to coarse support',
    consensus='median of six unit-area state powers, restricted to coarse support',
    gate='long state: >=80% of tone/reference-chain comparisons have R>=0.8 in every training session AND pooled leave-session-out effective phase RMS<=30deg; old has no transferable reference and gate=0',
    quality_weight='min(N/300,1) * contrast * captured_power_fraction * min(peak/noise,4)/4; contrast=(max/mean-1)/(max/mean+1); noise=max LOSO training peak; clipped [0,1]',
    robust='F4 is short evidence; log(F4) plus quality*gate times log(C1/C2), divided by 1+sum weights. If both gates reject, return exactly F4 and disclose fallback',
    near='exact spherical distances from array-centre angle and nuisance range; plane-wave FC approximation per tone as baseline; maximize fused score over fixed range grid, never use actual human range',
    budgets='full all safe data; eq6 uses m=min(floor(N_G3/6),min N_states) per state vs G3 6m; eq4 retains prior four-way m4 and G3 4m4',
    detection='method-specific maximum LOSO training full-run score, strict >; only scene-change candidate, never validated human detection. Positive delay does not rescue any empty-room candidate',
    GT_used=False,ToF_search=False,coherent_12Rx=False,mapping_primary='retained Intel mask3',mapping_sensitivity='public stable double-rank hypothesis; C++ tie order and historical firmware unresolved',
    reporting='every fixed method/branch/mapping/run, no winner selection; session cluster paired bootstrap and exact sign flip; unconditional nominal errors include rejected detections',
    preregistration='fixed-policy auditable reanalysis of previously known data, not a prospective blinded trial')
def template(cov,records,train):
    ss=sorted(set(records[j]['session'] for j in train))
    return np.mean([cov[[j for j in train if records[j]['session']==s],0].mean(0) for s in ss],axis=0)
def phase(bg,branch):
    if branch=='identity':return np.ones((6,30,3),complex)
    _,v=np.linalg.eigh(bg);q=v[:,:,:,-1];q=np.divide(q,abs(q),out=np.ones_like(q),where=abs(q)>1e-14);q*=q[:,:,1:2].conj()
    expected=[]
    for g in STATES:
        rr=np.hypot(3.65,coords(g));expected.append(np.exp(-2j*np.pi*FREQ[:,None]*(rr-rr[1])[None,:]/LIGHT))
    return q/np.array(expected)
def inputs(c,q,bg):return ((c-bg)*q.conj()[:,:,:,None]*q[:,:,None,:])[:,:,:,1]
def profiles(x,near=False):
    a=NEAR if near else PLANE
    return np.mean(abs(np.einsum('ska,sap->skp',x,a.conj()))**2,axis=1)/9
def long_gates(cov,records,train):
    sessions=sorted(set(records[j]['session'] for j in train));diff=[];cover=[]
    for s in sessions:
        own=[j for j in train if records[j]['session']==s];other=[j for j in train if records[j]['session']!=s]
        bg=template(cov,records,other);obs=template(cov,records,own)
        delta=np.angle(phase(obs,'kong_phase')/phase(bg,'kong_phase'))
        diff.append(delta[:,:,[0,2]])
        coh=abs(obs[:,:,[0,2],1])/np.sqrt(np.maximum(obs[:,:,[0,2],[0,2]].real*obs[:,:,1,1].real[:,:,None],1e-30))
        cover.append((coh>=.8).mean(axis=(1,2)))
    rms=np.degrees(np.sqrt(np.mean(np.array(diff)**2,axis=(0,2,3))))
    coverage=np.min(cover,axis=0);gate=(rms<=30)&(coverage>=.8)
    return gate,rms,coverage
def noise_for(cov,records,train,branch):
    scores=[]
    for j in train:
        tr=[k for k in train if records[k]['session']!=records[j]['session']]
        bg=template(cov,records,tr);q=phase(bg,branch)
        scores.append(profiles(inputs(cov[j,0],q,bg)).max(1))
    return np.max(scores,axis=0)
def coarse(p,width=15):
    f=norm(p[:4]).mean(0);ii=peaks(f,4)
    for pp in p[:4]:ii+=peaks(pp,2)
    return np.any(abs(ANGLES[:,None]-ANGLES[np.array(ii,dtype=int)][None,:])<=width,axis=1),sorted(set(ii))
def quality(p,x,n,noise):
    ratio=p.max(1)/np.maximum(p.mean(1),1e-30);contrast=(ratio-1)/(ratio+1)
    # |a^H x|^2 / (3 ||x||^2), averaged across independent tones.
    captured=np.clip(p.max(1)/np.maximum(np.mean(abs(x)**2,axis=(1,2)),1e-30),0,1)
    return np.nan_to_num(np.clip(np.minimum(n/300,1)*contrast*captured*np.minimum(p.max(1)/np.maximum(noise,1e-30),4)/4,0,1))
def fuse(p,pn,x,n,noise,gate):
    u=norm(p);short=u[:4].mean(0);mask,ii=coarse(p);maskn,_=coarse(p,20)
    w=quality(p,x,n,noise)*gate;w[:4]=0
    product=np.exp(np.log(np.maximum(u,.05)).mean(0));cons=np.median(u,axis=0)
    def joint(u0,short0,near=False):
        if w[4:].sum()==0:return short0.copy()
        return np.exp((np.log(np.maximum(short0,.05))+(w[4:,None]*np.log(np.maximum(u0[4:],.05))).sum(0))/(1+w[4:].sum()))
    robust=joint(u,short)
    # Normalize each spherical state jointly over fixed range-angle grid;
    # range is a nuisance profile, not a measured ToF or reported position.
    un=norm(pn);sn=un[:4].mean(0);jn=joint(un,sn,True)
    nearshort=sn.reshape(len(RANGES),len(ANGLES)).max(0);nearjoint=jn.reshape(len(RANGES),len(ANGLES)).max(0)
    if w[4:].sum()>0:robust=np.where(mask,robust,0);nearjoint=np.where(maskn,nearjoint,0)
    product=np.where(mask,product,0);cons=np.where(mask,cons,0)
    return dict(F4=short,J6_product=product,J6_consensus=cons,J6_robust=robust,F4_near=nearshort,J6_near=nearjoint),w,ii
def calculate(c,eq6,eq4,g6,g4,bg,q,noise,gate,counts):
    x=inputs(c,q,bg);p=profiles(x);pn=profiles(x,True)
    fp,w,ii=fuse(p,pn,x,counts,noise,gate)
    out={s:p[k] for k,s in enumerate(STATES)};out.update(fp)
    def single(cc,si):
        xx=((cc-bg[si])*q[si].conj()[:,:,None]*q[si][:,None,:])[:,:,1]
        return np.mean(abs(xx@PLANE[si].conj())**2,axis=0)/9
    out['G3_eq6']=single(g6,2);out['G3_eq4']=single(g4,2)
    xs=inputs(eq6,q,bg);ps=profiles(xs);ns=profiles(xs,True)
    m6=min(counts[2]//6,min(counts));fe,we,_=fuse(ps,ns,xs,np.full(6,m6),noise,gate)
    for k in ['J6_product','J6_robust','J6_near']:out[k+'_eq6']=fe[k]
    cor=q[:4].conj()[:,:,:,None]*q[:4][:,:,None,:];x4=((eq4-bg[:4])*cor)[:,:,:,1]
    p4=np.mean(abs(np.einsum('ska,sap->skp',x4,PLANE[:4].conj()))**2,axis=1)/9;out['F4_eq4']=norm(p4).mean(0)
    scores={}
    for k,v in out.items():
        a=int(np.argmax(v))
        if k in STATES:scores[k]=p[STATES.index(k)].max()
        elif k.startswith('G3_eq'):scores[k]=v.max()
        elif k=='F4_eq4':scores[k]=p4.mean(0).max()
        elif k.endswith('_eq6'):
            if k=='J6_product_eq6':scores[k]=ps.mean(0).max()
            else:
                weights=np.r_[np.ones(4)/4,we[4:]];arr=ns if k=='J6_near_eq6' else ps
                scores[k]=np.average(arr,axis=0,weights=weights).max()
        elif k=='F4':scores[k]=p[:4].mean(0).max()
        elif k=='F4_near':scores[k]=pn[:4].mean(0).max()
        elif k in ['J6_robust','J6_near']:
            weights=np.r_[np.ones(4)/4,w[4:]];arr=pn if k=='J6_near' else p;scores[k]=np.average(arr,axis=0,weights=weights).max()
        else:scores[k]=p.mean(0).max()
    return out,scores,x,p,pn,w,we,ii
