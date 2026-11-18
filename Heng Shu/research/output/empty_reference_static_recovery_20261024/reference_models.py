"""Bounded alternative references; no state-specific fitted hardware responses."""
import numpy as np
from common import *
def make_reference(x,kind):
    z=x[:CFG['prefix_packets']];anchor=0
    if kind!='first':
        features=np.einsum('nmk,nlk->nkml',z,z.conj()).reshape(len(z),-1)
        anchor=int(np.argmin(np.sum(abs(features-features.mean(0))**2,axis=1)))
    ref=z[anchor].copy();accepted=np.ones(1,bool)
    if kind=='aligned_robust_mean':
        zz=[];accepted=[]
        for a in z:
            t,p,c,e=rat.fit(a,ref);ok=c>=CFG['packet_corr_min'] and not e;accepted.append(ok)
            if ok:zz.append(rat.apply(a,t,p))
        if zz:
            # Complex mean only AFTER one common correction per packet; no chain reset.
            ref=rat.unit(np.mean(zz,axis=0))
    return ref,dict(anchor_offset=anchor,prefix_template_support=float(np.mean(accepted)),reference_valid=bool(np.mean(accepted)>=CFG['minimum_prefix_accept_fraction']))
def gate(x,ref):
    ds=[];ps=[];pred=[]
    for a in x[:CFG['prefix_packets']]:
        fits=[rat.fit(a[m:m+1],ref[m:m+1]) for m in range(3)]
        tt=np.array([f[0] for f in fits]);pp=np.array([f[1] for f in fits]);ds.append(np.max(abs(rat.wrap_delay(tt-tt[0]))));ps.append(np.max(abs(rat.wrap_phase(pp-pp[0])))*180/np.pi)
        pred.append(rat.prediction(a,ref)[-1][1])
    d=float(np.quantile(ds,.9));p=float(np.quantile(ps,.9));c=float(np.median(pred))
    return dict(chain_delay_p90_ns=d,chain_phase_p90_deg=p,prefix_predictive_correlation=c,common_pass=bool(d<=CFG['chain_delay_p90_ns'] and p<=CFG['chain_phase_p90_deg'] and c>=CFG['prediction_corr_median']))
def forward(x,ref,mode,passed):
    x=rat.unit(x);out=x.copy();reference=ref.copy();count=1;params=[]
    for a in x:
        if mode=='B':
            t=0.;p=float(np.angle(np.sum(a.conj()*reference)));c=rat.score(a,reference,np.ones_like(a.real))[1];edge=False;ok=True
        else:
            t,p,c,edge=rat.fit(a,reference);ok=passed and c>=CFG['packet_corr_min'] and not edge
        if ok:
            z=rat.apply(a,t,p);reference+=(z-reference)/(count+1);count+=1
        else:z=a
        out[len(params)]=z;params.append([t,p,c,float(edge),float(ok)])
    return out,np.asarray(params)
def predict(x,ref):
    train,test=rat.crossfit_masks();rows=[]
    for i in range(CFG['prefix_packets'],len(x),CFG['evaluation_stride']):
        a=x[i];t,p,co,edge=rat.fit(a,ref,train);td,pd,_,_=rat.fit(a,ref,test);bp=np.angle(np.sum(a.conj()*ref*train))
        sa=rat.score(a,ref,test);sb=rat.score(rat.apply(a,0.,bp),ref,test);sc=rat.score(rat.apply(a,t,p),ref,test)
        rows.append([i,t,td,float(rat.wrap_delay(td-t)),sa[1],sb[2],sc[1],sc[2],edge])
    return np.asarray(rows)
