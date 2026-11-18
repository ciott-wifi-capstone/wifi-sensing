"""Ratnam Eq27 conditional ML, adapted to three simultaneous receive chains.
Each chain retains its own reference. First-packet gauge replaces strong-LoS
initialisation: relative alignment only, never physical calibration.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import minimize_scalar
R=Path(__file__).resolve().parent
CFG=json.loads((R/'config.json').read_text())
F=np.array(CFG['frequency_hz']);K=len(F);PERIOD=CFG['delay_period_ns']
MAP=CFG['state_map'];Y=np.array(CFG['array_y_rxlocal_m']);FC=CFG['fc_hz']
ANG=np.arange(-90.,91.,3);TAU=np.arange(-400.,401.,4);M=15
def wrap_delay(t):return (np.asarray(t)+PERIOD/2)%PERIOD-PERIOD/2
def wrap_phase(t):return np.angle(np.exp(1j*t))
def unit(x):return x/(np.linalg.norm(x,axis=(-2,-1),keepdims=True)+1e-15)
def circ_sd(x):return float(np.sqrt(-2*np.log(np.clip(abs(np.mean(np.exp(1j*np.asarray(x)))),1e-15,1))))
def iqr(x):return float(np.subtract(*np.quantile(x,[.75,.25])))
def apply(x,tau,psi):
    rotation=np.exp(1j*(2*np.pi*np.asarray(tau)[...,None]*F*1e-9+np.asarray(psi)[...,None]))
    return x*rotation[...,None,:]
def fit(x,ref,mask=None,refine=True):
    """FFT exactly evaluates bounded 2ns grid, refined using original objective."""
    w=x.conj()*ref
    if mask is not None:w=w*mask
    w=w.sum(axis=0);n=int(round(PERIOD/CFG['delay_grid_ns']))
    tau=float(wrap_delay(int(np.argmax(abs(np.fft.fft(w,n=n))))*PERIOD/n))
    def value(t):return np.sum(w*np.exp(-2j*np.pi*F*t*1e-9))
    if refine:
        opt=minimize_scalar(lambda t:-abs(value(t))**2,bounds=(max(-400.,tau-2),min(400.,tau+2)),
            method='bounded',options={'xatol':CFG['refine_tolerance_ns'],'maxiter':40})
        if opt.success and abs(value(opt.x))>=abs(value(tau)):tau=float(opt.x)
    psi=float(np.angle(value(tau)))
    if mask is None:mask=np.ones_like(x.real)
    corr=float(abs(value(tau))/max(np.linalg.norm(x*mask)*np.linalg.norm(ref*mask),1e-15))
    return tau,psi,corr,abs(tau)>=CFG['delay_edge_ns']
def fit_wls(x,ref,coarse_tau):
    """Eq28-inspired local WLS; 7-neighbour robust unwrap, coarse ML seed.
    Empirical weight mask differs from paper's unit-scaled |b|^2>.1 threshold.
    """
    omega=x.conj()*ref*np.exp(-2j*np.pi*F*coarse_tau*1e-9)[None,:]
    center=np.angle(omega.sum());vals=[];weights=[];freqs=[]
    for m in range(len(x)):
        w=abs(omega[m]);ids=np.where(w>.1*np.median(w[w>0]))[0] if np.any(w>0) else []
        if len(ids)<5:continue
        z=omega[m,ids];smooth=np.array([z[max(0,i-3):min(len(z),i+4)].sum() for i in range(len(z))])
        base=np.unwrap(np.angle(smooth));ph=wrap_phase(np.angle(z)-base)+base
        ph-=2*np.pi*np.round((np.average(ph,weights=w[ids])-center)/(2*np.pi))
        vals.extend(ph);weights.extend(w[ids]);freqs.extend(F[ids])
    if len(vals)<5:return np.nan,np.nan,np.inf
    a=np.column_stack([2*np.pi*np.array(freqs)*1e-9,np.ones(len(vals))]);sw=np.sqrt(weights)
    b=np.linalg.lstsq(a*sw[:,None],np.array(vals)*sw,rcond=None)[0]
    residual=float(np.sqrt(np.average(wrap_phase(np.array(vals)-a@b)**2,weights=weights)))
    return float(wrap_delay(coarse_tau+b[0])),float(wrap_phase(b[1])),residual
def crossfit_masks():
    train=np.zeros((3,K));train[:2,np.arange(K)%3!=2]=1
    return train,1-train
def score(x,ref,mask):
    z=np.sum(x.conj()*ref*mask);den=max(np.linalg.norm(x*mask)*np.linalg.norm(ref*mask),1e-15)
    return float(z.real/den),float(abs(z)/den),float(np.angle(z))
def prediction(x,ref):
    train,test=crossfit_masks();t,p,_,edge=fit(x,ref,train)
    return t,p,edge,score(x,ref,test),score(apply(x,t,p),ref,test)
def common_gate(x):
    """Prefix gate. Independent chain fits are diagnostics, NEVER applied."""
    n=min(len(x),CFG['prefix_packets']);ref=x[0];delays=[];phases=[];pcorr=[]
    for p in range(1,n):
        a=[fit(x[p,m:m+1],ref[m:m+1]) for m in range(3)]
        ds=np.array([z[0] for z in a]);ps=np.array([z[1] for z in a])
        delays.append(float(np.max(abs(wrap_delay(ds-ds[0])))))
        phases.append(float(np.max(abs(wrap_phase(ps-ps[0]))))*180/np.pi)
        pcorr.append(prediction(x[p],ref)[-1][1])
    d=float(np.quantile(delays,.9));p=float(np.quantile(phases,.9));c=float(np.median(pcorr))
    ok=d<=CFG['common_gate_chain_delay_p90_ns'] and p<=CFG['common_gate_chain_phase_p90_deg'] and c>=CFG['common_gate_prediction_corr_median']
    return dict(common_model_pass=bool(ok),chain_delay_p90_ns=d,chain_phase_p90_deg=p,prediction_corr_prefix=c,prefix_packets=n)
def correct(x,mode,gate_pass=True):
    """Causal running reference, no future samples. Prefix gate itself needs 64 packets.
    Failed C packets retained unchanged and excluded from corrected-reference updates.
    """
    x=unit(np.asarray(x,dtype=np.complex128));out=x.copy();n=len(x)
    tau=np.zeros(n);psi=np.zeros(n);corr=np.ones(n);edge=np.zeros(n,bool);accepted=np.ones(n,bool)
    ref=x[0].copy();count=1;start=time.perf_counter()
    for p in range(1,n):
        if mode=='B':
            psi[p]=np.angle(np.sum(x[p].conj()*ref));corr[p]=abs(np.sum(x[p].conj()*ref))/max(np.linalg.norm(ref),1e-15)
        else:
            tau[p],psi[p],corr[p],edge[p]=fit(x[p],ref)
            accepted[p]=gate_pass and corr[p]>=CFG['correlation_accept'] and not edge[p]
        if accepted[p]:
            out[p]=apply(x[p],tau[p],psi[p]);count+=1;ref+=(out[p]-ref)/count
    if mode=='C' and not gate_pass:accepted[:]=False;out=x.copy()
    return out,dict(tau_ns=tau,psi_rad=psi,correlation=corr,edge=edge,accepted=accepted,seconds=time.perf_counter()-start)
def covariance(x):
    x=np.asarray(x,dtype=np.complex128);x=x/(np.linalg.norm(x,axis=(1,2),keepdims=True)+1e-12)
    r=np.zeros((3*M,3*M),complex)
    for k in range(K-M+1):
        z=x[:,:,k:k+M].reshape(len(x),-1);r+=z.T@z.conj()/len(z)
    return r/(K-M+1)
def steering(state):
    ys=Y[np.array(MAP[state])-1];delta=-ys[:,None]*np.sin(np.deg2rad(ANG))[None,:]
    a=np.exp(-2j*np.pi*(FC/299792458*np.repeat(delta,M,axis=0)[:,:,None]+np.tile(F[:M],3)[:,None,None]*TAU[None,None,:]*1e-9))
    return a.reshape(3*M,-1)
def music(r,a,q=3):
    _,v=np.linalg.eigh(r);den=np.sum(abs(v[:,:-q].conj().T@a)**2,axis=0)
    s=(1/np.maximum(den,1e-12)).reshape(len(ANG),len(TAU));return s/s.max()
def pair_error(a,b):
    worst=0.
    for m,n in [(0,1),(1,2),(0,2)]:
        z=a[:,m]*a[:,n].conj();zz=b[:,m]*b[:,n].conj();good=(abs(z)>1e-14)&(abs(zz)>1e-14)
        if good.any():worst=max(worst,float(np.max(abs(np.angle(zz[good]*z[good].conj()))))*180/np.pi)
    return worst
def peak(s):
    i,j=np.unravel_index(np.argmax(s),s.shape);return float(ANG[i]),float(TAU[j])
