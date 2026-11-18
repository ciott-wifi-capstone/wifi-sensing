"""Robustness cases supplement, rather than replace, the original passing tests."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from core import *
def main():
    rng=np.random.default_rng(32);rows=[];n=160
    for state in ['G1','C1','C2']:
        ys=Y[np.array(MAP[state])-1]
        delays=np.array([35.,95.,185.]);angles=np.array([0.,25.,-40.]);amps=np.array([1.,.45j,-.25])
        atom=np.exp(2j*np.pi*FC/299792458*ys[None,:,None]*np.sin(np.deg2rad(angles))[:,None,None]-2j*np.pi*F[None,None,:]*delays[:,None,None]*1e-9)
        h=np.sum(amps[:,None,None]*atom,axis=0)
        # Unknown fixed chain/frequency response. Retaining a template does not remove it.
        fixed=np.exp(1j*rng.uniform(-np.pi,np.pi,(3,1)))*(1+.2*np.cos(np.arange(30)[None,:]*.4))
        h=h*fixed;truth=np.tile(h,(n,1,1));tau=rng.uniform(-70,70,n);ph=rng.uniform(-np.pi,np.pi,n);tau[0]=ph[0]=0
        obs=apply(truth,-tau,-ph);x=unit(obs);gate=common_gate(x);z,p=correct(x,'C',gate['common_model_pass'])
        err=np.linalg.norm(z-unit(truth))/np.linalg.norm(unit(truth))
        rows.append(dict(case='fixed_hardware_retained',state=state,relative_error=float(err),gate=gate['common_model_pass'],pass_=bool(err<.001 and gate['common_model_pass'])))
        # A varying path with NO hardware disturbance: quantify processing distortion,
        # not just algebraically undoing its estimated correction.
        truth=np.tile(h,(n,1,1))+.08*np.sin(np.arange(n)*.07)[:,None,None]*atom[1]*fixed
        xx=unit(truth);g=common_gate(xx);zz,pp=correct(xx,'C',g['common_model_pass'])
        error=float(np.linalg.norm(zz-xx)/np.linalg.norm(xx))
        rows.append(dict(case='weak_dynamic_no_nuisance',state=state,relative_change=error,
            fitted_timing_iqr_ns=iqr(pp['tau_ns']),fitted_phase_sd_deg=circ_sd(pp['psi_rad'])*180/np.pi,
            note='Nonzero correction of a real channel change demonstrates ambiguity; cannot promise all temporal motion preserved',pass_=error<.1))
    # Finite frequency sampling has a real delay alias; never claim it was resolved.
    x=unit(h);t1,p1,_,_=fit(apply(x,-10.,-.4),x);t2,p2,_,_=fit(apply(x,-810.,-.4),x)
    rows.append(dict(case='800ns_alias',delay1=t1,delay2=t2,phase_difference_rad=float(wrap_phase(p2-p1)),
                     pass_=abs(float(wrap_delay(t2-t1)))<.01))
    for t in [399.,-399.]:
        tt,pp,co,edge=fit(apply(x,-t,0.),x);rows.append(dict(case='search_edge',injected=t,estimated=tt,edge=bool(edge),pass_=bool(edge)))
    out={'tests':rows,'all_pass':all(x['pass_'] for x in rows)}
    (R/'results/synthetic_extended.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2));assert out['all_pass']
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
