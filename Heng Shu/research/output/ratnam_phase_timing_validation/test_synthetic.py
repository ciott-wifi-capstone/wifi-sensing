"""Multipath recovery/invariance tests and deliberately invalid model controls."""
import json
import numpy as np
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits
from core import *
def run():
    rng=np.random.default_rng(20261024);n=300
    delays=np.array([25.,105.,205.]);angles=np.array([12.,-35.,48.]);amps=np.array([1.,.6j,-.35])
    ys=Y[np.array(MAP['C1'])-1]
    atom=np.exp(2j*np.pi*FC/299792458*ys[None,:,None]*np.sin(np.deg2rad(angles))[:,None,None]-2j*np.pi*F[None,None,:]*delays[:,None,None]*1e-9)
    h=np.sum(amps[:,None,None]*atom,axis=0);truth=np.tile(h,(n,1,1))
    tau=rng.uniform(-60,60,n);psi=rng.uniform(-np.pi,np.pi,n);tau[0]=0;psi[0]=0;rows=[]
    for snr in [None,CFG['synthetic_snr_db']]:
        obs=truth*np.exp(-1j*(2*np.pi*tau[:,None]*F*1e-9+psi[:,None]))[:,None,:]
        if snr is not None:
            sd=np.sqrt(np.mean(abs(h)**2)/10**(snr/10)/2);obs+=sd*(rng.normal(size=obs.shape)+1j*rng.normal(size=obs.shape))
        x=unit(obs);gate=common_gate(x);b,pb=correct(x,'B');c,pc=correct(x,'C',gate['common_model_pass'])
        ra=covariance(x);rb=covariance(b)
        d=wrap_delay(pc['tau_ns']-tau);d=wrap_delay(d-np.median(d))
        ph=wrap_phase(pc['psi_rad']-psi);ph=wrap_phase(ph-np.angle(np.exp(1j*ph).mean()))
        target=unit(truth);scale=np.vdot(target,c)/np.vdot(target,target);rel=float(np.linalg.norm(c-scale*target)/np.linalg.norm(target))
        grid=np.arange(-100.,351.,.25);basis=np.exp(2j*np.pi*F[:,None]*grid*1e-9)
        pa=abs(target.mean(0)[0]@basis)**2;pbroad=abs(c.mean(0)[0]@basis)**2
        ia=find_peaks(pa,distance=120)[0];ib=find_peaks(pbroad,distance=120)[0]
        ia=np.sort(ia[np.argsort(pa[ia])[-3:]]);ib=np.sort(ib[np.argsort(pbroad[ib])[-3:]])
        pdiff=float(np.max(abs(np.diff(grid[ia])-np.diff(grid[ib]))))
        a=steering('C1');sa=music(ra,a);sb=music(rb,a)
        row=dict(case='noiseless' if snr is None else f'snr_{snr:g}dB',**gate,cov_B_relative=float(np.linalg.norm(rb-ra)/np.linalg.norm(ra)),
            spectrum_B_maxdiff=float(np.max(abs(sa-sb))),relative_tau_rmse_ns=float(np.sqrt(np.mean(d*d))),
            relative_psi_rmse_deg=float(np.sqrt(np.mean(ph*ph))*180/np.pi),corrected_channel_error=rel,pair_phase_error_deg=pair_error(x,c),
            relative_path_spacing_error_ns=pdiff,accept_fraction=float(pc['accepted'].mean()))
        row['pass']=bool(gate['common_model_pass'] and row['cov_B_relative']<CFG['numerical_cov_tolerance'] and row['relative_tau_rmse_ns']<CFG['synthetic_delay_rmse_limit_ns'] and row['relative_psi_rmse_deg']<CFG['synthetic_cpe_rmse_limit_deg'] and rel<CFG['synthetic_relative_channel_error_limit'] and pdiff<2. and row['pair_phase_error_deg']<CFG['pair_phase_tolerance_deg'])
        rows.append(row)
    x=unit(truth);out,params=correct(x,'C');err=float(np.max(abs(out-x)))
    rows.append(dict(case='no_injected_error',max_abs_error=err,passed=err<CFG['synthetic_no_error_limit']))
    bad=truth.copy();bad[:,2]*=np.exp(-1j*(2*np.pi*F[None,:]*np.linspace(0,80,n)[:,None]*1e-9+np.linspace(0,2,n)[:,None]))
    g=common_gate(unit(bad));out,pp=correct(bad,'C',g['common_model_pass'])
    rows.append(dict(case='invalid_chain_specific_error',**g,unchanged_on_gate_failure=float(np.max(abs(out-unit(bad)))),passed=not g['common_model_pass'] and np.allclose(out,unit(bad))))
    dynamic=truth.copy();dynamic+=.15*np.sin(np.arange(n)*.08)[:,None,None]*atom[2]
    z=dynamic*np.exp(-1j*(2*np.pi*tau[:,None]*F*1e-9+psi[:,None]))[:,None,:]
    out,pp=correct(z,'C',True);pair=pair_error(unit(z),out);errs=[]
    for i in range(n):
        if pp['accepted'][i]:
            undo=out[i]*np.exp(-1j*(2*np.pi*F*(pp['tau_ns'][i]-tau[i])*1e-9+pp['psi_rad'][i]-psi[i]))
            errs.append(float(np.max(abs(undo-unit(dynamic[i])))))
    rows.append(dict(case='dynamic_path_structure',pair_phase_error_deg=pair,algebraic_preservation_max=max(errs),
                     note='Structural identity only; true common channel changes remain unidentifiable',passed=pair<CFG['pair_phase_tolerance_deg'] and max(errs)<1e-10))
    x=unit(truth*np.exp(-1j*(2*np.pi*tau[:,None]*F*1e-9+psi[:,None]))[:,None,:]);diff=[];wd=[];wp=[];grid=np.arange(-400,400,2.)
    for i in range(1,40):
        t,p,_,_=fit(x[i],x[0],refine=False);w=np.sum(x[i].conj()*x[0],axis=0)
        direct=grid[np.argmax(abs(w@np.exp(-2j*np.pi*F[:,None]*grid*1e-9)))];diff.append(abs(float(wrap_delay(t-direct))))
        tt,ps,_,_=fit(x[i],x[0]);wt,wps,res=fit_wls(x[i],x[0],tt)
        wd.append(abs(float(wrap_delay(wt-tt))));wp.append(abs(float(wrap_phase(wps-ps))))
    rows.append(dict(case='fft_and_wls',fft_direct_grid_max_ns=max(diff),wls_ml_max_ns=max(wd),wls_ml_phase_max_rad=max(wp),passed=max(diff)<1e-10 and max(wd)<.01 and max(wp)<.01))
    result={'config_sha256':(R/'config.sha256').read_text().strip(),'tests':rows,'all_pass':all(r.get('pass',r.get('passed',False)) for r in rows)}
    (R/'results/synthetic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));assert result['all_pass']
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
