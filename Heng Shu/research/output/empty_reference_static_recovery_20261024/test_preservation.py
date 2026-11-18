"""Test new references on static multipath, common nuisance and negative controls."""
import numpy as np
from threadpoolctl import threadpool_limits
from common import *
from reference_models import make_reference,gate,forward
def main():
    rng=np.random.default_rng(241024);n=160;ys=rat.Y[np.array(rat.MAP['C1'])-1]
    atom=np.exp(2j*np.pi*rat.FC/299792458*ys[None,:,None]*np.sin(np.deg2rad([0,30,-40]))[:,None,None]-2j*np.pi*rat.F[None,None,:]*np.array([10,90,190])[:,None,None]*1e-9)
    h=np.sum(np.array([1,.5j,.3])[:,None,None]*atom,axis=0);truth=rat.unit(np.tile(h,(n,1,1)));tau=rng.uniform(-40,40,n);ph=rng.uniform(-np.pi,np.pi,n);x=rat.apply(truth,-tau,-ph);rows=[]
    for kind in CFG['reference_variants']:
        ref,info=make_reference(x,kind);g=gate(x,ref);z,p=forward(x,ref,'C',g['common_pass'] and info['reference_valid']);b,_=forward(x,ref,'B',True)
        delta=rat.wrap_delay(p[:,0]-tau);err=float(np.std(rat.wrap_delay(delta-np.median(delta))));pair=rat.pair_error(x,z);sd=distance(spatial(x),spatial(z));cov=distance(rat.covariance(x),rat.covariance(b))
        clean_ref,inf=make_reference(truth,kind);clean,pc=forward(truth,clean_ref,'C',True);noerr=float(np.max(abs(clean-truth)))
        bad=x.copy();bad[:,1]*=np.exp(1j*np.linspace(0,2,n))[:,None];badref,bi=make_reference(bad,kind);bg=gate(bad,badref)
        rows.append(dict(variant=kind,common_pass=g['common_pass'],delay_rmse_after_single_gauge_ns=err,pair_phase_error_deg=pair,spatial_change=sd,B_cov_change=cov,no_error_change=noerr,noncommon_gate_pass=bg['common_pass'],passed=bool(g['common_pass'] and err<.01 and pair<1e-8 and sd<1e-10 and cov<1e-10 and noerr<1e-8 and not bg['common_pass'])))
    # Common phase/timing cannot change same-frequency spatial covariance.
    # Fixed hardware is deliberately not removed; no human truth here.
    dump(R/'results/synthetic_preservation.json',dict(tests=rows,all_pass=all(r['passed'] for r in rows),config_sha256=sha(R/'config.json')));print(rows);assert all(r['passed'] for r in rows)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
