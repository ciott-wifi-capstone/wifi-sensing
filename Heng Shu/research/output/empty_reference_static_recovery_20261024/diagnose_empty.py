"""Per-run reference checks plus invariant repeatability; full records retained."""
import argparse,itertools
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
from common import *
from reference_models import *
def run(role):
    assert sha(R/'config.json')==(R/'config.sha256').read_text().strip()
    assert load(R/'results/synthetic_preservation.json')['all_pass']
    if role=='development':ids=CFG['empty_development'];variants=CFG['reference_variants']
    else:
        sel=load(R/'results/development_selection.json');ids=CFG['empty_within_holdout']+CFG['empty_cross_holdout'];variants=['first',sel['selected_variant']];variants=list(dict.fromkeys(variants))
    meta=pd.read_csv(R/'empty_manifest.csv').set_index('index')
    for idx in ids:
        dest=R/'results'/f'reference_{idx:03d}.json'
        if dest.exists():continue
        rows=[]
        for state in rat.MAP:
            v=np.load(R/'cache'/f'{idx:03d}_{state}.npz');x=v['x'];base=dict(index=idx,capture=meta.loc[idx,'capture_id'],role=meta.loc[idx,'new_role'],state=state,packets=len(x))
            for kind in variants:
                ref,info=make_reference(x,kind);g=gate(x,ref);passed=g['common_pass'] and info['reference_valid'];b,pb=forward(x,ref,'B',True);c,pc=forward(x,ref,'C',passed);pred=predict(x,ref)
                rc=rat.covariance(c);rb=rat.covariance(b);ra=v['covariance'];a=rat.steering(state);sa=rat.music(ra,a);sc=rat.music(rc,a)
                assert distance(ra,rb)<1e-10 and rat.pair_error(x,c)<1e-8
                data=base|dict(variant=kind,**info,**g,actual_model_applied=bool(passed),accepted_fraction=float(pc[:,4].mean()),B_cov_change=distance(ra,rb),C_cov_change=distance(ra,rc),spatial_change=distance(spatial(x),spatial(c)),pair_phase_change_deg=rat.pair_error(x,c),
                    predicted_corr_A=float(np.median(pred[:,4])),predicted_corr_C=float(np.median(pred[:,6])),pred_delay_iqr_A_ns=rat.iqr(pred[:,2]),pred_delay_iqr_C_ns=rat.iqr(pred[:,3]),pred_phase_B_sd_deg=rat.circ_sd(pred[:,5])*180/np.pi,pred_phase_C_sd_deg=rat.circ_sd(pred[:,7])*180/np.pi,
                    C_spectrum_rms_change=float(np.sqrt(np.mean((sc-sa)**2))))
                rows.append(data);np.savez_compressed(R/'parameters'/f'{idx:03d}_{state}_{kind}.npz',reference=ref,B_parameters=pb,C_parameters=pc,prediction=pred,cov_A=ra,cov_B=rb,cov_C=rc,spectrum_A=sa,spectrum_C=sc)
                print(idx,state,kind,'gate',passed,'chain-phase',round(g['chain_phase_p90_deg'],2),'corr',round(data['predicted_corr_C'],3),flush=True)
        dump(dest,rows)
    allrows=[]
    for p in (R/'results').glob('reference_*.json'):allrows.extend(load(p))
    pd.DataFrame(allrows).to_csv(R/'results/reference_checks.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['development','heldout']);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.role)
