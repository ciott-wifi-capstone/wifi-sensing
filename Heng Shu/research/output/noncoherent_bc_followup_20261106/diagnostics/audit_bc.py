"""Bounded covariance diagnostics; no peak selection using human geometry."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
OLD=OUT.parents[1]/'noncoherent_uniform_20261106'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD))
import predict_uniform as p

rows=[];decomp=[]
for site in 'ABCDE':
    data=np.load(p.OLD/f'{site}_selected.npz')['switched'].astype(complex)
    for j,s in enumerate(p.ST):
        m=np.array([p.cov(data[r,:,j]) for r in range(3)])
        bg=p.background(m)
        R1=bg['ref'];r1h=m[0,8:].mean(0)
        r2=m[1,:12].mean(0);r3=m[2].mean(0)
        norm=lambda x:np.linalg.norm(x,axis=(-2,-1))
        trace=lambda x:np.trace(x,axis1=-2,axis2=-1).real
        dr1=norm(r1h-R1);dr2=norm(r2-R1);dr3=norm(r3-R1)
        for label,target in [('R1held',r1h),('R2first',r2),('R2last',m[1,12:].mean(0)),('R3full',r3)]:
            projected=p.herm(bg['B']@target@bg['B'])
            e=np.linalg.eigvalsh(projected)
            ew=np.linalg.eigvalsh(p.herm(bg['Bw']@bg['W']@target@bg['W']@bg['Bw']))
            delta=np.linalg.eigvalsh(p.herm(target-R1))
            floor=np.maximum(bg['spread'],1e-10)
            raw_e=np.linalg.eigvalsh(p.herm(target))
            r=dict(point=site,state=s,record=label,
                r1_dominance_median=float(np.median(bg['e'][:,-1]/bg['e'].sum(1))),
                raw_dominance_median=float(np.median(raw_e[:,-1]/raw_e.sum(1))),
                bg_second_to_first_eigenvalue_median=float(np.median(e[:,-2]/np.maximum(e[:,-1],1e-15))),
                wbg_second_to_first_eigenvalue_median=float(np.median(ew[:,-2]/np.maximum(ew[:,-1],1e-15))),
                positive_modes_gt_R1floor=int(np.sum(delta>floor[:,None])),
                negative_modes_gt_R1floor=int(np.sum(delta<-floor[:,None])),
                fullrank_change_tones=int(np.sum(np.sum(abs(delta)>floor[:,None],axis=1)==3)),
                relative_covariance_change=float(np.median(norm(target-R1)/np.maximum(norm(R1),1e-15))),
                projected_power=float(trace(projected).mean()/2),
                top_mode_drift_vs_R1=p.reference_check(bg,target)['phase_p90'])
            rows.append(r)
        # Amplitude normalisation prevents power claims; these ratios compare change only.
        decomp.append(dict(point=site,state=s,
            R2change_over_R3change=float(np.median(dr2/np.maximum(dr3,1e-15))),
            R2change_over_R1heldchange=float(np.median(dr2/np.maximum(dr1,1e-15))),
            R3change_over_R1heldchange=float(np.median(dr3/np.maximum(dr1,1e-15))),
            condition_regularized_R1=float(np.median(np.linalg.cond(.9*R1+.1*trace(R1)[:,None,None]/3*p.EYE))),
            change_direction_cosine_R2_R3=float(np.median(np.einsum('kij,kij->k',(r2-R1).conj(),r3-R1).real/np.maximum(dr2*dr3,1e-15)))))
pd.DataFrame(rows).to_csv(OUT/'covariance_rank_diagnostics.csv',index=False)
pd.DataFrame(decomp).to_csv(OUT/'empty_human_change_diagnostics.csv',index=False)
print(pd.DataFrame(decomp).query('point in ["B","C"]').round(3).to_string(index=False))
print(pd.DataFrame(rows).query('point in ["B","C"] and record=="R2first"').round(3).to_string(index=False))
