"""Read-only numeric audit of frozen empty validation; R2 is not processed."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import validate_empty_models as run
old=run.old
frozen=json.loads((R/'empty_stage_freeze.json').read_text('utf8'))
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
params=pd.read_csv(R/'model_parameters.csv');envsaved=pd.read_csv(R/'R1_calibration_envelopes.csv')
hsaved=pd.read_csv(R/'held_empty_checks.csv')
matrices=0;tones=0;diff_tones=0;gate_differences=0;max_tol=0.;max_env_error=0.;full_gate_reproduce=True
envelope_rows=0
for point in 'ABCDE':
    z=np.load(old.OLD/f'{point}_selected.npz')
    empty={s:{'R1':z['switched'][0,:,i].astype(complex),'R3':z['switched'][2,:,i].astype(complex)} for i,s in enumerate(old.ST)}
    empty['G2_control']={'R1':z['g2'][0].astype(complex),'R3':z['g2'][2].astype(complex)}
    for state,data in empty.items():
        mom={k:old.cov(v) for k,v in data.items()};scale={k:run.proxy(v) for k,v in data.items()}
        bg=old.background(np.stack([mom['R1'],mom['R1'],mom['R3']]))
        B=bg['B'];ref=bg['ref'];floor=run.pcd.training_change_floor(bg);qref=scale['R1'][:8].mean()
        calibration=np.stack([run.norm_spectrum(run.project(B,mom['R1'][i:i+4].mean(0)-ref)) for i in [8,12]])
        for band,mask in [('full',run.HELD),('low',run.HELD&(np.arange(30)<15)),('high',run.HELD&(np.arange(30)>=15))]:
            # Explicit block-by-block quantiles independently reproduce the vectorized envelope.
            manual=[max(np.quantile(calibration[b,mask],p) for b in range(2)) for p in [.5,.9]]
            row=envsaved[(envsaved.point==point)&(envsaved.state==state)&(envsaved.band==band)].iloc[0]
            max_env_error=max(max_env_error,float(np.max(abs(np.array(manual)-np.array([row['median'],row['p90']])))))
            envelope_rows+=1
        for record,starts in [('R1',[16,20]),('R3',[8,12,16,20])]:
            for i in starts:
                delta=run.project(B,mom[record][i:i+4].mean(0)-ref)
                u=scale[record][i:i+4].mean()/qref-1
                vals={'STATIC':np.linalg.eigvalsh(delta)}
                for model in run.MODELS:
                    row=params[(params.point==point)&(params.state==state)&(params.model==model)].iloc[0]
                    beta=np.array(json.loads(row.coefficients));N=np.einsum('kmnd,d->kmn',run.columns(B,model),beta)
                    vals[model]=np.linalg.eigvalsh(old.herm(delta-u*N))
                counts={}
                for name,v in vals.items():
                    tol=1e-10*np.maximum(1.,np.max(abs(v),axis=1))
                    fixed=(v>floor[:,None]+1e-10).sum(1)
                    scaled=(v>floor[:,None]+tol[:,None]).sum(1)
                    diff_tones+=int(np.sum(fixed!=scaled));matrices+=1;tones+=len(fixed);max_tol=max(max_tol,float(tol.max()))
                    counts[name]=(fixed,scaled)
                for model in run.MODELS:
                    for band,mask in [('full',run.HELD),('low',run.HELD&(np.arange(30)<15)),('high',run.HELD&(np.arange(30)>=15))]:
                        a=np.sum(counts[model][0][mask]>0)<=np.sum(counts['STATIC'][0][mask]>0)
                        b=np.sum(counts[model][1][mask]>0)<=np.sum(counts['STATIC'][1][mask]>0)
                        gate_differences+=int(a!=b)
                        saved=hsaved[(hsaved.point==point)&(hsaved.state==state)&(hsaved.model==model)&(hsaved.block==f'{record}_{i}_{i+4}')&(hsaved.band==band)].iloc[0].gate_positive_not_increased
                        full_gate_reproduce=bool(full_gate_reproduce and bool(a)==bool(saved))
out=dict(frozen_hashes_verified=True,R2_processed=False,models=['STATIC','NSI','NSD'],state_records=35,held_blocks_per_state=6,matrix_bundles=matrices,tone_matrices=tones,positive_rank_count_differences=diff_tones,positive_not_increased_gate_differences=gate_differences,positive_not_increased_saved_gate_reproduced=full_gate_reproduce,maximum_scaled_numeric_tolerance=max_tol,envelopes_checked=envelope_rows,maximum_envelope_reproduction_error=max_env_error,original_PCD_floor_reused=True,frozen_runner_or_results_modified=False)
(R/'review/tolerance_equivalence.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
assert diff_tones==0 and gate_differences==0 and full_gate_reproduce and max_env_error<1e-12
