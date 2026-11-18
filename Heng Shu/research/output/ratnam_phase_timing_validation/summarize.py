"""Tables from all completed runs; candidate metrics never labelled applied output."""
import json
import numpy as np
import pandas as pd
from core import R,CFG,wrap_delay
def main():
    states=[];methods=[];blocks=[];runs=[]
    for p in sorted((R/'results').glob('run_*.json')):
        d=json.loads(p.read_text());base={k:d[k] for k in ['index','group','batch','run','role','complete_window']}
        runs.append({k:v for k,v in d.items() if k!='states'})
        for s in d['states']:
            row=base|{k:v for k,v in s.items() if k not in ['methods','blocks']}
            row['prediction_is_candidate_only']=True
            states.append(row)
            for m in s.get('methods',[]):methods.append(base|{'state':s['state']}|m)
            for b in s.get('blocks',[]):blocks.append(base|{'state':s['state']}|b)
    s=pd.DataFrame(states);m=pd.DataFrame(methods);b=pd.DataFrame(blocks)
    s.to_csv(R/'results/state_summary.csv',index=False);m.to_csv(R/'results/method_summary.csv',index=False);b.to_csv(R/'results/temporal_blocks.csv',index=False)
    pd.DataFrame(runs).to_csv(R/'results/run_summary.csv',index=False)
    cols=['common_model_pass','C_accept_fraction','C_tau_iqr_ns','pred_delay_iqr_A_ns','pred_delay_iqr_C_ns','pred_delay_ratio','pred_phase_ratio_C_B','pred_corr_A','pred_corr_C','wls_delay_diff_p95_ns','seconds_total']
    g=s.groupby(['group','role'])[cols].median();g.to_csv(R/'results/group_medians.csv')
    # Compare temporal blocks paired at run/state level before aggregating.
    temporal=b.groupby(['index','group','role','state','method']).agg(
        block_cov_median=('covariance_to_whole','median'),block_spec_median=('spectrum_to_whole','median'),
        block_angle_span=('angle_deg',lambda v:v.max()-v.min()),
        block_delay_span=('delay_ns',lambda v:float(np.ptp(wrap_delay(np.asarray(v)-v.iloc[0]))))).reset_index()
    temporal.to_csv(R/'results/temporal_stability.csv',index=False)
    pair=[]
    for keys,t in temporal.groupby(['index','group','role','state']):
        t=t.set_index('method');row=dict(zip(['index','group','role','state'],keys))
        for col in ['block_cov_median','block_spec_median','block_angle_span','block_delay_span']:
            row[col+'_A']=t.loc['A',col];row[col+'_C']=t.loc['C',col]
            row[col+'_ratio']=t.loc['C',col]/max(t.loc['A',col],1e-12)
        pair.append(row)
    pair=pd.DataFrame(pair);pair.to_csv(R/'results/temporal_paired.csv',index=False)
    # Between-run comparisons, no fitted offsets. Separate same-batch vs cross-batch.
    cross=[]
    for (group,state),t in m[m.method=='A'].groupby(['group','state']):
        ids=sorted(t['index'].unique())
        for i,idx in enumerate(ids):
            for jdx in ids[:i]:
                b1=t[t['index']==idx].iloc[0]['batch'];b2=t[t['index']==jdx].iloc[0]['batch']
                v1=np.load(R/'packets'/f'{idx:03d}_{state}.npz');v2=np.load(R/'packets'/f'{jdx:03d}_{state}.npz')
                for met in ['A','B','C']:
                    r1=v1['cov_'+met];r2=v2['cov_'+met];r1=r1/np.trace(r1);r2=r2/np.trace(r2)
                    cross.append(dict(group=group,state=state,index1=idx,index2=jdx,same_batch=b1==b2,method=met,
                        cov_distance=float(np.linalg.norm(r1-r2)),spectrum_rms=float(np.sqrt(np.mean((v1['spectrum_'+met]-v2['spectrum_'+met])**2)))))
    pd.DataFrame(cross).to_csv(R/'results/between_run_comparisons.csv',index=False)
    actual=[]
    for p in (R/'results').glob('actual_*.json'):actual.extend(json.loads(p.read_text()))
    if actual:pd.DataFrame(actual).to_csv(R/'results/actual_output_stability.csv',index=False)
    print(g.to_string());print('\nTemporal paired medians:');print(pair.groupby(['group','role'])[['block_cov_median_ratio','block_spec_median_ratio']].median().to_string())
if __name__=='__main__':main()
