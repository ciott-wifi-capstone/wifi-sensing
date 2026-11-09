"""Record holdout gates and freeze bounded extension before opening its raw files."""
import json
import pandas as pd
from core import R,CFG
def main():
    s=pd.read_csv(R/'results/state_summary.csv');t=pd.read_csv(R/'results/temporal_paired.csv')
    h=s[(s.role=='held_out')&s.group.str.startswith('old')];ht=t[(t.role=='held_out')&t.group.str.startswith('old')]
    good=(h.common_model_pass & (h.C_accept_fraction>=CFG['min_state_accept_fraction']) & (h.pred_delay_ratio<=CFG['heldout_improvement_ratio']))
    ok=bool(good.mean()>=.9 and ht.block_cov_median_ratio.median()<=1 and ht.block_spec_median_ratio.median()<=1)
    obj={'expand_primary':ok,'config_sha256':(R/'config.sha256').read_text().strip(),'heldout_primary_states':len(h),
         'primary_states_passing_common_and_acceptance_and_predictive_gates':int(good.sum()),
         'median_temporal_covariance_ratio':float(ht.block_cov_median_ratio.median()),
         'median_temporal_spectrum_ratio':float(ht.block_spec_median_ratio.median()),
         'interpretation':'Relative within-state gains justify a fixed-parameter extension; empty common-model failures forbid shared empty/occupied calibration claims.',
         'selection':'All remaining primary occupied batches; first chronological batch of each A-D and both Oct18 empty batches. No quality/peak selection.'}
    (R/'results/expansion_decision.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    assert ok
    data=pd.read_csv(R/'input_manifest.csv');parts=[]
    for _,x in data.iterrows():
        if x.role=='expansion_not_scheduled':x=x.copy();x['role']='expansion';parts.append(x)
        elif (x.analysis_group in ['A','B','C','D'] and x.batch_rank==0) or x.analysis_group=='empty Oct18':
            x=x.copy();x['role']='supplementary';parts.append(x)
    out=pd.DataFrame(parts);out.to_csv(R/'extension_manifest.csv',index=False)
    print(json.dumps(obj,indent=2));print(out.groupby(['analysis_group','role','batch']).size().to_string())
if __name__=='__main__':main()
