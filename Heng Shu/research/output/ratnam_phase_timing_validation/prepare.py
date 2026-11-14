"""Freeze cohort selection and source snapshots before estimating any new spectra."""
from pathlib import Path
import hashlib, json, shutil
import pandas as pd

R=Path(__file__).resolve().parent
OLD=R.parent/'static_analysis_20261018'
TH=R.parent/'thesis_algorithm_draft_20261019'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    for d in ['provenance','results','packets','figures']:(R/d).mkdir(exist_ok=True)
    sources=[OLD/'analyze.py',OLD/'geometry_update.py',OLD/'geometry_update_20261019/inputs.csv',
             TH/'evidence/primary_four_point/cohort_manifest.csv',
             R.parent/'static_reanalysis_20261020/geometry_correction.json']
    provenance=[]
    for p in sources:
        dest=R/'provenance'/p.name
        if dest.exists(): assert sha(dest)==sha(p),'Source changed: preserve original snapshot'
        else: shutil.copy2(p,dest)
        provenance.append(dict(source=str(p),snapshot=str(dest.relative_to(R)),sha256=sha(p)))
    (R/'provenance/source_files.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    df=pd.read_csv(OLD/'geometry_update_20261019/inputs.csv').reset_index(names='index')
    assert len(df)==90 and df.sha256.nunique()==90
    assert not ((df.analysis_group.str.startswith('empty')) & (df.batch=='20261018_143435_student01_38459')).any()
    groups=['old+1','old-1','old+2','old-2','empty Oct03']
    rows=[]
    for _,row in df.iterrows():
        x=row.to_dict();g=x['analysis_group']; batches=sorted(df[df.analysis_group==g].batch.unique())
        rank=batches.index(x['batch'])
        x['batch_rank']=rank;x['role']='supplementary_not_scheduled'
        if g in groups:
            x['role']=['development','held_out'][rank] if rank<2 else 'expansion_not_scheduled'
        x['scheduled']=x['role'] in ['development','held_out']
        x['occupancy_truth']=0 if g.startswith('empty') else 1
        x['source_exists']=Path(x['path']).is_file()
        rows.append(x)
    out=pd.DataFrame(rows);out.to_csv(R/'input_manifest.csv',index=False)
    out[out.scheduled].to_csv(R/'bounded_manifest.csv',index=False)
    print(out[out.scheduled].groupby(['analysis_group','role','batch']).size().to_string())
    print('Scheduled runs',int(out.scheduled.sum()),'; max state analyses',int(out.scheduled.sum())*6)
    config={
      'version':1,'frequency_hz':[(n*312500.) for n in range(-58,59,4)],'fc_hz':5.23e9,
      'array_y_rxlocal_m':[(6.5-m)*.028 for m in range(1,13)],
      'state_map':{'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]},
      'global_tx':[0,0],'global_rx':[3.65,0],'global_to_rxlocal':'(3.65-X,-Y)',
      'gauge':'First retained packet separately per run/state; no absolute delay or cross-run phase origin',
      'delay_interval_ns':[-400,400],'delay_grid_ns':2.,'delay_period_ns':800.,'refine_tolerance_ns':.001,
      'prefix_packets':64,'min_packets':128,'correlation_accept':.90,
      'common_gate_chain_delay_p90_ns':5.,'common_gate_chain_phase_p90_deg':15.,
      'common_gate_prediction_corr_median':.90,'delay_edge_ns':398.,
      'numerical_cov_tolerance':1e-10,'pair_phase_tolerance_deg':1e-8,
      'synthetic_snr_db':30.,'synthetic_delay_rmse_limit_ns':2.,'synthetic_cpe_rmse_limit_deg':5.,
      'synthetic_relative_channel_error_limit':.1,'synthetic_no_error_limit':1e-8,
      'heldout_improvement_ratio':.90,'min_state_accept_fraction':.90,
      'notes':'Engineering acceptance limits, not calibrated physical error bounds. Frozen before real data. No GT tuning.',
      'method_C':'Conditional ML joint common-packet correction; causal first-packet gauge adaptation, not verbatim Algorithm 4 initialisation',
      'backward_pass':'Not enabled: optional offline algorithm excluded to preserve a causal temporal check',
      'noise_assumption':'Equal whitened packet-feature noise is an approximation; fixed responses not calibrated',
      'software':'python; single BLAS thread for repeatable timings'}
    p=R/'config.json'
    if p.exists(): assert json.loads(p.read_text())==config,'Do not silently change frozen config'
    else:p.write_text(json.dumps(config,indent=2),encoding='utf-8')
    (R/'config.sha256').write_text(sha(p)+'\n')

if __name__=='__main__':main()
