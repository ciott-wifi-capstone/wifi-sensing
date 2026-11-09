"""Freeze the corrected experiment relationships and bounded validation protocol."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
R=Path(__file__).resolve().parent;OLD=R.parent/'ratnam_phase_timing_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for name in ['provenance/ratnam','results','cache','figures','parameters']:(R/name).mkdir(parents=True,exist_ok=True)
    sources=[(OLD/'core.py','provenance/ratnam/core.py'),(OLD/'config.json','provenance/ratnam/config.json'),(OLD/'provenance/analyze.py','provenance/analyze.py'),(OLD/'input_manifest.csv','provenance/input_manifest.csv'),(OLD/'REPORT.md','provenance/previous_REPORT.md'),(OLD/'METHOD.md','provenance/previous_METHOD.md'),(OLD/'results/state_summary.csv','provenance/previous_state_summary.csv'),(OLD/'results/between_run_comparisons.csv','provenance/previous_between_run_comparisons.csv')]
    records=[]
    for src,dest in sources:
        dst=R/dest
        if dst.exists():assert sha(src)==sha(dst)
        else:shutil.copy2(src,dst)
        records.append(dict(source=str(src),snapshot=dest,sha256=sha(src)))
    (R/'provenance/sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    d=pd.read_csv(OLD/'input_manifest.csv');empty=d[d.analysis_group.str.startswith('empty')].copy()
    names=['20261003_130149_student01_6613','20261003_130256_student01_11355','20261018_151129_student01_118072','20261018_151420_student01_127667']
    mapping={b:'E'+str(i+1) for i,b in enumerate(names)}
    empty['capture_id']=empty.batch.map(mapping);empty['original_label']=empty.analysis_group
    empty['confirmed_relationship']=empty.capture_id.map({'E1':'continuous_empty_first','E3':'continuous_empty_second'}).fillna('not_confirmed')
    empty['new_role']=empty['index'].map({78:'development',79:'development',80:'within_capture_holdout',84:'cross_capture_holdout',85:'cross_capture_holdout',86:'cross_capture_holdout'}).fillna('provenance_only')
    empty.to_csv(R/'empty_manifest.csv',index=False)
    primary=d[d.analysis_group.str.startswith('old')].copy();primary['new_processing']='not_authorized_by_stage_gate_yet';primary.to_csv(R/'primary_manifest.csv',index=False)
    cfg=dict(version=1,empty_development=[78,79],empty_within_holdout=[80],empty_cross_holdout=[84,85,86],provenance_only=[81,82,83,87,88,89],
      user_confirmed_pair=['E1','E3'],pair_relation='same_scene_continuous_no_reboot_rewire_or_relocation_user_confirmed',
      occupied_session_relationship='unknown_pending_user',directory_timestamps_are_truth=False,
      prefix_packets=64,evaluation_stride=8,reference_variants=['first','invariant_medoid','aligned_robust_mean'],
      chain_delay_p90_ns=5.,chain_phase_p90_deg=15.,prediction_corr_median=.90,packet_corr_min=.90,edge_ns=398.,
      min_packets=128,minimum_prefix_accept_fraction=.90,minimum_valid_states_fraction=.80,
      invariant_repeatability_ratio_limit=1.5,repeatability_absolute_floor=.05,
      repeatability_note='Compare heldout distance with max development disjoint-block distance times 1.5 or 0.05 floor; engineering diagnostic not a calibrated significance test',
      guard_sensitivity_extra_ms=[0.,1.,2.],same_duration_seconds=10.,equal_packet_cap=512,
      amplitude_support_fraction=.1,maximum_reference_variants=3,temporal_blocks=4,
      background_shrinkage_candidates=[.05,.2],background_basis='full covariance whitening first; projection deferred if no validated background',
      background_false_candidate_target=.05,background_target_note='Engineering target at window level; too few independent runs for a 5 percent false-positive guarantee',
      maximum_human_pilot_indices=[3,15,24,33],human_truth_used_for_fitting=False,
      runtime='python; one BLAS thread',notes='No date/session inference from filenames; no raw edits; no thesis/Overleaf changes; new analysis stop gates retained')
    p=R/'config.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==cfg
    else:p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (R/'config.sha256').write_text(sha(p)+'\n',encoding='ascii')
    print(empty[['index','capture_id','run','new_role','sha256']].to_string(index=False));print('Config SHA256',sha(p))
if __name__=='__main__':main()
