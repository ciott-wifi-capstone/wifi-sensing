"""Check complete delivered tables and arrays against authoritative inputs."""
import zipfile
import pandas as pd,numpy as np
from common import *
def main():
    checks=[]
    def check(name,condition,evidence):
        assert condition,name
        checks.append(dict(requirement=name,status='verified',evidence=evidence))
    check('Frozen numeric configuration',sha(R/'config.json')==(R/'config.sha256').read_text().strip(),'config.json/config.sha256')
    f=load(R/'provenance/confirmed_facts.json')
    check('Actual acquisition relationships',f['empty_pair']==['E1','E3'] and f['occupied_relation'].startswith('different'),'confirmed_facts.json')
    for rec in load(R/'provenance/sources.json'):
        check('Source unchanged: '+Path(rec['source']).name,sha(rec['source'])==rec['sha256']==sha(R/rec['snapshot']),rec['snapshot'])
    for name in ['empty_manifest.csv','primary_manifest.csv']:
        mm=pd.read_csv(R/name);check('Raw hashes: '+name,all(sha(row.path)==row.sha256 for row in mm.itertuples(index=False)),name)
    empty=pd.read_csv(R/'empty_manifest.csv');check('A occupied duplicate excluded',not empty.batch.str.contains('143435_student01_38459').any(),'empty_manifest.csv')
    sa=pd.read_csv(R/'results/source_audit.csv');st=pd.read_csv(R/'results/state_audit.csv')
    check('12 source and 36 state audits',len(sa)==12 and len(st)==36,'source_audit.csv/state_audit.csv')
    check('Relative time translation invariant',(sa.relative_timestamp_shift_max_error_ns.dropna()==0).all(),'source_audit.csv')
    for i in [78,79,80,84,85,86]:
        for state in rat.MAP:
            p=R/'cache'/f'{i:03d}_{state}.npz'
            with zipfile.ZipFile(p) as z:assert z.testzip() is None
            v=np.load(p);old=np.load(OLD/'packets'/f'{i:03d}_{state}.npz')
            assert v['x'].shape[1:]==(3,30) and np.isfinite(v['x']).all()
            assert len(v['packet_id'])==len(v['x']) and (np.diff(v['packet_id'])>0).all()
            assert np.array_equal(v['packet_id'],old['packet_id']) and distance(v['covariance'],old['cov_A'])<1e-10
    check('All 36 caches readable; baseline selection and covariance match',True,'cache/*.npz and original packet_id/cov_A')
    syn=load(R/'results/synthetic_preservation.json');check('Three preservation and negative-control checks',syn['all_pass'] and len(syn['tests'])==3 and all(t['passed'] and not t['noncommon_gate_pass'] for t in syn['tests']),'synthetic_preservation.json')
    ref=pd.read_csv(R/'results/reference_checks.csv');sel=load(R/'results/development_selection.json')
    check('Selection only E1 development',sel['development_indices']==[78,79] and len(sel['scores'])==3,'development_selection.json')
    check('Complete 36 development + 48 holdout checks',len(ref[ref['index'].isin([78,79])])==36 and len(ref[~ref['index'].isin([78,79])])==48,'reference_checks.csv')
    check('All 36 selected-reference records rejected',len(ref[ref.variant==sel['selected_variant']])==36 and not ref.actual_model_applied.any() and (ref.accepted_fraction==0).all(),'reference_checks.csv')
    for row in ref.itertuples(index=False):
        v=np.load(R/'parameters'/f'{row.index:03d}_{row.state}_{row.variant}.npz')
        assert (v['C_parameters'][:,4]==0).all()
        assert distance(v['cov_A'],v['cov_B'])<1e-10 and distance(v['cov_A'],v['cov_C'])<1e-10
        assert np.allclose(v['spectrum_A'],v['spectrum_C'],atol=1e-12)
    check('All 84 A/B/C parameters and outputs consistent',True,'parameters/*.npz')
    pair=pd.read_csv(R/'results/empty_pair_comparisons.csv');null=pd.read_csv(R/'results/empty_null_spectral_changes.csv');cross=pd.read_csv(R/'results/cross_record_prediction.csv')
    check('450 pair sensitivity records without duplicate keys',len(pair)==450 and len(pair.drop_duplicates(['index_a','index_b','state','variant']))==450,'empty_pair_comparisons.csv')
    check('54 confirmed E1/E3 full comparisons',len(pair[(pair.comparison=='E1_E3')&(pair.variant=='full')])==54,'empty_pair_comparisons.csv')
    check('48 cross-record gates rejected',len(cross)==48 and not cross.common_pass.any() and not cross.actual_cross_alignment_applied.any(),'cross_record_prediction.csv')
    check('90 empty null pairs with 54 E1/E3 exceedances',len(null)==90 and int(null[null.comparison=='E1_E3'].threshold_exceeded.sum())==54,'empty_null_spectral_changes.csv')
    block=pd.read_csv(R/'results/within_run_blocks.csv');thr=pd.read_csv(R/'results/development_thresholds.csv').set_index('state')
    for state,z in block[block['index'].isin([78,79])].groupby('state'):
        for col in ['spatial_distance','amplitude_distance','covariance_distance']:assert np.isclose(thr.loc[state,col],max(.05,1.5*z[col].max()))
    check('Thresholds reconstruct from development blocks',True,'within_run_blocks.csv/development_thresholds.csv')
    reuse=pd.read_csv(R/'results/human_reuse_provenance.csv');pk=pd.read_csv(R/'results/human_diagnostic_peaks.csv');hs=pd.read_csv(R/'results/human_spectral_stability_pairs.csv');pt=pd.read_csv(R/'results/point_status.csv')
    for row in reuse.itertuples(index=False):
        assert sha(row.existing_output)==row.existing_output_sha256
        v=np.load(row.existing_output)
        for mode in ['A','C']:
            z=pk[(pk['index']==row.index)&(pk.state==row.state)&(pk.method==mode)].iloc[0]
            assert rat.peak(v['spectrum_'+mode])==(z.global_max_angle_parameter_deg,z.global_max_delay_parameter_ns)
    check('All 39 human runs and 468 diagnostic spectra traced',len(reuse)==234 and reuse['index'].nunique()==39 and len(pk)==468,'human_reuse_provenance.csv/human_diagnostic_peaks.csv')
    check('Every within-point run pair retained',len(hs)==(66+3*36)*6*2,'human_spectral_stability_pairs.csv')
    check('Four A statuses; no invented human or position estimate',len(pt)==4 and (pt.status=='A').all() and not pk.human_candidate_identified.any() and not pt.position_verified.any(),'point_status.csv')
    dec=load(R/'results/decision.json');check('Conditional recovery stop applied',not dec['background_recovery_gate_pass'] and dec['stage_decisions']['localization'].startswith('not entered'),'decision.json')
    rendered=load(R/'qa/render_manifest.json')
    check('47 pages rendered including 39 human atlas pages',len(rendered)==12 and sum(t['pages'] for t in rendered)==47 and sum(t['pages'] for t in rendered if 'human_atlas' in t['path'])==39,'qa/render_manifest.json')
    check('Rendered PDFs still current',all(sha(t['path'])==t['sha256'] for t in rendered),'qa/render_manifest.json')
    report=(R/'REPORT.md').read_text(encoding='utf-8')
    check('Report records facts, evidence, missing conditions and minimum acquisition',all(t in report for t in ['不是同一硬件会话','未执行','最小补充顺序','1.1454','0.3669','39页']),'REPORT.md')
    check('Visual review record exists',(R/'qa/VISUAL_REVIEW.md').is_file(),'qa/VISUAL_REVIEW.md')
    dump(R/'results/completion_audit.json',dict(all_pass=True,requirements=checks,conditional_not_run=['background whitening/projection','new human pilot or expansion','physical position/accuracy'],reason='Empty reference validation failed; the goal permits an evidence-backed finding of insufficient recoverability',scope='Existing-data bounded feasibility, not impossibility of every future method'))
    pd.DataFrame(checks).to_csv(R/'results/completion_requirements.csv',index=False)
    print('Verified',len(checks),'requirements')
if __name__=='__main__':main()
