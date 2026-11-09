"""Final read-only input verification and bounded saved-output audit. No CSI refit."""
import hashlib,json,platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy,matplotlib
from core import R,CFG,MAP

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    checks=[]
    def check(name,ok,detail):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
    cfgsha=sha(R/'config.json')
    check('frozen_config',cfgsha==(R/'config.sha256').read_text().strip(),cfgsha)
    sources=json.loads((R/'provenance/source_files.json').read_text())
    for s in sources:
        check('source_unchanged:'+s['snapshot'],sha(s['source'])==sha(R/s['snapshot'])==s['sha256'],s['source'])
    full=pd.read_csv(R/'input_manifest.csv')
    scheduled=pd.concat([pd.read_csv(R/'bounded_manifest.csv'),pd.read_csv(R/'extension_manifest.csv')],ignore_index=True)
    check('90_deduplicated_sources',len(full)==full.sha256.nunique()==90,'Unique raw SHA256; no repeated A/empty batch')
    check('A_duplicate_not_empty',not ((full.batch=='20261018_143435_student01_38459') & full.analysis_group.str.startswith('empty')).any(),'Previously confirmed occupied A')
    runs=[json.loads(p.read_text()) for p in sorted((R/'results').glob('run_*.json'))]
    check('scheduled_runs_complete',set(scheduled['index'])=={r['index'] for r in runs} and len(runs)==len(scheduled)==63,'30 bounded + 33 extension, complete per-run outputs')
    check('batch_split_disjoint',not any(z.role.nunique()>1 for _,z in scheduled.groupby('batch')),'No batch used in multiple roles')
    input_audit=[]
    for _,r in full.iterrows():
        ok=sha(r.path)==r.sha256
        input_audit.append(dict(index=r['index'],sha256=r.sha256,path=r.path,unchanged=ok,processed=int(r['index']) in set(scheduled['index'])))
    pd.DataFrame(input_audit).to_csv(R/'results/input_integrity.csv',index=False)
    check('all_raw_inputs_unchanged',all(x['unchanged'] for x in input_audit),'All 90 CSI hashes rechecked; 63 processed, 27 unscheduled supplementary records')
    packet_rows=[];wls_rows=[];missing=[];ncov=0.;nphase=0.;bcache=0.;nb=0.;nn=0;packet_count=0
    for r in runs:
        p=__import__('pathlib').Path(r['path'])
        check('metadata:'+str(r['index']),sha(p.parent/'run.json')==r['metadata_hash'] and sha(p.parent/'events.log')==r['events_hash'],'Run metadata and switching log unchanged')
        check('run_config:'+str(r['index']),r['config_sha256']==cfgsha,'Frozen configuration')
        check('all_states:'+str(r['index']),set(s['state'] for s in r['states'])==set(MAP),'G1-G4,C1,C2 all accounted for')
        for s in r['states']:
            base=dict(index=r['index'],group=r['group'],role=r['role'],state=s['state'])
            if s['status']!='computed':missing.append(base|s);continue
            v=np.load(R/'packets'/f'{r["index"]:03d}_{s["state"]}.npz');n=s['packets'];packet_count+=n;nn+=1
            assert all(len(v[k])==n for k in ['packet_id','host_ns','nic_timestamp','window_id','B_tau_ns','B_psi_rad','C_tau_ns','C_psi_rad','C_applied'])
            assert np.all(np.diff(v['host_ns'])>0) and np.all(np.diff(v['packet_id'])>0)
            assert v['packet_id'].min()>=0 and v['packet_id'].max()<r['frames']
            assert all(np.isfinite(v[k]).all() for k in ['B_tau_ns','B_psi_rad','C_tau_ns','C_psi_rad','C_correlation','cov_A','cov_B','cov_C','spectrum_A','spectrum_B','spectrum_C','prediction'])
            accepted=v['C_applied'];low=v['C_correlation']<CFG['correlation_accept'];edge=v['C_edge'];expected=(~low)&(~edge)
            if not s['common_model_pass']:expected[:]=False
            assert np.array_equal(accepted,expected)
            # correct() repeats unit normalisation; reject fallback is unchanged to roundoff.
            if not s['common_model_pass']:
                assert np.linalg.norm(v['cov_A']-v['cov_C'])/np.linalg.norm(v['cov_A'])<CFG['numerical_cov_tolerance']
                assert np.max(abs(v['spectrum_A']-v['spectrum_C']))<CFG['numerical_cov_tolerance']
            covdiff=np.linalg.norm(v['cov_B']-v['cov_A'])/np.linalg.norm(v['cov_A']);ncov=max(ncov,covdiff)
            nb=max(nb,float(np.max(abs(v['spectrum_B']-v['spectrum_A']))))
            nphase=max(nphase,max(m['pair_phase_change_deg'] for m in s['methods']));bcache=max(bcache,s['baseline_cached_cov_relative'])
            reasons=[]
            if s['chain_delay_p90_ns']>CFG['common_gate_chain_delay_p90_ns']:reasons.append('chain_delay')
            if s['chain_phase_p90_deg']>CFG['common_gate_chain_phase_p90_deg']:reasons.append('chain_phase')
            if s['prediction_corr_prefix']<CFG['common_gate_prediction_corr_median']:reasons.append('prefix_prediction')
            packet_rows.append(base|dict(packets=n,accepted=int(accepted.sum()),rejected=int((~accepted).sum()),state_gate_pass=s['common_model_pass'],gate_reasons=';'.join(reasons),low_correlation_candidates=int(low.sum()),edge_candidates=int(edge.sum()),large_delay_steps=int((abs((np.diff(v['C_tau_ns'])+400)%800-400)>50).sum())))
            w=v['wls_comparison'];bad=~np.isfinite(w[:,1:4]).all(axis=1)
            disagreement=bad|(abs(w[:,1])>CFG['common_gate_chain_delay_p90_ns'])|(abs(w[:,2])>CFG['common_gate_chain_phase_p90_deg'])|(w[:,3]>CFG['common_gate_chain_phase_p90_deg'])
            for row,b,flag in zip(w,bad,disagreement):
                wls_rows.append(base|dict(packet_offset=int(row[0]),delay_diff_ns=row[1],phase_diff_deg=row[2],residual_deg=row[3],nonfinite=bool(b),diagnostic_disagreement=bool(flag)))
    check('375_archives_complete',nn==len(list((R/'packets').glob('*.npz')))==375,'Lengths, finite values, timestamps, masks and rejected-state fallback checked')
    check('missing_states_retained',len(missing)==3 and all(s['group']=='A' and s['state']=='G4' for s in missing),missing)
    check('CPE_covariance_invariant',ncov<CFG['numerical_cov_tolerance'],ncov)
    check('pair_phase_invariant',nphase<CFG['pair_phase_tolerance_deg'],nphase)
    check('baseline_matches_previous',bcache<2e-6,bcache)
    for f in ['synthetic.json','synthetic_extended.json','synthetic_fades.json']:
        d=json.loads((R/'results'/f).read_text());check('synthetic:'+f,d['all_pass'],'Includes expected negative controls; all_pass is not universal scientific success')
    pd.DataFrame(packet_rows).to_csv(R/'results/packet_acceptance.csv',index=False)
    pd.DataFrame(wls_rows).to_csv(R/'results/wls_diagnostics.csv',index=False)
    pd.DataFrame(missing).to_csv(R/'results/missing_states.csv',index=False)
    # Explicit peak stability across runs; no truth labels or gauge fitting.
    pairs=pd.read_csv(R/'results/between_run_comparisons.csv')
    methods=pd.read_csv(R/'results/method_summary.csv').set_index(['index','state','method'])
    peaks=[]
    for row in pairs.itertuples(index=False):
        p1=methods.loc[(row.index1,row.state,row.method)];p2=methods.loc[(row.index2,row.state,row.method)]
        peaks.append(dict(group=row.group,state=row.state,index1=row.index1,index2=row.index2,same_batch=row.same_batch,method=row.method,angle_abs_difference_deg=abs(p1.angle_deg-p2.angle_deg),delay_circular_abs_difference_ns=abs((p1.delay_ns-p2.delay_ns+400)%800-400)))
    pd.DataFrame(peaks).to_csv(R/'results/between_run_peak_differences.csv',index=False)
    check('actual_output_coverage',len(list((R/'results').glob('actual_*.json')))==63 and len(pd.read_csv(R/'results/actual_output_stability.csv'))==1125,'375 states x 3 methods')
    check('figures_complete',len(list((R/'figures').glob('*.png')))==len(list((R/'figures').glob('*.svg')))==7,'PNG and vector SVG; source table')
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),config_sha256=cfgsha,all_pass=all(c['passed'] for c in checks),checks=checks,processed_runs=len(runs),computed_states=nn,packets=packet_count,max_CPE_cov_difference=ncov,max_CPE_spectrum_difference=nb,max_pair_phase_difference_deg=nphase,max_baseline_cache_difference=bcache,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__)
    (R/'results/completion_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2));assert result['all_pass']
if __name__=='__main__':main()
