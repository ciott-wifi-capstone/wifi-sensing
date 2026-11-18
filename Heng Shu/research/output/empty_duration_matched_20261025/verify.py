from analyze import *
def main():
    pairs=pd.read_csv(R/'results/pair_comparisons.csv');ph=pd.read_csv(R/'results/chain_pair_phase.csv');src=pd.read_csv(R/'results/source_hashes.csv');deps=pd.read_csv(R/'dependency_hashes.csv');checks={}
    checks['360_unique_comparisons']=len(pairs)==360 and len(pairs.drop_duplicates(['index_a','index_b','state','variant']))==360
    for group,sec in [('E1_E2',10),('E3_E4',15)]:
        z=pairs[(pairs.pairing==group)&(pairs.variant=='full')]
        checks[group+'_54_full_18_ordinal']=len(z)==54 and int(z.same_ordinal_run.sum())==18 and (z.duration_s==sec).all()
    eq=pairs[pairs.variant=='equal_packet_count_max512'];checks['equal_packet_counts']=(eq.packets_a==eq.packets_b).all() and (eq.packets_a<=512).all()
    checks['all_chain_diagnostics_retained']=len(ph)==324 and len(ph.drop_duplicates(['pairing','index_a','index_b','state','chain_pair']))==324 and (ph.supported_tones>0).all()
    checks['raw_hashes_unchanged']=all(sha(x.path)==x.raw_sha256 for x in src.drop_duplicates('path').itertuples(index=False))
    checks['old_result_hashes_unchanged']=all(sha(x.cached_result)==x.cached_result_sha256 for x in src.itertuples(index=False))
    checks['dependencies_unchanged']=all(sha(x.path)==x.sha256 for x in deps.itertuples(index=False))
    checks['no_missing_metrics']=bool(np.isfinite(pairs[['spatial_distance','amplitude_distance','covariance_distance','spectrum_rms','pair_phase_median_deg']]).all().all())
    checks['all_72_states_present']=len(pd.read_csv(R/'results/state_quality.csv'))==72
    assert all(checks.values()),checks
    (R/'results/verification.json').write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2),encoding='utf-8')
    print(json.dumps({k:bool(v) for k,v in checks.items()},indent=2))
if __name__=='__main__':main()
