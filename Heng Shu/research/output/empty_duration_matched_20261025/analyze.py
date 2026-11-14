"""Duration-matched E1/E2 and E3/E4 comparison. No new fitting or truth use."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;PRE=R.parent/'empty_reference_static_recovery_20261024';OLD=R.parent/'ratnam_phase_timing_validation'
sys.path.insert(0,str(PRE))
from common import rat,dec,invariants,distance,sha

def main():
    for d in ['results','figures','cache']:(R/d).mkdir(exist_ok=True)
    cfg=dict(pairings={'10s':['E1','E2'],'15s':['E3','E4']},rounds='All 3x3 pairs; equal ordinal run pairs additionally labeled, not simultaneous.',states=list(rat.MAP),comparisons=['full','equal_packet_count_max512'],session_relation='Duration is not evidence of same session. E1/E3 same-session confirmed previously; E1/E2 and E3/E4 unconfirmed.',new_phase_fitting=False)
    (R/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    mm=pd.read_csv(PRE/'empty_manifest.csv');meta=pd.read_csv(PRE/'results/source_audit.csv').set_index('index');sources=[];states=[];data={};out=[]
    previous=pd.read_csv(OLD/'results/state_summary.csv').set_index(['index','state'])
    for row in mm.itertuples(index=False):
        p=Path(row.path);m=dec.read(p.parent/'run.json');assert sha(p)==row.sha256
        duration=m['requested_seconds'];assert duration==(10 if row.capture_id in ['E1','E2'] else 15)
        old_cache=(PRE/'cache'/f'{row.index:03d}_G1.npz').exists()
        if not old_cache:x,host,nic=dec.decode(p)
        for state in rat.MAP:
            op=OLD/'packets'/f'{row.index:03d}_{state}.npz';v=np.load(op);ids=v['packet_id']
            if old_cache:
                w=np.load(PRE/'cache'/f'{row.index:03d}_{state}.npz');z=w['x'];assert np.array_equal(ids,w['packet_id'])
            else:
                assert ids.min()>=0 and ids.max()<len(x)
                rel=(host[ids]-m['start_ns'])/1e9;assert rel.min()>=0 and rel.max()<=duration
                z=rat.unit(x[ids].astype(complex))
            inv=invariants(z);assert distance(inv['covariance'],v['cov_A'])<1e-10
            data[(row.index,state)]=dict(x=z,inv=inv,A=v['spectrum_A'],C=v['spectrum_C'],cov_C=v['cov_C'])
            st=previous.loc[(row.index,state)]
            states.append(dict(index=row.index,capture=row.capture_id,run=row.run,seconds=duration,state=state,packets=len(z),old_common_model_pass=bool(st.common_model_pass),old_C_accept_fraction=st.C_accept_fraction))
            sources.append(dict(index=row.index,capture=row.capture_id,run=row.run,state=state,seconds=duration,path=str(p),raw_sha256=row.sha256,cached_result=str(op),cached_result_sha256=sha(op)))
        print('Loaded',row.capture_id,row.run,duration,flush=True)
    mm.to_csv(R/'input_manifest.csv',index=False)
    def pair(a,b,kind):
        for state in rat.MAP:
            u=data[(a['index'],state)];v=data[(b['index'],state)]
            for variant in cfg['comparisons']:
                if variant=='full':aa,bb=u['inv'],v['inv'];sa,sb=u['A'],v['A'];na=len(u['x']);nb=len(v['x'])
                else:
                    cap=min(512,len(u['x']),len(v['x']));aa,bb=[invariants(t['x'][np.linspace(0,len(t['x'])-1,cap,dtype=int)]) for t in [u,v]]
                    steer=rat.steering(state);sa=rat.music(aa['covariance'],steer);sb=rat.music(bb['covariance'],steer);na=nb=cap
                mask=(aa['pair_coherence']>=.5)&(bb['pair_coherence']>=.5)
                ph=abs(rat.wrap_phase(aa['pair_angle']-bb['pair_angle']))*180/np.pi
                out.append(dict(index_a=a['index'],index_b=b['index'],capture_a=a['capture_id'],capture_b=b['capture_id'],run_a=a['run'],run_b=b['run'],duration_s=int(meta.loc[a['index'],'requested_seconds']),pairing=kind,same_ordinal_run=a['run']==b['run'],state=state,variant=variant,packets_a=na,packets_b=nb,spatial_distance=distance(aa['spatial'],bb['spatial']),amplitude_distance=distance(aa['amplitude'],bb['amplitude']),covariance_distance=distance(aa['covariance'],bb['covariance']),spectrum_rms=float(np.sqrt(np.mean((sa-sb)**2))),pair_phase_median_deg=float(np.median(ph[mask])) if mask.any() else None,phase_support_fraction=float(mask.mean()),C_cov_distance=distance(u['cov_C'],v['cov_C']) if variant=='full' else None,C_spectrum_rms=float(np.sqrt(np.mean((u['C']-v['C'])**2))) if variant=='full' else None))
    for cap,z in mm.groupby('capture_id'):
        for a,b in itertools.combinations(z.to_dict('records'),2):pair(a,b,'within_'+cap)
    for ca,cb in [('E1','E2'),('E3','E4')]:
        for a in mm[mm.capture_id==ca].to_dict('records'):
            for b in mm[mm.capture_id==cb].to_dict('records'):pair(a,b,ca+'_'+cb)
    df=pd.DataFrame(out);df.to_csv(R/'results/pair_comparisons.csv',index=False);pd.DataFrame(states).to_csv(R/'results/state_quality.csv',index=False);pd.DataFrame(sources).to_csv(R/'results/source_hashes.csv',index=False)
    metrics=['spatial_distance','amplitude_distance','covariance_distance','spectrum_rms','pair_phase_median_deg']
    rows=[]
    for (pairing,var),z in df.groupby(['pairing','variant']):
        for scope,zz in [('all_run_pairs',z),('same_ordinal_runs',z[z.same_ordinal_run])]:
            if len(zz)==0:continue
            rows.append(dict(pairing=pairing,variant=var,scope=scope,state_pairs=len(zz),**{k:float(zz[k].median()) for k in metrics}))
    summ=pd.DataFrame(rows);summ.to_csv(R/'results/summary.csv',index=False)
    df.groupby(['pairing','variant','state'])[metrics].median().reset_index().to_csv(R/'results/per_state_summary.csv',index=False)
    for cap,z in mm.groupby('capture_id'):
        np.savez_compressed(R/'cache'/f'{cap}_spectra.npz',indices=z['index'].to_numpy(),states=list(rat.MAP),A=np.stack([[data[(i,s)]['A'] for s in rat.MAP] for i in z['index']]),C=np.stack([[data[(i,s)]['C'] for s in rat.MAP] for i in z['index']]))
    assert len(df)==360 and len(states)==72 and len(sources)==72
    print(summ[summ.variant=='full'].to_string(index=False))
    (R/'results/checks.json').write_text(json.dumps(dict(all_raw_hashes_match=True,all_72_baseline_covariances_match=True,state_records=72,pair_records=len(df),cross_capture_pairs_per_duration=54,same_ordinal_state_pairs_per_duration=18,new_phase_parameters_fitted=False),indent=2),encoding='utf-8')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
