"""Bounded pre-registered real-data A/B/C comparison; resumable per-run outputs."""
import argparse,hashlib,importlib.util,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core import *
spec=importlib.util.spec_from_file_location('original_decoder',R/'provenance/analyze.py')
decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=True),encoding='utf-8')
def scalar(v):return float(v)

def analyse_state(x,state,host,nic,packet_ids,window_ids,key):
    start=time.perf_counter();x=unit(np.asarray(x,dtype=np.complex128));n=len(x)
    gate=common_gate(x);b,pb=correct(x,'B');c,pc=correct(x,'C',gate['common_model_pass'])
    a=steering(state);ys={'A':x,'B':b,'C':c};covs={k:covariance(v) for k,v in ys.items()}
    spectra={k:music(v,a) for k,v in covs.items()};results=[];windows=[]
    # All records remain in primary diagnostic covariances, including rejected C packets.
    # A common mask could make a conditional comparison artificially favourable, so no deletion.
    for method,z in ys.items():
        angle,delay=peak(spectra[method]);rr=covs[method]
        results.append(dict(method=method,angle_deg=angle,delay_ns=delay,
            covariance_change=float(np.linalg.norm(rr-covs['A'])/np.linalg.norm(covs['A'])),
            pair_phase_change_deg=pair_error(x,z),spectrum_max_change=float(np.max(abs(spectra[method]-spectra['A'])))))
        # Four contiguous chronological blocks, never randomly split packets.
        for q,ix in enumerate(np.array_split(np.arange(n),4)):
            if len(ix)<30:continue
            rc=covariance(z[ix]);sc=music(rc,a);ang,dt=peak(sc)
            windows.append(dict(method=method,block=q,n=len(ix),start_host_ns=int(host[ix[0]]),end_host_ns=int(host[ix[-1]]),
               angle_deg=ang,delay_ns=dt,covariance_to_whole=float(np.linalg.norm(rc-rr)/np.linalg.norm(rr)),
               spectrum_to_whole=float(np.sqrt(np.mean((sc-spectra[method])**2)))))
    assert results[1]['covariance_change']<CFG['numerical_cov_tolerance']
    assert max(v['pair_phase_change_deg'] for v in results)<CFG['pair_phase_tolerance_deg']
    # Fixed early reference; test future samples on disjoint chain/tone entries.
    # These checks diagnose conditional common-model prediction, not physical timing truth.
    prefix=CFG['prefix_packets'];ref=c[:prefix].mean(0) if gate['common_model_pass'] else x[0]
    train,test=crossfit_masks();pred=[];wls=[]
    for p in range(prefix,n,8):
        t,ph,_,edge=fit(x[p],ref,train);tt,pt,_,_=fit(x[p],ref,test)
        bphase=float(np.angle(np.sum(x[p].conj()*ref*train)))
        scorea=score(x[p],ref,test);scoreb=score(apply(x[p],0.,bphase),ref,test);scorec=score(apply(x[p],t,ph),ref,test)
        pred.append([p,t,tt,float(wrap_delay(tt-t)),scorea[0],scorea[1],scorea[2],scoreb[0],scoreb[1],scoreb[2],scorec[0],scorec[1],scorec[2],edge])
        # Approximation benchmark on same input/reference; no parameter selection from labels.
        t0=time.perf_counter();tm,pm,_,_=fit(x[p],ref);sec_ml=time.perf_counter()-t0
        t0=time.perf_counter();tw,pw,res=fit_wls(x[p],ref,tm);sec_wls=time.perf_counter()-t0
        wls.append([p,float(wrap_delay(tw-tm)),float(wrap_phase(pw-pm))*180/np.pi,res*180/np.pi,sec_ml,sec_wls])
    pred=np.array(pred);wls=np.array(wls)
    # Measurements at two actual packet time scales; no interpolation of switched gaps.
    dh=np.diff(host)/1e6;dn=np.diff(nic);same=np.diff(window_ids)==0
    summary=dict(state=state,packets=n,windows=int(len(np.unique(window_ids))),**gate,
        C_accept_fraction=float(pc['accepted'].mean()),C_edge_fraction=float(pc['edge'].mean()),
        C_tau_iqr_ns=iqr(pc['tau_ns']),C_psi_circular_sd_deg=circ_sd(pc['psi_rad'])*180/np.pi,
        C_tau_step_p95_ns=float(np.quantile(abs(wrap_delay(np.diff(pc['tau_ns']))),.95)),
        C_large_step_fraction=float(np.mean(abs(wrap_delay(np.diff(pc['tau_ns'])))>50)),
        B_psi_circular_sd_deg=circ_sd(pb['psi_rad'])*180/np.pi,
        B_seconds=pb['seconds'],C_seconds=pc['seconds'],
        intra_window_interval_median_ms=float(np.median(dh[same])) if same.any() else None,
        inter_window_gap_median_ms=float(np.median(dh[~same])) if (~same).any() else None,
        interval_p99_ms=float(np.quantile(dh,.99)),host_nonmonotonic=int((dh<=0).sum()),nic_nonmonotonic=int((dn<=0).sum()),
        pred_packets=len(pred),pred_delay_iqr_A_ns=iqr(pred[:,2]),pred_delay_iqr_C_ns=iqr(pred[:,3]),
        pred_corr_A=float(np.median(pred[:,5])),pred_corr_C=float(np.median(pred[:,11])),
        pred_real_corr_A=float(np.median(pred[:,4])),pred_real_corr_B=float(np.median(pred[:,7])),pred_real_corr_C=float(np.median(pred[:,10])),
        pred_phase_sd_A_deg=circ_sd(pred[:,6])*180/np.pi,pred_phase_sd_B_deg=circ_sd(pred[:,9])*180/np.pi,pred_phase_sd_C_deg=circ_sd(pred[:,12])*180/np.pi,
        wls_delay_diff_p95_ns=float(np.nanquantile(abs(wls[:,1]),.95)),wls_phase_diff_p95_deg=float(np.nanquantile(abs(wls[:,2]),.95)),
        wls_residual_p95_deg=float(np.nanquantile(wls[:,3],.95)),wls_failed_fraction=float(np.mean(~np.isfinite(wls[:,1]))),
        wls_seconds_median=float(np.median(wls[:,5])),ml_seconds_median=float(np.median(wls[:,4])))
    summary['pred_delay_ratio']=summary['pred_delay_iqr_C_ns']/max(summary['pred_delay_iqr_A_ns'],1e-9)
    summary['pred_phase_ratio_C_B']=summary['pred_phase_sd_C_deg']/max(summary['pred_phase_sd_B_deg'],1e-9)
    summary['seconds_total']=time.perf_counter()-start
    # Correction parameters sufficient to reconstruct each output from immutable raw input.
    np.savez_compressed(R/'packets'/f'{key}.npz',packet_id=packet_ids,host_ns=host,nic_timestamp=nic,window_id=window_ids,
        B_tau_ns=pb['tau_ns'],B_psi_rad=pb['psi_rad'],B_correlation=pb['correlation'],
        C_tau_ns=pc['tau_ns'],C_psi_rad=pc['psi_rad'],C_correlation=pc['correlation'],C_edge=pc['edge'],C_applied=pc['accepted'],
        prefix_warmup=np.arange(n)<prefix,prediction=pred,wls_comparison=wls,
        reference_prefix=ref,first_packet_reference=x[0],**{f'cov_{k}':v for k,v in covs.items()},**{f'spectrum_{k}':v for k,v in spectra.items()})
    return summary,results,windows

def run(role):
    assert sha(R/'config.json')==(R/'config.sha256').read_text().strip()
    assert json.loads((R/'results/synthetic.json').read_text())['all_pass']
    if role=='held_out':
        assert (R/'results/development_review.json').exists(),'Review development before opening holdout'
        assert json.loads((R/'results/development_review.json').read_text())['config_sha256']==sha(R/'config.json')
    if role in ['expansion','supplementary']:
        assert json.loads((R/'results/expansion_decision.json').read_text())['expand_primary']
        data=pd.read_csv(R/'extension_manifest.csv')
    else:data=pd.read_csv(R/'bounded_manifest.csv')
    data=data[data.role==role].sort_values(['analysis_group','batch','run'])
    saved=np.load(R.parent/'static_analysis_20261018/geometry_update_20261019/covariances.npz')
    for _,row in data.iterrows():
        idx=int(row['index']);dest=R/'results'/f'run_{idx:03d}.json'
        if dest.exists():
            prior=json.loads(dest.read_text());assert prior['config_sha256']==sha(R/'config.json')
            print('Already complete',idx,flush=True);continue
        start=time.perf_counter();p=Path(row.path);assert sha(p)==row.sha256
        meta=decoder.read(p.parent/'run.json');x,host,nic=decoder.decode(p);ts=(host-meta['start_ns'])/1e9
        assert np.all(np.diff(host)>0),'Nonmonotonic timestamps require investigation'
        align=decoder.align(p.parent/'events.log',meta['start_ns']/1e9);ix={s:[] for s in MAP};win={s:[] for s in MAP}
        entry=dict(index=idx,group=row.analysis_group,batch=row.batch,run=row.run,role=role,path=str(p),sha256=row.sha256,
                   config_sha256=sha(R/'config.json'),frames=len(x),complete_window=bool(row.complete_window),trailing_s=float(row.trailing_s),
                   input_failure=str(row.failure),metadata_hash=sha(p.parent/'run.json'),events_hash=sha(p.parent/'events.log'),states=[])
        if align:
            starts,states,guard=align
            entry.update(guard_ms=guard*1e3,state_period_ms=float(np.median(np.diff(starts))*1e3),cycle_period_ms=float(np.median(starts[6:]-starts[:-6])*1e3))
            for ei,(st,si) in enumerate(zip(starts[:-1],states[:-1])):
                if st+.025>meta['requested_seconds'] or .025-2*guard<.005:continue
                l,r=np.searchsorted(ts,[st+guard,st+.025-guard]);s=decoder.STATES[si]
                ix[s].extend(range(l,r));win[s].extend([ei]*(r-l))
        for state in MAP:
            ids=np.array(ix[state],dtype=int);key=f'{idx:03d}_{state}'
            if len(ids)<CFG['min_packets']:
                entry['states'].append(dict(state=state,packets=len(ids),status='insufficient',reason='fewer_than_128_guarded_packets'));continue
            summary,methods,windows=analyse_state(x[ids],state,host[ids],nic[ids],ids,np.array(win[state]),key)
            old=saved[key];new=np.load(R/'packets'/f'{key}.npz')['cov_A']
            discrepancy=float(np.linalg.norm(old-new)/np.linalg.norm(old));assert discrepancy<2e-6
            summary.update(status='computed',baseline_cached_cov_relative=discrepancy,methods=methods,blocks=windows)
            entry['states'].append(summary)
            print(idx,row.analysis_group,row.run,state,'gate',summary['common_model_pass'],'accept',round(summary['C_accept_fraction'],3),'ratio',round(summary['pred_delay_ratio'],3),flush=True)
        entry['seconds_total']=time.perf_counter()-start;dump(dest,entry)
    print('ROLE COMPLETE',role,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['development','held_out','expansion','supplementary']);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.role)
