"""Paired empty / A / empty pilot. Predictions are frozen before geometry scoring."""
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "kong1_kong1a_20261105"
sys.path.insert(0, str(PREVIOUS))
import analyze_captures as base
# geometry_update imports its sibling 'analyze'; run this file as a script, not as that module.
sys.path.insert(0, str(ROOT.parent / "same_frequency_spatial_background_20261104"))
import spatial_model as background

RAW = Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop'))) / "New A"
LABEL = {1: "empty_pre", 2: "A_occupied", 3: "empty_post"}
PERIODS = [(1, 2), (2, 3), (1, 3)]
ANGLES = base.ANGLE
assert np.array_equal(ANGLES, background.ANG) and np.array_equal(base.F, background.F)


def write(name, rows, columns=None):
    data = pd.DataFrame(rows, columns=columns)
    data.to_csv(ROOT/name, index=False, encoding="utf-8-sig")
    return data


def time_blocks(x, times, start=0., end=10.):
    edges = np.linspace(start, end, 5)
    blocks = [x[(times>=left)&(times<right)] for left,right in zip(edges[:-1],edges[1:])]
    assert all(len(b)>0 for b in blocks)
    return blocks


def read_all():
    inventory, operations = base.identity_audit(RAW)
    assert all(item["ok"] for item in inventory)
    write("manifest_checks.csv", inventory)
    write("receiver_events.csv", operations)
    old_digests = set(pd.read_csv(PREVIOUS/"runs.csv").raw_sha256)
    old_digests.update(pd.read_csv(base.VALIDATION/"input_manifest.csv").sha256)
    samples, deeper = {}, {}
    results, quality, closure = [], [], []
    for run in LABEL:
        folder = RAW/"capture"/f"run{run}"
        meta = base.read(folder/"run.json")
        qc = base.read(folder/"metadata_qc.json")
        path = folder/meta["raw_path"]
        x, host, nic = base.decoder.decode(path)
        x = x.astype(np.complex128)
        addresses, orders = base.frame_details(path)
        assert len(x)==len(addresses)==meta["frames"]==qc["complete_frames"]
        assert np.isfinite(x).all() and np.all(np.diff(host)>0) and np.all(np.diff(nic)>0)
        times = (host-meta["start_ns"])/1e9
        duration = (meta["end_ns"]-meta["start_ns"])/1e9
        formal = (times>=0)&(times<duration)
        assert formal.sum()==meta["window"]["records"]
        target = addresses==base.APPARENT_TA
        event_text = (folder/"events.log").read_text(encoding="utf8")
        events = base.EVENT.findall(event_text)
        wrong = sum(list(map(int,row[2].split(",")))!=base.MAP[row[1]] or int(row[4])!=base.STATES.index(row[1]) for row in events)
        assert wrong==0 and len(events)==meta["events"]["ready_events"]
        starts, order, guard = base.decoder.align(folder/"events.log",meta["start_ns"]/1e9)
        interior, eligible = base.select_visits(times,starts,order,duration,guard,target)
        strict, _ = base.select_visits(times,starts,order,duration,.005,target)
        for state in base.STATES:
            ids = np.concatenate([idx for _,idx in interior[state]])
            safe = np.concatenate([idx for _,idx in strict[state]])
            assert len(np.unique(ids))==len(ids) and np.all(formal[ids])
            samples[run,state] = x[ids],times[ids]
            deeper[run,state] = base.spatial(x[safe])
            eig=np.linalg.eigvalsh(base.spatial(x[ids]))
            quality.append(dict(run=run,state=state,guarded_packets=len(ids),guard_5ms_packets=len(safe),
                                complete_visits=eligible[state],sampled_visits=len(interior[state]),
                                coverage=len(interior[state])/eligible[state],first_half_packets=np.sum(times[ids]<5),
                                second_half_packets=np.sum(times[ids]>=5),dominant_eigen_fraction=np.median(eig[:,-1]/eig.sum(axis=1))))
        shared = {}
        for s in ["C1","C2"]:
            shared[s]={e//6:base.relative(base.spatial(x[ids]))[2] for e,ids in interior[s]}
        cycles=sorted(set(shared["C1"])&set(shared["C2"]))
        q1=np.array([shared["C1"][n] for n in cycles])
        q2=np.array([shared["C2"][n] for n in cycles])
        support=(abs(q1)>=.5)&(abs(q2)>=.5)
        error=np.degrees(abs(np.angle(q1*q2.conj())))[support]
        closure.append(dict(run=run,matched_cycles=len(cycles),supported_fraction=support.mean(),
                            median_relative_phase_deg=np.median(error),p90_relative_phase_deg=np.quantile(error,.9)))
        results.append(dict(run=run,label=LABEL[run],source=str(path),sha256=base.sha(path),matches_old=base.sha(path) in old_digests,
                            boot_id=meta["before"]["boot_id"],phy=meta["before"]["phy"].split("/")[-1],
                            ifindex=meta["before"]["ifindex"],before_after_identical=meta["before"]==meta["after"],
                            continuity_gate=meta["continuity_gate"],frames=len(x),formal_frames=int(formal.sum()),
                            target_frames=int((formal&target).sum()),pps=(formal&target).sum()/duration,
                            start_ns=meta["start_ns"],end_ns=meta["end_ns"],duration_s=duration,
                            max_gap_ms=np.diff(host[formal]).max()/1e6,guard_ms=guard*1e3,ready_count=len(events),
                            mapping_errors=wrong,header_mismatches=sum(qc["mismatches"].values()),
                            antenna_perm_counts=json.dumps(dict(orders)),rx_initialization=meta["radio_initialization"]))
    runs = write("raw_runs.csv",results)
    assert runs.sha256.nunique()==3 and runs.phy.nunique()==runs.ifindex.nunique()==runs.boot_id.nunique()==1
    write("state_quality.csv",quality)
    write("shared_pair_check.csv",closure)
    return samples,deeper


def changes(samples,deeper):
    metrics = []
    quarters = []
    for state in base.STATES:
        cov = {r:base.spatial(samples[r,state][0]) for r in LABEL}
        blocks = {r:[base.spatial(b) for b in time_blocks(*samples[r,state])] for r in LABEL}
        for r,s in PERIODS:
            for name,covs in [("primary",cov),("guard5ms",{i:deeper[i,state] for i in LABEL})]:
                metrics.append(dict(state=state,pair=f"{r}{s}",guard=name,**base.compare(covs[r],covs[s])))
            for t in range(4):
                quarters.append(dict(state=state,pair=f"{r}{s}",quarter=t+1,**base.compare(blocks[r][t],blocks[s][t])))
    write("paired_changes.csv",metrics)
    write("paired_quarters.csv",quarters)


def fit_r1_reference(samples):
    fits, checks, empirical = [], [], []
    templates = {}
    held = np.arange(len(base.F))%3==2
    for state in base.STATES:
        data,times = samples[1,state]
        qfit = base.relative(base.spatial(data[times<5]))
        qdev = base.relative(base.spatial(data[times>=5]))
        qpost = base.relative(base.spatial(samples[3,state][0]))
        los=base.los_response(state)
        corr=[np.ones(len(base.F),complex)]
        for j in [1,2]:
            options={}
            for model,linear in [("constant",False),("linear",True)]:
                params = base.fit_offset(qfit[j-1],los[j],linear)
                if params is None:
                    continue
                h, intercept, slope = params
                check=base.residual(qdev[j-1],h*los[j],held)
                fits.append(dict(state=state,chain=j+1,model=model,intercept_deg=intercept,slope_ns=slope,**check))
                if np.isfinite(check["p90_phase_deg"]):
                    options[model]=(h,check["p90_phase_deg"])
            if options:
                chosen="constant" if "constant" in options and options["constant"][1]<=15 else min(options,key=lambda n:options[n][1])
                response=options[chosen][0]
                check=base.residual(qpost[j-1],response*los[j],np.ones(len(held),bool))
            else:
                chosen="unavailable"
                response=np.full(len(held),np.nan+1j*np.nan)
                check=dict(support=0,median_phase_deg=np.nan,p90_phase_deg=np.nan)
            corr.append(response)
            checks.append(dict(state=state,chain=j+1,chosen=chosen,**check,
                               pass_pair=check["support"]>=.8 and check["p90_phase_deg"]<=15))
            mask=(abs(qfit[j-1])>=.5)&(abs(qpost[j-1])>=.5)
            diff=np.degrees(abs(np.angle(qpost[j-1]*qfit[j-1].conj())))[mask]
            p90=np.quantile(diff,.9) if len(diff) else np.nan
            empirical.append(dict(state=state,chain=j+1,support=mask.mean(),p90_deg=p90,
                                  pass_pair=mask.mean()>=.8 and p90<=15))
        templates[state]=np.array(corr)
    write("ref_development.csv",fits)
    write("ref_run3_validation.csv",checks)
    write("empirical_repeatability.csv",empirical)
    return templates


def raw_spectra(samples, templates):
    angular, joint, maxima, locals_, joint_max = {}, {}, [], [], []
    for state in base.STATES:
        vectors=base.steering(state)
        ja=base.joint.steering(base.Y[np.array(base.MAP[state])-1])
        for r in LABEL:
            xs,times=samples[r,state]
            for mode in ("raw","r1_template"):
                xx=xs if mode=="raw" else xs*templates[state].conj()[None,:,:]
                if not np.isfinite(xx).all():
                    continue
                spec=base.angular_spectra(base.spatial(xx),vectors)
                parts=[base.angular_spectra(base.spatial(b),vectors)[0] for b in time_blocks(xx,times)]
                rows=base.peak_summary(spec,parts)
                locs=[dict(state=state,run=r,mode=mode,rank=i,**p) for i,p in enumerate(rows,1)]
                locals_.extend(locs)
                maxima.append(locs[0])
                angular[f"{state}_R{r}_{mode}"]=spec
            result=base.joint.music(base.joint.covariance(xs),ja,q=3)[:,:-1]
            ai,ti=np.unravel_index(result.argmax(),result.shape)
            joint[f"{state}_R{r}"]=result
            joint_max.append(dict(state=state,run=r,angle_parameter_deg=ANGLES[ai],uncalibrated_delay_ns=base.joint.TAU[ti]))
    for r in LABEL:
        for mode in ("raw","r1_template"):
            keys=[f"{s}_R{r}_{mode}" for s in base.GROUPS]
            if not all(k in angular for k in keys):
                continue
            fused=np.mean([angular[k][0] for k in keys],axis=0)
            angular[f"fusion_R{r}_{mode}"]=fused/fused.max()
            maxima.append(dict(state="G fusion",run=r,mode=mode,peak_deg=ANGLES[fused.argmax()]))
    write("blind_angular_maxima.csv",maxima)
    write("blind_local_peaks.csv",locals_)
    write("blind_joint_maxima.csv",joint_max)
    np.savez_compressed(ROOT/"angular_spectra.npz",angles=ANGLES,**angular)
    np.savez_compressed(ROOT/"joint_spectra.npz",angles=ANGLES,tau_ns=base.joint.TAU[:-1],**joint)
    return angular, pd.DataFrame(locals_)


def beam_fraction(cov,steering,indices):
    a=steering[indices]
    r=cov[indices]
    power=np.real(np.einsum("ki,kij,kj->k",a.conj(),r,a))
    return float(power.sum()/(3*np.trace(r,axis1=1,axis2=2).real.sum()))


def fixed_branch(samples,angular,peaks):
    ix=np.flatnonzero(ANGLES==-21)[0]
    vec=base.steering("C1")[:,:,ix]
    rows=[]
    for r in LABEL:
        x,t=samples[r,"C1"]
        peaks_window=peaks[(peaks.run==r)&(peaks.state=="C1")&(peaks["mode"]=="raw")&
                           ((peaks.peak_deg+21).abs()<=9)]
        chosen=peaks_window.sort_values("rank").iloc[0] if len(peaks_window) else None
        for b,xx in enumerate([x,*time_blocks(x,t)]):
            cov=base.spatial(xx)
            rows.append(dict(run=r,quarter=b,**{f"fraction_{label}":beam_fraction(cov,vec,idx)
                             for label,idx in [("all",slice(None)),("low",slice(0,15)),("high",slice(15,None))]},
                             fixed_music_relative_db=10*np.log10(base.angular_spectra(cov,base.steering("C1"))[0,ix]),
                             nearby_full_peak_deg=chosen.peak_deg if chosen is not None else np.nan,
                             nearby_full_peak_rank=chosen["rank"] if chosen is not None else np.nan,
                             nearby_full_peak_stable=bool(chosen.repeat_within_window) if chosen is not None else False))
    df=write("C1_minus21_before_during_after.csv",rows)
    index=df.set_index(["quarter","run"])
    comparison=[]
    for quarter in range(5):
        for band in ("all","low","high"):
            db=10*np.log10(index.loc[quarter,f"fraction_{band}"].reindex([1,2,3]).to_numpy())
            above=db[1]-max(db[0],db[2]); drift=abs(db[0]-db[2])
            comparison.append(dict(quarter=quarter,band=band,run2_minus_run1_db=db[1]-db[0],
                                   run2_minus_run3_db=db[1]-db[2],above_both_db=above,
                                   absolute_empty_gap_db=drift,above_by_more_than_empty_gap=above>drift))
    write("C1_minus21_contrasts.csv",comparison)


def whitened_search(samples):
    tests, candidates, arrays = [], [], {}
    for state in base.STATES:
        for ref,negative in ((1,3),(3,1)):
            x,t=samples[ref,state]
            b=base.spatial(x[t<5])
            model=background.prepare(b,state)
            dev=[base.spatial(z) for z in time_blocks(x,t,5,10)]
            td=max(base.distance(z,b) for z in dev)
            ts=np.maximum(1,np.stack([background.scores(z,model) for z in dev]).max(axis=(0,2)))
            for test in [2,negative]:
                xx,tt=samples[test,state]
                c=base.spatial(xx)
                dist=base.distance(c,b)
                scores=background.scores(c,model)
                blocks=[base.spatial(z) for z in time_blocks(xx,tt)]
                full=background.peaks(scores[0],ts[0]) if dist>td else []
                peaks=[]
                for i in full:
                    def supported(curve, threshold):
                        return bool(np.any(abs(ANGLES[background.peaks(curve,threshold)]-ANGLES[i])<=9))
                    temporal=sum(base.distance(cc,b)>td and supported(background.scores(cc,model)[0],ts[0]) for cc in blocks)
                    bands=all(supported(scores[k],ts[k]) for k in [1,2])
                    peaks.append(dict(state=state,reference=ref,test=test,angle_parameter_deg=ANGLES[i],
                                      ratio=scores[0,i],quarter_support=temporal,both_bands=bands,
                                      stable=bool(bands and temporal>=3),boundary=abs(ANGLES[i])>=87))
                candidates.extend(peaks)
                tests.append(dict(state=state,reference=ref,test=test,distance=dist,distance_threshold=td,
                                  score_threshold=ts[0],max_score=scores[0].max(),raw_peaks=len(full),
                                  stable_peaks=sum(row["stable"] for row in peaks)))
                arrays[f"{state}_ref{ref}_R{test}"]=scores
    write("background_tests.csv",tests)
    write("blind_background_candidates.csv",candidates,columns=["state","reference","test","angle_parameter_deg",
          "ratio","quarter_support","both_bands","stable","boundary"])
    np.savez_compressed(ROOT/"background_scores.npz",angles=ANGLES,**arrays)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        samples,deeper=read_all()
        changes(samples,deeper)
        templates=fit_r1_reference(samples)
        angular, peaks=raw_spectra(samples,templates)
        fixed_branch(samples,angular,peaks)
        whitened_search(samples)
        frozen=["rules.json","analyze.py","blind_angular_maxima.csv","blind_local_peaks.csv","blind_joint_maxima.csv",
                "blind_background_candidates.csv","background_tests.csv","C1_minus21_contrasts.csv",
                "C1_minus21_before_during_after.csv","paired_changes.csv","ref_run3_validation.csv"]
        (ROOT/"frozen_outputs.json").write_text(json.dumps({f:base.sha(ROOT/f) for f in frozen},indent=2),encoding="utf8")
        print("Decoded three runs, checked both empty references, froze peaks.")
