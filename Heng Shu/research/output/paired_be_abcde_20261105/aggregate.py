"""Score registered outputs. Check sensitivity to within-record sampling for new B/E."""
from pathlib import Path
import sys
import importlib.util
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/"new_a_paired_20261105"
BC = ROOT.parent/"paired_bc_20261105"
CD = ROOT.parent/"real_cd_paired_20261105"
PRIMARY = ("A", "B", "C", "D", "E")
FOLDERS = {"A": BASE, "B": ROOT/"B", "C": CD/"C", "D": CD/"D", "E": ROOT/"E",
           "B_earlier": BC/"B", "C_earlier": BC/"C"}
POINTS = {"A": (.8,.6), "B": (2.8,1.2), "C": (.8,-1.2), "D": (2.4,-1.6), "E": (1.64,-2.01)}
POINTS.update({"B_earlier": POINTS["B"], "C_earlier": POINTS["C"]})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewer():
    previous = load("paired_report_cd", CD/"report.py")
    previous.ROOT = previous.bc.ROOT = ROOT
    previous.bc.FOLDERS, previous.bc.POINTS = FOLDERS, POINTS
    previous.SITES = tuple(FOLDERS)
    return previous


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/name, index=False, encoding="utf-8-sig")
    return df


def validate_frozen(bc):
    for d,registry in [(ROOT,"blind_registry.json"),(BASE,"frozen_outputs.json"),
                       (BC,"blind_registry.json"),(CD,"blind_registry.json")]:
        hashes=json.loads((d/registry).read_text(encoding="utf8"))
        assert all(bc.digest(d/key) == sha for key,sha in hashes.items())


def summarize_directions(bc, angles, contrast):
    rows = []
    for site in PRIMARY:
        r = angles[angles.point==site].set_index("state")
        g2, g3, fusion = r.loc["G2"], r.loc["G3"], r.loc["G fusion"]
        cs = contrast.query('point==@site and metric=="covariance_distance"')
        x, y = POINTS[site]
        rows.append(dict(point=site, X_m=x, Y_m=y,
                         source=Path(bc.read(site,"raw_runs").source.iloc[0]).parents[2].name,
                         theory_centre_deg=fusion.theory_deg, theory_G2_deg=g2.theory_deg,
                         G2_raw_R2=g2.raw_r2, G2_R1ref_R2=g2.per_tone_r2,
                         four_G_raw_R1=fusion.raw_r1,four_G_raw_R2=fusion.raw_r2,four_G_raw_R3=fusion.raw_r3,
                         four_G_R1ref_R2=fusion.per_tone_r2,
                         four_G_raw_abs_diff_deg=abs(fusion.raw_r2-fusion.theory_deg),
                         cov_states_ratio_gt_1=int((cs.ratio>1).sum()),cov_states_passing_all_quarters=int(cs.all_four_quarters_pass.sum()),
                         independently_confirmed_human_aoa_deg=np.nan,
                         theory_G3_deg=g3.theory_deg, G3_R1ref_R1=g3.per_tone_r1, G3_R1ref_R2=g3.per_tone_r2,
                         G3_R1ref_R3=g3.per_tone_r3, G3_R1ref_abs_diff_deg=g3.per_tone_abs_difference,
                         G3_reference_pairs_passed=int(g3.supported_pairs)))
    return save("ABCDE_angles.csv",rows)


def check_empty_edges(bc, matches):
    # Reject edge peaks as candidates, while preserving edge peaks as negative controls
    # when their +/-9 degree neighbourhood overlaps an interior foreground peak.
    tests = {"raw_background": ("blind_background_candidates", "angle_parameter_deg"),
             "empirical_background": ("exploratory_empirical_bg_peaks", "angle")}
    checks = []
    groups = matches.groupby(["site", "mode", "state"], sort=False)
    for (site, mode, state), group in groups:
        stem, col = tests[mode]
        data = bc.read(site,stem)
        stable = data[(data.state==state)&data.stable]
        for row in group.itertuples():
            pre = stable.query("reference==1 and test==3")[col].sub(row.pre_angle).abs().le(9).any()
            post = stable.query("reference==3 and test==1")[col].sub(row.post_angle).abs().le(9).any()
            checks.append(dict(site=site,mode=mode,state=state,pre_angle=row.pre_angle,post_angle=row.post_angle,
                               pre_ratio=row.pre_ratio,post_ratio=row.post_ratio,
                               pre_empty_control_near_including_edges=bool(pre),post_empty_control_near_including_edges=bool(post),
                               clears_empty_controls=not(pre or post),theory_subarray_deg=row.theoretical_deg,
                               pre_difference=row.pre_difference,post_difference=row.post_difference))
    return save("two_reference_including_empty_edges.csv",checks)


def geometric_examples(bc):
    rows = []
    for state, u, v in [("C1",42,-30),("C2",45,-18),("C2",18,-45),("C2",48,-18)]:
        y=np.array(bc.GEOM["array_y_local_m"])[np.array(bc.GEOM["state_map"][state])-1]
        frequency = 5.23e9+np.arange(-58,59,4)*312500.
        delta = np.sin(np.radians(u))-np.sin(np.radians(v))
        rho = abs(np.exp(2j*np.pi/299792458.*frequency[:,None]*y[None,:]*delta).mean(axis=1))
        rows.append(dict(state=state,angle_deg=u,alternative_deg=v,ideal_min=rho.min(),
                         ideal_median=np.median(rho),ideal_max=rho.max()))
    return save("steering_similarity.csv",rows)


def old_b_branch_profile():
    from scipy.ndimage import maximum_filter1d
    rows=[]
    for site in ["B_earlier","B"]:
        with np.load(FOLDERS[site]/"exploratory_empirical_spectra.npz") as z:
            angles=z["angles"]
            j=int(np.flatnonzero(angles==-63)[0])
            for run in [1,2,3]:
                curve=z[f"C1_R{run}"]
                local=np.flatnonzero(curve==maximum_filter1d(curve,size=3,mode="nearest"))
                ranked=sorted(local,key=lambda i:-curve[i])
                rows.append(dict(site=site,run=run,peak_deg=angles[ranked[0]],
                                 fixed_minus63_db=10*np.log10(curve[j]/curve.max()),
                                 fixed_minus63_local_rank=ranked.index(j)+1 if j in ranked else None))
    return save("B_C1_minus63_fixed_branch.csv",rows)


def fusion_checks(bc):
    # Same fixed grid, frequency bands, quarters and state weights in all six new records.
    methods=load("methods_for_fusion_check",BASE/"analyze.py")
    base=methods.base
    spectra_rows=[]; equal_cov={}; window_rows=[]
    for site in ["B","E"]:
        records=bc.read(site,"raw_runs")
        quality=bc.read(site,"state_quality").set_index(["run","state"])
        with np.load(FOLDERS[site]/"angular_spectra.npz") as saved:
            for row in records.itertuples():
                path=Path(row.source)
                folder=path.parent
                meta=base.read(folder/"run.json")
                qc=base.read(folder/"metadata_qc.json")
                x,host,_=base.decoder.decode(path)
                ta,_=base.frame_details(path)
                t=(host-meta["start_ns"])/1e9
                starts,order,guard=base.decoder.align(folder/"events.log",meta["start_ns"]/1e9)
                visits,eligible=base.select_visits(t,starts,order,10.,guard,ta==base.APPARENT_TA)
                per_second=np.histogram(t[(ta==base.APPARENT_TA)&(t>=0)&(t<10)],bins=np.arange(11))[0]
                assert per_second.sum()==row.target_frames
                window_rows.append(dict(point=site,run=row.run,packets=row.target_frames,
                                        target_packets_per_second="/".join(map(str,per_second)),
                                        leading_gap_ms=meta["window"]["leading_gap_s"]*1e3,
                                        trailing_gap_ms=meta["window"]["trailing_gap_s"]*1e3,
                                        longest_formal_gap_ms=row.max_gap_ms,metadata_frame_mismatches=sum(qc["mismatches"].values())))
                views={}
                for state, visits_in_state in visits.items():
                    ids=[i for _,i in visits_in_state]
                    selected=np.concatenate(ids)
                    assert len(ids)==eligible[state] and len(selected)==quality.loc[row.run,state].guarded_packets
                    # The primary estimator weights packets equally. This sensitivity weights complete switch visits equally.
                    eq=np.mean([base.spatial(x[i].astype(complex)) for i in ids],axis=0)
                    equal_cov[site,row.run,state]=eq
                    if state not in base.GROUPS:
                        continue
                    xx=x[selected].astype(complex)
                    tt=t[selected]
                    steer=base.steering(state)
                    full=base.angular_spectra(base.spatial(xx),steer)
                    assert np.allclose(full,saved[f"{state}_R{row.run}_raw"])
                    views.setdefault(("packet","full"),[]).append(full)
                    views.setdefault(("visit","full"),[]).append(base.angular_spectra(eq,steer))
                    for i,b in enumerate(methods.time_blocks(xx,tt),1):
                        views.setdefault(("packet",f"Q{i}"),[]).append(base.angular_spectra(base.spatial(b),steer))
                for (weight,window),samples in views.items():
                    assert len(samples)==4
                    combined=np.mean(samples,axis=0)
                    for k,band in enumerate(["full","low","high"]):
                        curve=combined[k]
                        peak=curve.argmax()
                        local=sorted(base.local_indices(curve),key=lambda i:-curve[i])
                        assert peak==local[0]
                        alt=local[1] if len(local)>1 else None
                        spectra_rows.append(dict(point=site,run=row.run,weight=weight,window=window,band=band,
                            peak_deg=base.ANGLE[peak],next_local_deg=base.ANGLE[alt] if alt is not None else None,
                            next_local_db=10*np.log10(curve[alt]/curve[peak]) if alt is not None else None))
                        if (weight,window,band)==("packet","full","full"):
                            assert np.argmax(saved[f"fusion_R{row.run}_raw"])==peak
    save("raw_fusion_sensitivity.csv",spectra_rows)
    save("capture_window_check.csv",window_rows)
    rows=[]
    for site in ["B","E"]:
        for state in base.STATES:
            c={r:equal_cov[site,r,state] for r in [1,2,3]}
            d12,d23,d13=(base.distance(c[r],c[s]) for r,s in [(1,2),(2,3),(1,3)])
            rows.append(dict(point=site,state=state,D12=d12,D23=d23,D13=d13,ratio=min(d12,d23)/d13))
    save("equal_visit_contrast.csv",rows)


def main():
    previous=reviewer(); bc=previous.bc
    validate_frozen(bc)
    contrast=bc.contrast_summary()
    angles=bc.angle_summary()
    matches=previous.two_reference_candidates()
    controlled=check_empty_edges(bc,matches)
    geometric_examples(bc)
    old_b_branch_profile()
    fixed=summarize_directions(bc,angles,contrast)
    with threadpool_limits(limits=1):
        fusion_checks(bc)
    print(fixed.to_string(index=False))
    print("B/E primary covariances\n",contrast.query('point in ["B","E"] and metric=="covariance_distance"').to_string(index=False))
    print("Paired stable candidates surviving empty vetoes including edge controls (B/E)\n",controlled.query('site in ["B","E"] and clears_empty_controls')[["site","mode","state","pre_angle","post_angle","theory_subarray_deg"]].to_string(index=False))


if __name__=="__main__":
    main()

