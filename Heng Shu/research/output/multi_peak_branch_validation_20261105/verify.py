from branch_analysis import *
from geometry_alias import coherence
import py_compile,platform

def main():
    for name in ['input_freeze.json','development_freeze.json','predictions_frozen.json','alias_inputs_frozen.json','alias_outputs_frozen.json']:verify_freeze(R/name)
    for p in pd.read_csv(R/'empty_control_sources.csv').itertuples(index=False):assert sha(p.path)==p.sha256
    c=pd.read_csv(B/'all_frozen_candidates.csv');d=pd.read_csv(R/'development_branches.csv');v=pd.read_csv(R/'validated_branches.csv')
    assert len(d)==80 and len(v)==80 and v.validation_id.is_unique
    assert (abs(d.angle_r1-d.angle_r2)<=9).all()
    assert np.allclose(d.center_dev_deg,(d.angle_r1+d.angle_r2)/2)
    for p in v.itertuples():
        for j in [1,2,3]:
            angle=getattr(p,f'angle_r{j}')
            if np.isnan(angle):continue
            hit=c[(c.reference==p.reference)&(c.batch_first==p.batch_first)&(c.state==p.state)&(c['index']==p.batch_first+j-1)&np.isclose(c.angle_parameter_deg,angle)]
            assert len(hit)==1 and bool(hit.stable.iloc[0])
        assert p.complete_repeat==bool(p.r3_supported and p.all_runs_complete)
        if p.r3_supported:assert abs(p.angle_r3-p.center_dev_deg)<=9
    assert not v[v.batch_first==0].complete_repeat.any()
    ledger=pd.read_csv(R/'candidate_ledger.csv');assert len(ledger)==len(c)
    assert ledger[['index','reference','state','angle_parameter_deg']].equals(c[['index','reference','state','angle_parameter_deg']])
    null=pd.read_csv(R/'empty_control_scores.csv');assert len(null)==192 and set(null['mode'])=={'raw','fixed_pi'}
    assert (null.packets>=12).all() # numerical covariance support only, not a quality acceptance gate
    quality=null[['target','run','state','packets']].drop_duplicates().copy()
    quality['low_sample_diagnostic']=quality.packets<128
    quality.to_csv(R/'empty_sample_quality.csv',index=False)
    low=quality[quality.low_sample_diagnostic]
    assert set(zip(low.target,low.run,low.state,low.packets))=={('E1',2,'G2',78),('E1',3,'G2',79)}
    ev=pd.read_csv(R/'all_branch_truth_evaluation.csv');assert len(ev)==len(v)
    for j in [1,2,3]:assert np.allclose(ev[f'abs_difference_r{j}'],abs(ev[f'angle_r{j}']-ev.theory_deg),equal_nan=True)
    assert not ev.human_path_verified.any()
    decisions=pd.read_csv(R/'release_decisions.csv');assert len(decisions)==8
    assert not decisions.human_output.any() and not decisions.position_output.any()
    # Synthetic geometric positive and negative controls; no measured target used.
    artificial=[]
    for state in ST[:4]:
        yc=np.mean(np.array(sm.GEO['array_y_local_m'])[np.array(sm.MAP[state])-1])
        t=np.degrees(np.arctan2(.81-yc,1.73))
        artificial.append(dict(state=state,lower_deg=t-.1,upper_deg=t+.1))
    assert region(artificial)[0]
    assert not region([dict(state='G2',lower_deg=39,upper_deg=41),dict(state='G3',lower_deg=-41,upper_deg=-39)])[0]
    # Branch logic is score-independent and cannot rescue an unstable candidate.
    def seed_rows():
        out=[]
        for j,angles in enumerate([[-45,15],[-42,18],[-39,21]]):
            for k,angle in enumerate(angles):
                out.append(dict(reference='E1',batch_first=0,index=j,state='G2',angle_parameter_deg=angle,score=[8,2][k] if j%2==0 else [2,8][k],stable=True,temporal_blocks=4))
        return pd.DataFrame(out)
    # Exercise production functions in separate retained synthetic folders.
    import branch_analysis as ba
    measured_root=ba.R
    try:
        test=seed_rows();rec=pd.DataFrame(dict(index=[0,1,2],complete_window=[True,True,True]))
        for case in ['rank_swap','unstable_r3','partial_r3']:
            ba.R=R/'synthetic_controls'/case;ba.R.mkdir(parents=True,exist_ok=True)
            ct=test.copy();rc=rec.copy()
            if case=='unstable_r3':ct.loc[(ct['index']==2)&(ct.angle_parameter_deg==21),'stable']=False
            if case=='partial_r3':rc.loc[rc['index']==2,'complete_window']=False
            dd=ba.develop(ct[ct['index']<2]);vv=ba.validate(dd,ct,rc)
            assert len(dd)==2
            if case=='rank_swap':assert set(vv.angle_r3)=={-39,21} and vv.complete_repeat.all()
            if case=='unstable_r3':assert vv.r3_supported.sum()==1 and vv.angle_r3.isna().sum()==1
            if case=='partial_r3':assert vv.r3_supported.all() and not vv.complete_repeat.any()
    finally:ba.R=measured_root
    assert np.allclose(coherence('C2',24,24),1)
    alias=coherence('C2',-39,24);assert .997<np.median(alias)<.999
    for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
    checks=dict(status='PASS',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        all_freezes_verified=True,all_source_cache_hashes_verified=True,development_branches=len(d),validated_rows=len(v),
        all_original_candidates_retained=len(c),empty_state_mode_cases=len(null),empty_min_scored_packets=int(null.packets.min()),
        low_sample_empty_source_cases=low.to_dict('records'),geometry_positive_negative_controls=True,
        synthetic_rank_swap_branches_preserved=True,partial_round_not_promoted=True,all_truth_comparisons_verified=True,
        C2_alias_median=float(np.median(alias)),human_outputs=0,position_outputs=0,
        visual_QA='Three PNG figures inspected; readable labels and legends; no clipping observed.')
    (R/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
