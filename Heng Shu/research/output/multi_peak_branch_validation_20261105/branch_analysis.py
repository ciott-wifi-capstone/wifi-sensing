from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.optimize import linprog

R=Path(__file__).resolve().parent
O=R.parent
A=O/'e1_e3_mapping_phase_timing_audit_20261104'
B=O/'four_point_blind_peak_truth_comparison_20261104'
S=O/'same_frequency_spatial_background_20261104'
sys.path.insert(0,str(S))
import spatial_model as sm

GROUPS=[0,12,21,30]
ST=list(sm.MAP)
TOL=9.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(name,paths):
    (R/name).write_text(json.dumps({str(Path(p).resolve()):sha(p) for p in paths},indent=2),encoding='utf8')
def verify_freeze(p):
    for path,h in json.loads(Path(p).read_text()).items():assert sha(path)==h,(path,h)
def write(rows,name,cols=None):
    df=pd.DataFrame(rows,columns=cols);df.to_csv(R/name,index=False);return df

def develop(c):
    rows=[]
    for ref,first,state in itertools.product(['E1','E2'],GROUPS,ST):
        c1=c[(c.reference==ref)&(c['index']==first)&(c.state==state)&c.stable]
        c2=c[(c.reference==ref)&(c['index']==first+1)&(c.state==state)&c.stable]
        for a,b in itertools.product(c1.itertuples(),c2.itertuples()):
            if abs(a.angle_parameter_deg-b.angle_parameter_deg)>TOL:continue
            rows.append(dict(branch_id=f'{ref}_{first}_{state}_{len(rows):03d}',reference=ref,batch_first=first,state=state,
                angle_r1=a.angle_parameter_deg,angle_r2=b.angle_parameter_deg,
                score_r1=a.score,score_r2=b.score,center_dev_deg=(a.angle_parameter_deg+b.angle_parameter_deg)/2,
                temporal_r1=a.temporal_blocks,temporal_r2=b.temporal_blocks))
    return write(rows,'development_branches.csv')

def validate(dev,c,records):
    rows=[];rec=records.set_index('index')
    for a in dev.itertuples(index=False):
        q=c[(c.reference==a.reference)&(c['index']==a.batch_first+2)&(c.state==a.state)&c.stable]
        hits=q[abs(q.angle_parameter_deg-a.center_dev_deg)<=TOL]
        vals=list(hits.itertuples()) if len(hits) else [None]
        for j,h in enumerate(vals):
            complete=bool(rec.loc[[a.batch_first,a.batch_first+1,a.batch_first+2],'complete_window'].all())
            angles=[a.angle_r1,a.angle_r2]+([h.angle_parameter_deg] if h is not None else [])
            rows.append(dict(**a._asdict(),validation_id=a.branch_id+f'_v{j}',angle_r3=h.angle_parameter_deg if h is not None else np.nan,
                score_r3=h.score if h is not None else np.nan,temporal_r3=h.temporal_blocks if h is not None else np.nan,
                validation_matches=len(hits),r3_supported=h is not None,all_runs_complete=complete,complete_repeat=bool(complete and h is not None),
                all_rounds_center_deg=float(np.mean(angles)),branch_span_deg=float(np.ptp(angles)),
                lower_deg=max(angles)-TOL,upper_deg=min(angles)+TOL,
                boundary=any(abs(t)>=87 for t in angles),ambiguous_r3=len(hits)>1,
                physical_angle_calibrated=False,human_path_verified=False))
    return write(rows,'validated_branches.csv')

def region(branches,min_x=1e-6):
    """Feasible bearing wedges in RX-local front half-plane; witness is NOT range."""
    aa=[];bb=[]
    for p in branches:
        y=float(np.mean(np.array(sm.GEO['array_y_local_m'])[np.array(sm.MAP[p['state']])-1]))
        lo=max(-90.,p['lower_deg']);hi=min(90.,p['upper_deg'])
        if lo>hi:return False,None
        if lo>-90+1e-9:aa.append([np.tan(np.deg2rad(lo)),-1.]);bb.append(-y)
        if hi<90-1e-9:aa.append([-np.tan(np.deg2rad(hi)),1.]);bb.append(y)
    z=linprog([0,0],A_ub=np.asarray(aa) if aa else None,b_ub=np.asarray(bb) if aa else None,
              bounds=[(min_x,None),(None,None)],method='highs')
    return bool(z.success),z.x if z.success else None

def geometry_checks(v):
    rows=[];summary=[]
    for ref,first in itertools.product(['E1','E2'],GROUPS):
        vv=v[(v.reference==ref)&(v.batch_first==first)]
        complete=bool(vv.all_runs_complete.all())
        q=vv[vv.complete_repeat & ~vv.boundary & ~vv.ambiguous_r3]
        available={s:q[q.state==s].to_dict('records') for s in ST[:4]}
        feasible_counts={2:0,3:0,4:0}
        for n in [2,3,4]:
            for states in itertools.combinations(ST[:4],n):
                for combo in itertools.product(*(available[s] for s in states)):
                    ok,w=region(combo)
                    feasible_counts[n]+=int(ok)
                    sens={f'feasible_x_ge_{str(mx).replace(".","p")}':region(combo,mx)[0] for mx in [.25,.5,1.]}
                    aux=[]
                    if ok:
                        for p in q[q.state.isin(['C1','C2'])].to_dict('records'):
                            if region([*combo,p])[0]:aux.append(p['validation_id'])
                    rows.append(dict(reference=ref,batch_first=first,independent_G_count=n,states=';'.join(states),
                        branches=';'.join(p['validation_id'] for p in combo),feasible=ok,
                        mathematical_witness_x=(w[0] if ok else np.nan),mathematical_witness_y=(w[1] if ok else np.nan),
                        auxiliary_C_matches=';'.join(aux),**sens,position_estimate=False))
        summary.append(dict(reference=ref,batch_first=first,all_runs_complete=complete,
            repeat_nonboundary_G_states=';'.join(s for s in ST[:4] if available[s]),
            repeat_nonboundary_G_count=sum(bool(available[s]) for s in ST[:4]),
            feasible_G2_combinations=feasible_counts[2],feasible_G3_combinations=feasible_counts[3],feasible_G4_combinations=feasible_counts[4],
            three_G_gate=feasible_counts[3]>0 or feasible_counts[4]>0,physical_calibration_available=False,
            paired_human_transition_available=False,human_output=False,position_output=False))
    cols=['reference','batch_first','independent_G_count','states','branches','feasible','mathematical_witness_x','mathematical_witness_y','auxiliary_C_matches','feasible_x_ge_0p25','feasible_x_ge_0p5','feasible_x_ge_1p0','position_estimate']
    write(rows,'geometry_combinations.csv',cols)
    return write(summary,'release_decisions.csv')

def empty_controls():
    signs=json.loads((O/'empty_pi_sign_validation_20261104/frozen_signs.json').read_text())['pairs']
    sign={'E1':np.ones(3),'E2':np.array(signs['E1_E2']['sign']),'E3':np.array(signs['E1_E3']['sign'])}
    sign['E4']=sign['E3']*np.array(signs['E3_E4']['sign'])
    rows=[];peaks=[];sources=[];arrays={}
    for ref,target,off,mode,state in itertools.product(['E1','E2'],sm.CAP,[1,2],['raw','fixed_pi'],ST):
        idx=sm.CAP[target]+off
        p=sm.B/'cache'/f'{idx:03d}_{state}.npz'
        x=np.load(p)['x'][64:]
        if mode=='fixed_pi':x=x*sign[target][None,:,None]
        mpath=S/f'occupied_model_{ref}_{state}.npz';z=np.load(mpath)
        m=sm.prepare(z['background'],state)
        stat,pp,ar=sm.assess(x,m,float(z['distance_threshold']),z['score_thresholds'])
        role='model_development' if target==ref and off==1 else ('within_source_run3' if target==ref else 'cross_record_diagnostic')
        base=dict(reference=ref,target=target,run=off+1,mode=mode,state=state,role=role,packets=len(x),composed_sign=target=='E4' and mode=='fixed_pi')
        rows.append(dict(**base,**stat))
        for pk in pp:peaks.append(dict(**base,**pk))
        sources.extend([dict(path=str(p),sha256=sha(p)),dict(path=str(mpath),sha256=sha(mpath))])
        arrays[f'{ref}_{target}_{off+1}_{mode}_{state}']=ar['scores']
    write(rows,'empty_control_scores.csv')
    cp=write(peaks,'empty_control_candidates.csv')
    write(sources,'empty_control_sources.csv').drop_duplicates().to_csv(R/'empty_control_sources.csv',index=False)
    np.savez_compressed(R/'empty_control_spectra.npz',angles=sm.ANG,**arrays)
    repeated=[]
    for ref,target,mode,state in itertools.product(['E1','E2'],sm.CAP,['raw','fixed_pi'],ST):
        q=cp[(cp.reference==ref)&(cp.target==target)&(cp['mode']==mode)&(cp.state==state)&cp.stable]
        for a,b in itertools.product(q[q.run==2].itertuples(),q[q.run==3].itertuples()):
            if abs(a.angle_parameter_deg-b.angle_parameter_deg)<=TOL:
                repeated.append(dict(reference=ref,target=target,mode=mode,state=state,angle_run2=a.angle_parameter_deg,
                    angle_run3=b.angle_parameter_deg,center_deg=(a.angle_parameter_deg+b.angle_parameter_deg)/2,
                    boundary=max(abs(a.angle_parameter_deg),abs(b.angle_parameter_deg))>=87,
                    score_run2=a.score,score_run3=b.score,run2_development=target==ref))
    return write(repeated,'empty_repeated_branches.csv')

def annotations(v,e,decisions):
    rows=[]
    for p in v.to_dict('records'):
        q=e[(e.reference==p['reference'])&(e.state==p['state'])&(e['mode']=='fixed_pi')&(~e.boundary)]
        hits=q[abs(q.center_deg-p['center_dev_deg'])<=TOL]
        other=v[(v.reference==p['reference'])&(v.batch_first!=p['batch_first'])&(v.state==p['state'])&v.complete_repeat & ~v.boundary]
        count=other[abs(other.center_dev_deg-p['center_dev_deg'])<=TOL].batch_first.nunique()
        alt=v[(v.reference!=p['reference'])&(v.batch_first==p['batch_first'])&(v.state==p['state'])&v.r3_supported]
        alt=alt[abs(alt.center_dev_deg-p['center_dev_deg'])<=TOL]
        gate=decisions[(decisions.reference==p['reference'])&(decisions.batch_first==p['batch_first'])].iloc[0]
        reasons=[]
        if not p['r3_supported']:reasons.append('third_round_not_supported')
        if not p['all_runs_complete']:reasons.append('incomplete_round')
        if p['boundary']:reasons.append('boundary_peak')
        if p['ambiguous_r3']:reasons.append('multiple_third_round_matches')
        if not len(alt):reasons.append('no_alternative_reference_repeat')
        if len(hits):reasons.append('similar_parameter_in_empty_control')
        if not gate.three_G_gate:reasons.append('no_three_independent_G_geometry_support')
        reasons.extend(['uncalibrated_fixed_response','no_paired_human_transition'])
        rows.append(dict(**p,alternative_reference_support=bool(len(alt)),empty_nearby_count=len(hits),
            empty_nearby_examples=';'.join(f'{r.target}:{r.center_deg:g}' for r in hits.itertuples()),
            other_occupied_batch_count=count,release_status='UNCONFIRMED',failure_reasons=';'.join(reasons)))
    return write(rows,'branch_evidence.csv')

def candidate_ledger(c,v):
    rows=[]
    for p in c.to_dict('records'):
        run=p['index']-p['batch_first']+1
        q=v[(v.reference==p['reference'])&(v.batch_first==p['batch_first'])&(v.state==p['state'])]
        matched=q[np.isclose(q[f'angle_r{run}'],p['angle_parameter_deg'])]
        rows.append(dict(**p,run=run,linked_branch_ids=';'.join(matched.validation_id),
                         tracking_status=('unstable_peak_retained' if not p['stable'] else ('linked' if len(matched) else 'stable_unlinked_retained'))))
    write(rows,'candidate_ledger.csv')

def main():
    for fname,h in json.loads((B/'blind_predictions_frozen.json').read_text()).items():assert sha(B/fname)==h
    for fname,h in json.loads((A/'within_batch_predictions.json').read_text()).items():assert sha(A/fname)==h
    for fname,h in json.loads((S/'occupied_model_freeze.json').read_text()).items():assert sha(S/fname)==h
    c=pd.read_csv(B/'all_frozen_candidates.csv')
    records=pd.read_csv(A/'occupied_within_records.csv')[['index','batch_first','complete_window','trailing_s']]
    inputs=[B/'all_frozen_candidates.csv',A/'occupied_within_records.csv',A/'occupied_within_invariants.csv',
            O/'static_human_aoa_validation_20261101/geometry.json',S/'spatial_model.py',
            O/'empty_pi_sign_validation_20261104/frozen_signs.json',R/'PROTOCOL.md',R/'branch_analysis.py']
    inputs+=list(S.glob('occupied_model_*.npz'))
    freeze('input_freeze.json',inputs)
    dev=develop(c[c['index'].isin([i+j for i in GROUPS for j in [0,1]])])
    freeze('development_freeze.json',[R/'development_branches.csv',R/'PROTOCOL.md',R/'branch_analysis.py'])
    v=validate(dev,c,records)
    candidate_ledger(c,v)
    print('development branches',len(dev),'third round supported',int(v.r3_supported.sum()),flush=True)
    decisions=geometry_checks(v)
    e=empty_controls()
    ev=annotations(v,e,decisions)
    predictions=['development_branches.csv','validated_branches.csv','branch_evidence.csv','candidate_ledger.csv',
                 'geometry_combinations.csv','release_decisions.csv','empty_control_scores.csv','empty_control_candidates.csv',
                 'empty_repeated_branches.csv','empty_control_sources.csv','empty_control_spectra.npz']
    freeze('predictions_frozen.json',[R/f for f in predictions])
    print(decisions.to_string(index=False))
    print(ev.query('reference=="E1"')[['batch_first','state','angle_r1','angle_r2','angle_r3','complete_repeat','boundary','empty_nearby_examples']].to_string(index=False))
if __name__=='__main__':
    with sm.threadpool_limits(limits=1):main()
