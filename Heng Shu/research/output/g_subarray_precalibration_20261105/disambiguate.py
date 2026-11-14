from run_check import *

def coh(s,a,b):
    y=Y[np.array(sm.MAP[s])-1]
    d=np.sin(np.deg2rad(b))-np.sin(np.deg2rad(a))
    return abs(np.exp(2j*np.pi*sm.F[:,None]*y[None,:]*d/C).mean(axis=1))

def main():
    verify(R/'model_freeze.json');verify(R/'spectral_predictions_frozen.json')
    cp=pd.read_csv(R/'occupied_candidates.csv');rec=pd.read_csv(A/'occupied_within_records.csv').set_index('index')
    dev=[]
    for mode,first,s in itertools.product(MODES,FIRST,ST):
        q=cp[(cp.model==mode)&(cp.batch_first==first)&(cp.state==s)&cp.stable]
        for a,b in itertools.product(q[q['index']==first].itertuples(),q[q['index']==first+1].itertuples()):
            if abs(a.angle_parameter_deg-b.angle_parameter_deg)<=9:
                dev.append(dict(branch_id=f'{mode}_{first}_{s}_{len(dev)}',model=mode,batch_first=first,state=s,
                    angle_r1=a.angle_parameter_deg,angle_r2=b.angle_parameter_deg,center_dev_deg=(a.angle_parameter_deg+b.angle_parameter_deg)/2))
    d=save(dev,'development_branches.csv');freeze('branch_development_freeze.json',[R/'development_branches.csv',R/'disambiguate.py'])
    rows=[]
    for p in d.to_dict('records'):
        q=cp[(cp.model==p['model'])&(cp['index']==p['batch_first']+2)&(cp.state==p['state'])&cp.stable]
        hits=q[abs(q.angle_parameter_deg-p['center_dev_deg'])<=9]
        for h in list(hits.itertuples()) if len(hits) else [None]:
            angles=[p['angle_r1'],p['angle_r2']]+([h.angle_parameter_deg] if h is not None else [])
            complete=bool(rec.loc[[p['batch_first']+j for j in [0,1,2]],'complete_window'].all())
            rows.append(dict(**p,validation_id=p['branch_id']+f'_v{len(rows)}',angle_r3=h.angle_parameter_deg if h is not None else np.nan,
                complete_rounds=complete,r3_supported=h is not None,complete_repeat=bool(complete and h is not None),
                boundary=max(abs(np.array(angles)))>=87,ambiguous_match=len(hits)>1,
                lower_deg=max(angles)-9,upper_deg=min(angles)+9,human_path_verified=False))
    v=save(rows,'validated_branches.csv');release=[];geo=[];pairs=[];nominal=[]
    chosen=json.loads((R/'development_decision.json').read_text())['candidate']
    for mode,first in itertools.product(MODES,FIRST):
        q=v[(v.model==mode)&(v.batch_first==first)&v.complete_repeat&~v.boundary&~v.ambiguous_match]
        by={s:q[q.state==s].to_dict('records') for s in ST}
        gate=False
        for states in itertools.combinations(GS,3):
            for combo in itertools.product(*(by[s] for s in states)):
                ok,w=region(combo);gate|=ok
                geo.append(dict(model=mode,batch_first=first,states=';'.join(states),branches=';'.join(p['validation_id'] for p in combo),feasible=ok,position_estimate=False))
        release.append(dict(model=mode,batch_first=first,complete_nonboundary_G_states=';'.join(s for s in GS if by[s]),
            complete_nonboundary_G_count=sum(bool(by[s]) for s in GS),three_G_geometry=gate,
            developed_model=mode==chosen,physical_calibration_verified=False,human_output=False,position_output=False))
        for theta in [-39.,24.]:
            votes=[];g1=False
            for s in GS:
                passes=[];peaklists=[]
                for off in [0,1,2]:
                    qq=cp[(cp.model==mode)&(cp['index']==first+off)&(cp.state==s)&cp.stable]
                    hit=qq[abs(qq.angle_parameter_deg-theta)<=9]
                    passes.append(bool(len(hit)));peaklists.append(';'.join(str(t) for t in hit.angle_parameter_deg))
                yes=all(passes) and bool(rec.loc[[first+j for j in [0,1,2]],'complete_window'].all())
                if yes:votes.append(s)
                if s=='G1':g1=yes
                nominal.append(dict(model=mode,batch_first=first,nominal_old_angle=theta,state=s,run1_matches=peaklists[0],run2_matches=peaklists[1],run3_matches=peaklists[2],
                    all_complete_rounds_support=yes,shares_antenna_with_C2=bool(set(sm.MAP[s])&set(sm.MAP['C2'])),
                    calibrated_C2_identity_established=False,physical_choice='UNCONFIRMED'))
        for a,b in itertools.combinations(by['C2'],2):
            rho=coh('C2',a['center_dev_deg'],b['center_dev_deg'])
            if np.mean(rho)<.99:continue
            support={};feasible={}
            for label,p in [('a',a),('b',b)]:
                support[label]=[s for s in GS if any(abs(t['center_dev_deg']-p['center_dev_deg'])<=9 for t in by[s])]
                feasible[label]=False
                for states in itertools.combinations(GS,3):
                    for combo in itertools.product(*(by[s] for s in states)):
                        if region([p,*combo])[0]:feasible[label]=True
            pairs.append(dict(model=mode,batch_first=first,branch_a=a['validation_id'],branch_b=b['validation_id'],angle_a=a['center_dev_deg'],angle_b=b['center_dev_deg'],
                ideal_coherence=float(rho.mean()),G_support_a=';'.join(support['a']),G_support_b=';'.join(support['b']),
                three_G_geometry_a=feasible['a'],three_G_geometry_b=feasible['b'],human_choice='UNCONFIRMED'))
    save(release,'release_decisions.csv');save(geo,'three_G_geometry.csv',['model','batch_first','states','branches','feasible','position_estimate'])
    nd=save(nominal,'nominal_C2_hypotheses.csv')
    save(pairs,'reestimated_C2_alias_pairs.csv',['model','batch_first','branch_a','branch_b','angle_a','angle_b','ideal_coherence','G_support_a','G_support_b','three_G_geometry_a','three_G_geometry_b','human_choice'])
    nc=nd.groupby(['model','batch_first','nominal_old_angle']).agg(persistent_G_votes=('all_complete_rounds_support','sum')).reset_index()
    nc.to_csv(R/'nominal_C2_vote_counts.csv',index=False)
    save([dict(state=s,coherence_min=coh(s,-39,24).min(),coherence_median=np.median(coh(s,-39,24)),coherence_max=coh(s,-39,24).max(),physical_overlap_with_C2=';'.join(map(str,sorted(set(sm.MAP[s])&set(sm.MAP['C2']))))) for s in ST],'ideal_candidate_distinguishability.csv')
    out=['development_branches.csv','validated_branches.csv','release_decisions.csv','three_G_geometry.csv','nominal_C2_hypotheses.csv','reestimated_C2_alias_pairs.csv','nominal_C2_vote_counts.csv','ideal_candidate_distinguishability.csv']
    freeze('decisions_frozen.json',[R/f for f in out])
    print(pd.DataFrame(release).to_string(index=False));print(nc[nc.batch_first==12].to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))
if __name__=='__main__':main()
