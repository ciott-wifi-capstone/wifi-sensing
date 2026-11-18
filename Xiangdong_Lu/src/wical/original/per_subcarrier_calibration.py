"""Run1-only per-subcarrier broadside graph calibration and held-out validation.
Requires numpy only. No smoothing, temporal stitching, AoA, or D03 processing.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

MAP = {'G1':[1,2,3], 'G2':[4,5,6], 'G3':[7,8,9], 'G4':[10,11,12], 'C1':[1,8,6], 'C2':[10,8,6]}
EDGES = [(s,a,b) for s,m in MAP.items() for a,b in [(m[1],m[0]),(m[2],m[1]),(m[2],m[0])]]
# Engineering acceptance criteria fixed before examining held-out runs; not a standard.
POLICY = dict(min_training_R=.5, min_samples=30, residual_mean_max_deg=5,
              residual_p95_max_deg=10, worst_pair_mean_max_deg=10,
              shared_p95_max_deg=5, heldout_min_edge_R=.9)
def wrap(x): return (np.asarray(x)+180)%360-180
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.bool_): return bool(x)
    return x
def save(p,x): p.write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8')
def csvsave(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def stats(x):
    x=np.abs(np.asarray(x,float)); x=x[np.isfinite(x)]
    return dict(count=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x))) if len(x) else dict(count=0,mean=None,median=None,p95=None,max=None)
def measure(h,a,b):
    aa,bb=h[:,a,:],h[:,b,:]
    valid=np.isfinite(aa)&np.isfinite(bb)&(np.abs(aa)>0)&(np.abs(bb)>0)
    u=np.zeros(aa.shape,complex)
    u[valid]=aa[valid]/np.abs(aa[valid])*np.conj(bb[valid]/np.abs(bb[valid]))
    return u.sum(axis=0),valid.sum(axis=0)
def phase_r(z,n):
    mean=np.divide(z,n,out=np.zeros_like(z),where=n>0)
    r=np.abs(mean); p=np.degrees(np.angle(mean)); p[(n==0)|(r<1e-12)]=np.nan
    return p,r
def solve(p,r,n):
    offsets=np.full((12,30),np.nan); diagnostics=[]
    A=np.zeros((18,12))
    for e,(_,a,b) in enumerate(EDGES): A[e,a-1]=1; A[e,b-1]=-1
    for k in range(30):
        valid=np.isfinite(p[:,k])&(r[:,k]>=POLICY['min_training_R'])&(n[:,k]>=POLICY['min_samples'])
        reached={0}
        while True:
            nxt=reached|{v for e,(_,a,b) in enumerate(EDGES) if valid[e] and ({a-1,b-1}&reached) for v in (a-1,b-1)}
            if nxt==reached: break
            reached=nxt
        nodes=sorted(reached-{0}); use=valid.copy()
        for e,(_,a,b) in enumerate(EDGES): use[e] &= a-1 in reached and b-1 in reached
        w=n[:,k]*r[:,k]**2 # concentration and valid packet count; no equal weighting
        H=np.zeros((12,12),complex)
        for e,(_,a,b) in enumerate(EDGES):
            if use[e]:
                z=w[e]*np.exp(1j*np.radians(p[e,k])); H[a-1,b-1]+=z; H[b-1,a-1]+=z.conjugate()
        idx=sorted(reached); _,v=np.linalg.eigh(H[np.ix_(idx,idx)])
        q=np.zeros(12); q[idx]=np.degrees(np.angle(v[:,-1])); q=wrap(q-q[0])
        if nodes:
            B=A[use][:,nodes]; sw=np.sqrt(w[use]/np.max(w[use]))
            for _ in range(100):
                residual=wrap(A[use]@q-p[use,k])
                step=np.linalg.lstsq(B*sw[:,None],-residual*sw,rcond=None)[0]
                q[nodes]+=step
                if np.max(np.abs(step))<1e-10: break
        offsets[idx,k]=wrap(q[idx]); offsets[0,k]=0
        for e,(s,a,b) in enumerate(EDGES):
            diagnostics.append(dict(state=s,pair=f'{a}-{b}',k=k,observed_phase_deg=p[e,k],R=r[e,k],valid_samples=n[e,k],weight=w[e] if use[e] else 0,used=use[e],weak_R_below_09=r[e,k]<.9,residual_deg=wrap(p[e,k]-(offsets[a-1,k]-offsets[b-1,k])),status='used' if use[e] else 'excluded_low_R_samples_or_disconnected'))
    return offsets,diagnostics
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); out=args.output
    if out.exists(): raise ValueError('Refusing to overwrite output directory')
    meta=read(args.data_dir/'occurrences.json'); scalar=read(args.data_dir/'calibration_run1.json'); config=read(args.config)
    assert meta['intervals']=='half-open' and meta['continuous_across_occurrences'] is False
    assert config['switching']['chain_to_antenna']==MAP and scalar['mapping']==MAP and scalar['training_run']==1
    x=config['array']['subcarrier_indices']; assert len(x)==30 and np.all(np.diff(x)>0)
    inputs=[args.config,args.data_dir/'occurrences.json',args.data_dir/'calibration_run1.json']+[args.data_dir/(s+'.npz') for s in MAP]
    hashes={str(p):digest(p) for p in inputs}
    arrays={}; owners={}
    for s in MAP:
        with np.load(args.data_dir/(s+'.npz'),allow_pickle=False) as d:
            h=d['csi']; t=d['timestamp'].reshape(-1); assert np.array_equal(d['antenna_map'],MAP[s])
        assert h.shape==(len(t),90)
        arrays[s]=(h.reshape(-1,3,30),t); owners[s]=np.zeros(len(t),bool)
    assert {o['run'] for o in meta['occurrences']}=={1,2,3}
    seen=set()
    for o in meta['occurrences']:
        s=o['state']; lo,hi=o['state_row_start'],o['state_row_end']; h,t=arrays[s]
        assert (o['run'],o['occurrence']) not in seen; seen.add((o['run'],o['occurrence']))
        assert 0<=lo<hi<=len(t) and not owners[s][lo:hi].any()
        assert hi-lo==o['packet_end_index']-o['packet_start_index']
        assert np.isfinite(t[lo:hi]).all() and np.all(np.diff(t[lo:hi])>0)
        assert np.all((t[lo:hi]>=o['safe_start_s'])&(t[lo:hi]<o['safe_end_s']))
        owners[s][lo:hi]=True
    assert all(v.all() for v in owners.values())
    # Fit is completed using Run1 statistics before computing ANY held-out statistics.
    occurrence_rows=[]; cache={}; counts={}; summary={}; details=[]; shared_rows=[]
    scalar_offsets=np.array([scalar['phase_offsets_deg'][str(a)] for a in range(1,13)])[:,None]*np.ones((1,30))
    for run in (1,2,3):
        z=np.zeros((18,30),complex); n=np.zeros((18,30),int); counts[run]={}
        for o in (v for v in meta['occurrences'] if v['run']==run):
            s=o['state']; lo,hi=o['state_row_start'],o['state_row_end']; h=arrays[s][0][lo:hi]
            ct=counts[run].setdefault(s,dict(rows=0,occurrences=0)); ct['rows']+=hi-lo; ct['occurrences']+=1
            for e,(state,a,b) in enumerate(EDGES):
                if state!=s: continue
                zz,nn=measure(h,MAP[s].index(a),MAP[s].index(b)); pp,rr=phase_r(zz,nn); z[e]+=zz; n[e]+=nn
                cache[(run,o['occurrence'],a,b)]=pp
                for k in range(30): occurrence_rows.append(dict(run=run,occurrence=o['occurrence'],state=s,pair=f'{a}-{b}',subcarrier_index=x[k],phase_deg=pp[k],R_k=rr[k],valid_samples=nn[k]))
        p,r=phase_r(z,n)
        if run==1:
            offsets,diagnostics=solve(p,r,n)
            # Recover known offsets, sign convention, and disconnected node behavior.
            truth=wrap(np.arange(12)[:,None]*29+np.arange(30)[None,:]*np.arange(12)[:,None]); truth-=truth[0]
            synthetic=np.array([wrap(truth[a-1]-truth[b-1]) for _,a,b in EDGES])
            recovered,_=solve(synthetic,np.ones_like(synthetic),np.full_like(synthetic,100))
            assert np.max(np.abs(wrap(recovered-truth)))<1e-7
            disconnected=np.ones_like(synthetic)
            for e,(_,a,b) in enumerate(EDGES):
                if 12 in (a,b): disconnected[e]=0
            rec,_=solve(synthetic,disconnected,np.full_like(synthetic,100)); assert np.isnan(rec[11]).all()
            frozen=offsets.copy()
        summary[run]={}
        for method,off in [('scalar',scalar_offsets),('per_subcarrier',offsets)]:
            residual=np.array([wrap(p[e]-(off[a-1]-off[b-1])) for e,(_,a,b) in enumerate(EDGES)])
            pair_metrics={}
            for e,(s,a,b) in enumerate(EDGES):
                pair_metrics[s+':'+f'{a}-{b}']=dict(residual=stats(residual[e]),R_min=float(r[e].min()),R_median=float(np.median(r[e])))
                for k in range(30): details.append(dict(run=run,method=method,state=s,pair=f'{a}-{b}',subcarrier_index=x[k],residual_deg=residual[e,k],R_k=r[e,k],valid_samples=n[e,k]))
                # Direct complex correction on a real occurrence verifies algebra and R invariant.
                o=next(o for o in meta['occurrences'] if o['run']==run and o['state']==s)
                h=arrays[s][0][o['state_row_start']:o['state_row_end']]
                hh=h*np.exp(-1j*np.radians(off[np.array(MAP[s])-1]))[None,:,:]
                zz,nn=measure(hh,MAP[s].index(a),MAP[s].index(b)); actual,actual_r=phase_r(zz,nn)
                raw_z,raw_n=measure(h,MAP[s].index(a),MAP[s].index(b)); _,raw_r=phase_r(raw_z,raw_n)
                expected=wrap(cache[(run,o['occurrence'],a,b)]-(off[a-1]-off[b-1]))
                mask=np.isfinite(expected)
                assert np.allclose(wrap(actual[mask]-expected[mask]),0,atol=1e-7)
                assert np.allclose(actual_r[mask],raw_r[mask],atol=1e-10)
            summary[run][method]=dict(residual=stats(residual),pairs=pair_metrics,per_subcarrier=[dict(subcarrier_index=x[k],**stats(residual[:,k])) for k in range(30)])
        # Adjacent safe C1/C2 observations only; no packet series concatenation.
        lookup={o['occurrence']:o for o in meta['occurrences'] if o['run']==run}
        for oid,o in lookup.items():
            if o['state']!='C1': continue
            nxt=lookup.get(oid+1)
            assert nxt and nxt['state']=='C2' and o['safe_end_s']<=nxt['safe_start_s']
            delta=wrap(cache[(run,oid+1,6,8)]-cache[(run,oid,6,8)])
            for k in range(30): shared_rows.append(dict(run=run,C1_occurrence=oid,C2_occurrence=oid+1,subcarrier_index=x[k],C2_minus_C1_deg=delta[k]))
        summary[run]['shared_adjacent']=stats([v['C2_minus_C1_deg'] for v in shared_rows if v['run']==run])
        e1=EDGES.index(('C1',6,8)); e2=EDGES.index(('C2',6,8))
        summary[run]['shared_run_means']=stats(wrap(p[e2]-p[e1]))
        summary[run]['min_edge_R']=float(r.min())
        assert np.array_equal(offsets,frozen,equal_nan=True)
    for run in (2,3):
        v=summary[run]; p=v['per_subcarrier']; s=v['scalar']; m=p['residual']
        checks=dict(all_offsets_identifiable=bool(np.isfinite(offsets).all()),mean_below_5=m['mean']<=5,p95_below_10=m['p95']<=10,
                    every_pair_mean_below_10=all(z['residual']['mean'] is not None and z['residual']['mean']<=10 for z in p['pairs'].values()),
                    mean_and_p95_improve=m['mean']<s['residual']['mean'] and m['p95']<s['residual']['p95'],
                    shared_p95_below_5=v['shared_adjacent']['p95']<=5,all_edges_R_at_least_09=v['min_edge_R']>=.9)
        v['acceptance_checks']=checks; v['passed']=all(checks.values())
    validated=all(summary[r]['passed'] for r in (2,3))
    assert all(digest(Path(p))==v for p,v in hashes.items())
    out.mkdir(parents=True)
    calibration=dict(subcarrier_indices=x,phase_offsets_deg={a:offsets[a-1] for a in range(1,13)},reference_antenna=1,mapping=MAP,training_run=1,source_session=scalar['source_session'],training_counts_by_state=counts[1],method='Independent per-k circular weighted least squares; eigenvector initialization; weight=N*R^2; R<0.5 or N<30 excluded; no smoothing or interpolation',edge_residual_diagnostics=diagnostics,scope='fixed per-subcarrier phase-offset correction from broadside Run1',acceptance_policy=POLICY,validated_for_cross_run_static_broadside=validated,input_sha256=hashes,limitations='Hardware plus fixed propagation path; single-session static broadside only; correlated packets and triangle edges are not independent evidence; null means unidentifiable')
    save(out/'calibration_run1_per_subcarrier.json',calibration)
    save(out/'validation_summary.json',dict(counts=counts,runs=summary,acceptance_policy=POLICY,validated_for_cross_run_static_broadside=validated,input_hashes_unchanged=True,self_checks_passed=True,metrics='Absolute wrapped circular mean residual of each run/state/pair/subcarrier; 18*30 values/run. Occurrence CSV preserves local statistics; R is invariant under fixed correction. Shared-pair difference is identical for raw/scalar/per-subcarrier.'))
    csvsave(out/'edge_residuals.csv',details); csvsave(out/'occurrence_statistics.csv',occurrence_rows); csvsave(out/'shared_pair_differences.csv',shared_rows)
    csvsave(out/'training_edges.csv',[dict(subcarrier_index=x[d['k']],**d) for d in diagnostics])
    print(json.dumps(clean(dict(counts=counts,results={r:{m:summary[r][m]['residual'] for m in ('scalar','per_subcarrier')} for r in (1,2,3)},shared={r:summary[r]['shared_adjacent'] for r in (1,2,3)},checks={r:summary[r]['acceptance_checks'] for r in (2,3)},validated_for_cross_run_static_broadside=validated)),indent=2))
if __name__=='__main__': main()
