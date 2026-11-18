"""Read cached evidence and bounded moments; no raw redecoding or angle tuning."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
import pcd_kernel as pcd
p=pcd.prior
norm=lambda x:np.linalg.norm(x,axis=(-2,-1))
tr=lambda x:np.trace(x,axis1=-2,axis2=-1).real
rows=[];maps=[]
for site in 'ABCDE':
    z=np.load(p.OLD/f'{site}_selected.npz')
    raw={s:z['switched'][:,:,j].astype(complex) for j,s in enumerate(p.ST)}
    raw['G2_control']=z['g2'].astype(complex)
    for source in pd.read_csv(p.OLD/f'{site}_sources.csv').itertuples():
        log=Path(source.path).parent/'events.log'
        ev=p.BASE.EVENT.findall(log.read_text(encoding='utf8'))
        saved=pd.read_csv(p.OLD/f'{site}_packets.csv')
        q=saved.query('run == @source.run')
        maps.append(dict(point=site,run=source.run,events_hash_matches=hashlib.sha256(log.read_bytes()).hexdigest()==source.events_sha256,
            logged_maps_match=all(list(map(int,x[2].split(',')))==p.BASE.MAP[x[1]] for x in ev),
            event_count=len(ev),ledger_min_visit_ms=q.visit_dt_ms.min(),ledger_max_visit_ms=q.visit_dt_ms.max(),
            ledger_all_states='/'.join(sorted(q.query('arm=="twelve"').state.unique())),
            selected_cache_hash_matches=hashlib.sha256((p.OLD/f'{site}_selected.npz').read_bytes()).hexdigest()==json.loads((p.OLD/'prediction_freeze.json').read_text())[f'{site}_selected.npz']))
    for s,x in raw.items():
        m=np.array([p.cov(x[r]) for r in range(3)])
        bg=p.background(m);B=bg['B'];ref=bg['ref'];refres=p.herm(B@ref@B)
        refpower=tr(refres);refeig=np.linalg.eigvalsh(refres)
        refraw=x[0,:8].reshape(-1,3,30)
        rawR1=np.einsum('nmk,nlk->kml',refraw,refraw.conj())/len(refraw)
        rawbase=float(norm(refraw).mean())
        for name,r,sl in [('R1held',0,slice(8,None)),('R2first',1,slice(0,12)),('R2last',1,slice(12,None)),('R3full',2,slice(None))]:
            target=m[r,sl].mean(0);projected=p.herm(B@target@B)
            delta=p.herm(projected-refres);ev=np.linalg.eigvalsh(delta)
            _,rawvec=np.linalg.eigh(np.einsum('nmk,nlk->kml',x[r,sl].reshape(-1,3,30),x[r,sl].reshape(-1,3,30).conj())/x[r,sl].reshape(-1,3,30).shape[0])
            _,normalvec=np.linalg.eigh(target)
            packetpower=np.sum(abs(x[r,sl])**2,axis=(-2,-1))
            refpacketpower=np.sum(abs(refraw)**2,axis=(-2,-1))
            _,_,meta=pcd.state_profile(bg,target,'G2' if s=='G2_control' else s)
            rows.append(dict(point=site,state=s,record=name,
                target_projected_fraction=float(np.sum(tr(projected))/np.sum(tr(target))),
                reference_residual_over_target_residual=float(np.sum(refpower)/np.sum(tr(projected))),
                tone_median_reference_residual_fraction=float(np.median(refpower/np.maximum(tr(projected),1e-15))),
                residual_R1_secondary_eigen_sum=float(np.sum(refeig[:,-2:])),
                projected_target_trace_sum=float(np.sum(tr(projected))),
                median_raw_packet_energy_over_R1=float(np.median(packetpower)/np.median(refpacketpower)),
                normalized_covariance_total_trace=float(np.sum(tr(target))),
                unnormalized_vs_normalized_principal_overlap=float(np.median(abs(np.sum(rawvec[:,:,-1].conj()*normalvec[:,:,-1],axis=1)))),
                median_PCD_eigen2_over_eigen1=float(np.median(np.maximum(ev[:,-2],0)/np.maximum(ev[:,-1],1e-15))),
                **meta))
pd.DataFrame(rows).to_csv(OUT/'projection_residual_and_normalization.csv',index=False)
pd.DataFrame(maps).to_csv(OUT/'cached_mapping_checks.csv',index=False)
print(pd.DataFrame(rows).query('point in ["B","C"] and record=="R2first"')[['point','state','target_projected_fraction','reference_residual_over_target_residual','median_raw_packet_energy_over_R1','unnormalized_vs_normalized_principal_overlap','median_PCD_eigen2_over_eigen1','valid_tones','q2']].round(4).to_string(index=False))
print('mapping checks',len(maps),all(r['events_hash_matches'] and r['logged_maps_match'] and r['selected_cache_hash_matches'] for r in maps))
