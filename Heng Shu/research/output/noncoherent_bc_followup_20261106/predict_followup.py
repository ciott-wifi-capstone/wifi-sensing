"""One predeclared follow-up: same packet caches, no survey truth in prediction."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import pcd_kernel as pcd
old=pcd.prior
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'noncoherent_uniform_20261106'
ST=old.ST
ARMS={'G2_PCD':('PCD','pool',('G2_control',)),
      'SIX_PCD_POOL':('PCD','pool',ST),
      'SIX_PCD_BAL':('PCD','balanced',ST),
      'SIX_PCD_R1':('PCD','weighted',ST),
      'SIX_SIGNED_BAL':('SIGNED','balanced',ST),
      'SIX_SIGNED_R1':('SIGNED','weighted',ST)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,rows):pd.DataFrame(rows).to_csv(ROOT/n,index=False,encoding='utf-8-sig')
def dump(n,obj):(ROOT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def fuse(profiles,states,band,weights,mode):
    p=[profiles[s][band] for s in states]
    if any(not x['valid'] for x in p):return np.full(181,np.nan)
    # Same candidate eligibility for all new aggregation rules.
    allowed=np.all(np.stack([np.isfinite(x['cost']) for x in p]),axis=0)
    if mode=='pool':
        n=sum(x['num'] for x in p);d=sum(x['den'] for x in p)
        c=np.divide(n,d,out=np.full(181,np.nan),where=d>0)
    else:
        w=np.array([weights[s] if mode=='weighted' else 1. for s in states]);w/=w.sum()
        c=np.sum(np.stack([x['cost'] for x in p])*w[:,None],axis=0)
    return np.where(allowed,c,np.nan)

def audit():
    frozen=json.loads((PREV/'prediction_freeze.json').read_text('utf8'))
    assert all(sha(PREV/n)==h for n,h in frozen.items())
    cache=json.loads((PREV/'input_cache_hashes.json').read_text('utf8'))
    assert all(sha(n)==h for n,h in cache.items())
    ledger=[]
    for site in 'ABCDE':
        p=pd.read_csv(old.OLD/f'{site}_packets.csv')
        for (run,arm),d in p.groupby(['run','arm']):
            first=d[d['sample']<12];last=d[d['sample']>=12]
            assert len(d)==d.packet.nunique()==864
            assert len(first)==len(last)==432
            assert len(d[d['sample']<8])==288
            assert first.time_s.max()<last.time_s.min()
            if arm=='twelve':assert d.groupby('state').size().eq(144).all()
            ledger.append(dict(point=site,run=run,arm=arm,packets=864,first=432,held=432,reference_fit=288,chronological_disjoint=True))
    save('packet_budget.csv',ledger)
    dump('input_hashes.json',{'prior_frozen':frozen,'selected_cache':cache})

def math_checks():
    checks=pcd.mathematical_checks()
    p={s:[dict(valid=True,cost=np.full(181,.3),num=np.full(181,.9),den=np.full(181,3.))] for s in ST}
    for mode in ['pool','balanced','weighted']:
        assert np.allclose(fuse(p,ST,0,dict.fromkeys(ST,1.),mode),.3)
        p['C2'][0]['cost'][50]=np.nan
        assert np.isnan(fuse(p,ST,0,dict.fromkeys(ST,1.),mode)[50])
        p['C2'][0]['cost'][50]=.3
    checks.update(common_eligibility_all_new_fusions=True,constant_cost_fusion_identity=True)
    dump('mathematical_checks.json',checks)

def main():
    audit();math_checks()
    dump('pre_execution_freeze.json',{n:sha(ROOT/n) for n in ['protocol.json','predict_followup.py','pcd_kernel.py']})
    weights=pd.read_csv(PREV/'R1_frozen_weights.csv')
    rows=[];support=[];state_rows=[];spectra={};candidates=[];retention=[]
    for site in 'ABCDE':
        cache=np.load(old.OLD/f'{site}_selected.npz')
        raw={s:cache['switched'][:,:,j].astype(complex) for j,s in enumerate(ST)}
        raw['G2_control']=cache['g2'].astype(complex)
        mom={s:np.stack([old.cov(x[r]) for r in range(3)]) for s,x in raw.items()}
        bg={s:old.background(m) for s,m in mom.items()}
        w=weights[weights.point==site].set_index('state').weight.to_dict()
        for s in raw:
            computed,_=old.reliability(raw[s],bg[s]);assert np.isclose(computed,w[s],atol=1e-12)
        target={}
        for s,m in mom.items():
            target[s]={'R1_held':m[0,8:].mean(0)}
            for r in [2,3]:
                target[s].update({f'R{r}_first':m[r-1,:12].mean(0),f'R{r}_last':m[r-1,12:].mean(0),f'R{r}_full':m[r-1].mean(0),**{f'R{r}_Q{q+1}':m[r-1,q*6:(q+1)*6].mean(0) for q in range(4)}})
        costs={};powers={}
        for kernel in ['PCD','SIGNED']:
            costs[kernel]={};powers[kernel]={}
            for block in target['G1']:
                costs[kernel][block]={};powers[kernel][block]={}
                for s in mom:
                    actual='G2' if s=='G2_control' else s
                    if kernel=='PCD':p,power,meta=pcd.state_profile(bg[s],target[s][block],actual)
                    else:p,power,meta=old.state_profile(bg[s],target[s][block],actual,'SIGNED')
                    costs[kernel][block][s]=p;powers[kernel][block][s]=power
                    support.append(dict(point=site,kernel=kernel,state=s,block=block,**meta))
                    if block in ['R2_first','R2_last','R2_full','R3_full']:
                        spectra[f'{site}_state_{s}_{kernel}_{block}']=np.stack([x['cost'] for x in p])
                        for band in range(3):
                            angle,alts,margin=old.peaks(p[band]['cost'])
                            state_rows.append(dict(point=site,state=s,kernel=kernel,block=block,band=old.BANDS[band],prediction=angle,margin_db=margin,valid_tones=p[band]['tones'],candidates=json.dumps(alts)))
                    if kernel=='PCD' and block=='R2_first':
                        for band in range(3):
                            z=p[band];fraction=np.divide(z['den'],z['energy'],out=np.full(181,np.nan),where=z['energy']>0)
                            for i,a in enumerate(old.ANG):retention.append(dict(point=site,state=s,band=old.BANDS[band],angle=a,projected_fraction=fraction[i],eligible=np.isfinite(z['cost'][i])))
        for arm,(kernel,mode,states) in ARMS.items():
            view={block:np.stack([fuse(c,states,b,w,mode) for b in range(3)]) for block,c in costs[kernel].items()}
            ww=np.array([w[s] if mode=='weighted' else 1 for s in states]);ww=ww/ww.sum()
            pp={block:float(np.dot(ww,[v[s] for s in states])) for block,v in powers[kernel].items()}
            for block,v in view.items():spectra[f'{site}_{arm}_{block}']=v
            for r in [2,3]:rows.append(dict(point=site,arm=arm,kernel=kernel,mode=mode,states='/'.join(states),**old.assess(view,pp,r)))
            for block in ['R2_first','R2_last','R2_full','R3_full']:
                for b in range(3):
                    _,alts,_=old.peaks(view[block][b])
                    for rank,v in enumerate(alts,1):candidates.append(dict(point=site,arm=arm,block=block,band=old.BANDS[b],rank=rank,**v))
        # Baseline reproduction is numerical, not an untested file-copy claim.
        oldspec=np.load(PREV/'spectra.npz')
        for kernel,arm,balanced in [('BG','G2_BG',False),('SIGNED','G2_SIGNED',False),('WBG','SIX_WBG_R1',True)]:
            states=('G2_control',) if arm.startswith('G2') else ST
            for block in ['R2_first','R2_last','R2_full','R3_full']:
                c={s:old.state_profile(bg[s],target[s][block],'G2' if s=='G2_control' else s,kernel)[0] for s in states}
                v=np.stack([old.fuse(c,states,b,w,balanced) for b in range(3)])
                assert np.allclose(v,oldspec[f'{site}_{arm}_{block}'],equal_nan=True,atol=1e-10),(site,arm,block)
        print(site,'six fixed new arms complete; baseline spectra reproduced',flush=True)
    save('blind_predictions.csv',rows);save('all_candidates.csv',candidates);save('tone_support.csv',support);save('state_diagnostics.csv',state_rows);save('projection_retention.csv',retention)
    np.savez_compressed(ROOT/'spectra.npz',angles=old.ANG,**spectra)
    files=['protocol.json','predict_followup.py','pcd_kernel.py','pre_execution_freeze.json','input_hashes.json','packet_budget.csv','mathematical_checks.json','blind_predictions.csv','all_candidates.csv','tone_support.csv','state_diagnostics.csv','projection_retention.csv','spectra.npz']
    dump('prediction_freeze.json',{n:sha(ROOT/n) for n in files})
    print('Predictions frozen before survey scoring. No physical release.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
