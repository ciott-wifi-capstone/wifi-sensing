"""Uniform, equal-packet noncoherent MUSIC. No human geometry in this runner."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'coherent_array_validation_20261106'
sys.path.insert(0,str(OLD))
import validate_coherent as old
BASE=old.base
ST=tuple(BASE.STATES);POINTS=tuple('ABCDE')
ANG=np.arange(-90.,91.); BANDS=('full','low','high');EYE=np.eye(3,dtype=complex)
ARMS={'G2_BG':('BG',False,('G2',)), 'SIX_BG':('BG',False,ST),
      'SIX_BG_R1':('BG',True,ST), 'G2_POS':('POS',False,('G2',)),
      'SIX_POS':('POS',False,ST), 'G2_SIGNED':('SIGNED',False,('G2',)),
      'SIX_SIGNED':('SIGNED',False,ST), 'G2_WBG':('WBG',False,('G2',)),
      'SIX_WBG':('WBG',False,ST), 'SIX_WBG_R1':('WBG',True,ST)}
STEER={s:np.exp(2j*np.pi/BASE.C*BASE.F[:,None,None]*BASE.Y[np.array(BASE.MAP[s])-1][None,:,None]*np.sin(np.radians(ANG))[None,None,:]) for s in ST}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding='utf-8-sig')
def dump(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def herm(a):return (a+a.conj().swapaxes(-1,-2))/2
def outer(v):return v[:,:,None]*v[:,None,:].conj()
def cov(x):
    """cycle x packets x chain x tone -> cycle x tone x chain x chain."""
    z=x*np.sqrt(90)/np.maximum(np.linalg.norm(x,axis=(-2,-1),keepdims=True),1e-15)
    return np.einsum('npmk,nplk->nkml',z,z.conj())/x.shape[1]

def input_audit():
    prior=json.loads((OLD/'prediction_freeze.json').read_text('utf8'))
    rows=[];budgets=[];cache={}
    for site in POINTS:
        for name in [f'{site}_selected.npz',f'{site}_packets.csv',f'{site}_sources.csv']:
            f=OLD/name;h=sha(f);assert h==prior[name],name;cache[str(f)]=h
        for x in pd.read_csv(OLD/f'{site}_sources.csv').itertuples():
            assert sha(x.path)==x.sha256
            assert sha(Path(x.path).parent/'events.log')==x.events_sha256
            rows.append(dict(point=site,run=x.run,path=x.path,sha256=x.sha256,events_sha256=x.events_sha256))
        p=pd.read_csv(OLD/f'{site}_packets.csv')
        for (run,arm),d in p.groupby(['run','arm']):
            assert len(d)==864 and d.packet.nunique()==864
            assert d.visit_dt_ms.min()>=3 and d.visit_dt_ms.max()<21
            if arm=='twelve':assert d.groupby('state').size().eq(144).all()
            fit=d[d['sample']<8];first=d[d['sample']<12]
            assert len(fit)==288 and len(first)==432
            budgets.append(dict(point=site,run=run,arm=arm,packets=864,unique=864,reference_first8=288,prediction_first12=432,held_last12=432,cycle_or_batch_count=24))
    save('source_manifest.csv',rows);save('packet_budget.csv',budgets);dump('input_cache_hashes.json',cache)

def background(m):
    ref=m[0,:8].mean(0)
    train=np.stack([m[0,:4].mean(0),m[0,4:8].mean(0)])
    blocks=np.stack([m[0,i:i+2].mean(0) for i in range(0,8,2)])
    e,v=np.linalg.eigh(herm(ref));h=v[:,:,-1];B=EYE-outer(h)
    spread=np.max(abs(np.linalg.eigvalsh(herm(train-ref))),axis=(0,2))
    floor=np.linalg.eigvalsh(herm(B[None]@train@B[None]))[...,-1].max(0)
    regular=.9*ref+.1*np.trace(ref,axis1=1,axis2=2).real[:,None,None]/3*EYE
    ew,vw=np.linalg.eigh(herm(regular));W=(vw/np.sqrt(np.maximum(ew,1e-12))[:,None,:])@vw.conj().transpose(0,2,1)
    rw=herm(W@ref@W);_,vh=np.linalg.eigh(rw);Bw=EYE-outer(vh[:,:,-1])
    fw=np.linalg.eigvalsh(herm(Bw[None]@W[None]@train@W[None]@Bw[None]))[...,-1].max(0)
    drift=float(np.median(np.max(np.linalg.norm(blocks-ref,axis=(-2,-1)),axis=0)/np.maximum(np.linalg.norm(ref,axis=(-2,-1)),1e-12)))
    return dict(ref=ref,train=train,e=e,h=h,B=B,spread=spread,floor=floor,W=W,Bw=Bw,fw=fw,drift=drift)

def reliability(raw,bg):
    x=raw[0,:8].reshape(-1,3,30)
    coherence=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        z=x[:,a]*x[:,b].conj();mag=abs(z);unit=np.divide(z,mag,out=np.zeros_like(z),where=mag>0)
        coherence.append(abs(unit.mean(0)))
    coh=np.asarray(coherence)
    quality=float(np.exp(np.log(np.clip(coh**2,.05,1)).mean()))
    weight=float(np.clip(quality/(1+bg['drift']),.05,1))
    return weight,dict(circular_quality=quality,covariance_drift=bg['drift'],weight=weight,min_pair_tone_coherence=float(coh.min()),mean_pair_tone_coherence=float(coh.mean()))

def state_profile(bg,target,s,kernel):
    ref=bg['ref'];a=STEER[s];dominance=bg['e'][:,-1]/np.maximum(bg['e'].sum(1),1e-12)
    if kernel in ['BG','WBG']:
        B=bg['B'] if kernel=='BG' else bg['Bw']
        tr=target if kernel=='BG' else bg['W']@target@bg['W']
        aa=a if kernel=='BG' else bg['W']@a
        vals,vec=np.linalg.eigh(herm(B@tr@B));floor=bg['floor'] if kernel=='BG' else bg['fw']
        use=(dominance>=.7)&(vals[:,-1]>floor+1e-10)&(vals[:,-1]-vals[:,-2]>1e-10)
        N=B-outer(vec[:,:,-1]);q=np.ones(30,dtype=int)
    else:
        vals,vec=np.linalg.eigh(herm(target-ref));aa=a;B=np.broadcast_to(EYE,ref.shape)
        if kernel=='POS':
            tol=1e-10*np.maximum(1,abs(vals).max(1));use=(vals[:,-1]>tol)&(vals[:,-1]-vals[:,-2]>tol)
            N=B-outer(vec[:,:,-1]);q=np.ones(30,dtype=int)
        else:
            selected=abs(vals)>np.maximum(bg['spread'][:,None],1e-10);q=selected.sum(1)
            U=(vec*selected[:,None,:])@vec.conj().transpose(0,2,1)
            use=(q>=1)&(q<=2);N=(B-U)*(3/np.maximum(3-q,1))[:,None,None]
    num=np.maximum(np.einsum('kma,kmn,kna->ka',aa.conj(),N,aa).real,0)
    den=np.einsum('kma,kmn,kna->ka',aa.conj(),B,aa).real
    original=np.sum(abs(aa)**2,axis=1)
    out=[]
    for ids in [np.arange(30),np.arange(15),np.arange(15,30)]:
        good=ids[use[ids]];valid=2*len(good)>=len(ids)
        if valid:
            n=num[good].sum(0);d=den[good].sum(0);energy=original[good].sum(0)
            cost=np.divide(n,d,out=np.full(181,np.nan),where=d>=.1*energy)
        else:n=np.zeros(181);d=np.zeros(181);energy=np.zeros(181);cost=np.full(181,np.nan)
        out.append(dict(num=n,den=d,energy=energy,cost=cost,valid=valid,tones=len(good)))
    raw_power=float(np.trace(bg['B']@target@bg['B'],axis1=1,axis2=2).real.mean()/2)
    return out,raw_power,dict(valid_tones=int(use.sum()),q0=int((q==0).sum()),q1=int((q==1).sum()),q2=int((q==2).sum()),q3=int((q==3).sum()))

def fuse(profiles,states,band,weights,balanced):
    ps=[profiles[s][band] for s in states]
    if any(not p['valid'] for p in ps):return np.full(181,np.nan)
    if balanced:
        cs=np.stack([p['cost'] for p in ps]);w=np.array([weights[s] for s in states]);w=w/w.sum()
        # No partial-state rescue: every included state must support the candidate.
        return np.sum(cs*w[:,None],axis=0)
    num=sum(p['num'] for p in ps);den=sum(p['den'] for p in ps);eng=sum(p['energy'] for p in ps)
    return np.divide(num,den,out=np.full(181,np.nan),where=den>=.1*eng)

def peaks(cost):
    if not np.isfinite(cost).any():return np.nan,[],np.nan
    finite=np.where(np.isfinite(cost))[0];k=int(finite[np.argmin(cost[finite])]);x=np.where(np.isfinite(cost),-cost,-np.inf)
    ix=list(find_peaks(x)[0])+[int(finite[0]),int(finite[-1]),k]
    ix=sorted(set(ix),key=lambda i:cost[i]);chosen=[]
    for i in ix:
        if all(abs(i-j)>=15 for j in chosen):chosen.append(i)
        if len(chosen)==3:break
    choices=[dict(angle=float(ANG[i]),cost=float(cost[i]),relative_db=float(10*np.log10(max(cost[i],1e-12)/max(cost[k],1e-12)))) for i in chosen]
    margin=choices[1]['relative_db'] if len(choices)>1 else float('inf')
    return float(ANG[k]),choices,margin

def reference_check(bg,target):
    _,v=np.linalg.eigh(herm(target));h=v[:,:,-1];ref=bg['h']
    errs=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        errs.extend(np.degrees(abs(np.angle(h[:,a]*h[:,b].conj()*ref[:,a].conj()*ref[:,b]))).tolist())
    corr=abs(np.sum(h.conj()*ref,axis=1))
    p90=float(np.quantile(errs,.9));rho=float(np.median(corr))
    return dict(phase_p90=p90,correlation=rho,pass_reference=bool(p90<=15 and rho>=.9))

def near(a,b):return bool(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=9)
def assess(view,power,run):
    first=f'R{run}_first';last=f'R{run}_last';full=f'R{run}_full'
    theta,alternatives,margin=peaks(view[first][0]);held=peaks(view[last][0])[0];whole=peaks(view[full][0])[0]
    bands=[peaks(view[b][k])[0] for b in [first,last] for k in [1,2]]
    quarter=[peaks(view[f'R{run}_Q{q}'][0])[0] for q in range(1,5)]
    empty=[peaks(view['R3_full'][b])[0] for b in range(3)]
    gate=dict(nonedge=bool(np.isfinite(theta) and abs(theta)<87),time=near(theta,held),frequency=all(near(theta,v) for v in bands),
      quarters=sum(near(theta,v) for v in quarter)>=3,unique=bool(np.isfinite(theta) and margin>=3),above_R1held=power[first]>power['R1_held'])
    base_ok=all(gate.values());r3similar=all(near(theta,v) for v in empty)
    pair_ok=(power[first]>power['R3_full'] and not r3similar) if run==2 else None
    result=dict(run=run,prediction=theta,held=held,whole=whole,margin_db=margin,alternatives=json.dumps(alternatives),
      first_low=bands[0],first_high=bands[1],held_low=bands[2],held_high=bands[3],quarter_angles=json.dumps(quarter),
      pre_return_candidate=bool(base_ok),paired_candidate=bool(base_ok and pair_ok) if run==2 else False,
      empty_peak=empty[0],R3_same_direction=r3similar,power_ratio_R1held=power[first]/max(power['R1_held'],1e-15),
      power_ratio_R3=power[first]/max(power['R3_full'],1e-15),physical_release=False,
      failure=';'.join(k for k,v in gate.items() if not v)+((';R3_control' if not pair_ok else '') if run==2 else ''))
    result.update({'pass_'+k:v for k,v in gate.items()})
    return result

def synthetic():
    # A known software scene distinct from field geometry, with coherent static paths.
    from copy import deepcopy
    outputs={}
    for s in ST:
        h=STEER[s][:,:,np.where(ANG==4)[0][0]];target=STEER[s][:,:,np.where(ANG==-28)[0][0]]
        ref=outer(h)+.02*EYE;rt=outer(h+(.68+.45j)*target)+.02*EYE
        m=np.broadcast_to(ref,(3,24,30,3,3)).copy();bg=background(m)
        for method in ['BG','WBG','SIGNED']:
            p,_,_=state_profile(bg,rt,s,method);val=p[0]['cost'][np.where(ANG==-28)[0][0]];assert val<1e-8,(s,method,val)
        # Identity whitening must reduce exactly to the existing BG formula.
        bgid=deepcopy(bg);bgid['W']=np.broadcast_to(EYE,ref.shape);bgid['Bw']=bgid['B'];bgid['fw']=bgid['floor']
        p,_,_=state_profile(bgid,rt,s,'WBG');p0,_,_=state_profile(bg,rt,s,'BG')
        assert np.allclose(p[0]['cost'],p0[0]['cost'],equal_nan=True,atol=1e-10)
    rng=np.random.default_rng(3);x=rng.normal(size=(4,6,3,30))+1j*rng.normal(size=(4,6,3,30))
    phase=np.exp(1j*rng.normal(size=(4,6,1,1))+1j*rng.normal(size=(4,6,1,1))*np.arange(30)[None,None,None,:])
    assert np.allclose(cov(x),cov(x*phase),atol=1e-12)
    # Invalid state cannot be rescued by silently shrinking the array.
    bad={s:[dict(valid=False)] for s in ST};assert not np.isfinite(fuse(bad,ST,0,dict.fromkeys(ST,1),False)).any()
    dump('software_checks.json',dict(known_source_in_MUSIC_null=True,whitening_identity_reproduces_BG=True,
      same_frequency_covariance_common_phase_slope_invariance=True,all_state_requirement=True,physical_validation=False))

def main():
    input_audit();synthetic();weights=[];moments={};raws={};backgrounds={};summary=[];support=[];refs=[];profiles_saved={};candidates=[];ablation=[]
    for site in POINTS:
        cache=np.load(OLD/f'{site}_selected.npz');p=cache['switched'].astype(complex);g=cache['g2'].astype(complex)
        assert p.shape==(3,24,6,6,3,30) and g.shape==(3,24,36,3,30)
        raw={s:p[:,:,j] for j,s in enumerate(ST)};raw['G2_control']=g
        mom={s:np.array([cov(x[r]) for r in range(3)]) for s,x in raw.items()}
        prev=np.load(OLD/f'{site}_complex_covariances.npz')['G2'];assert np.allclose(mom['G2_control'],prev,atol=1e-12)
        bg={s:background(x) for s,x in mom.items()}
        for s in raw:
            weight,info=reliability(raw[s],bg[s]);weights.append(dict(point=site,state=s,**info))
        moments[site]=mom;backgrounds[site]=bg
    save('R1_frozen_weights.csv',weights)
    dump('R1_weight_freeze.json',{n:sha(ROOT/n) for n in ['protocol.json','predict_uniform.py','R1_frozen_weights.csv']})
    for site in POINTS:
        mom=moments[site];bg=backgrounds[site];sw={x['state']:x['weight'] for x in weights if x['point']==site}
        target={}
        for s,d in mom.items():
            target[s]={'R1_held':d[0,8:].mean(0)}
            for r in [2,3]:
                target[s].update({f'R{r}_first':d[r-1,:12].mean(0),f'R{r}_last':d[r-1,12:].mean(0),f'R{r}_full':d[r-1].mean(0),**{f'R{r}_Q{q+1}':d[r-1,q*6:(q+1)*6].mean(0) for q in range(4)}})
            for label in ['R1_held','R3_full']:
                refs.append(dict(point=site,state=s,record=label,**reference_check(bg[s],target[s][label])))
        costs={};power={}
        for kernel in ['BG','POS','SIGNED','WBG']:
            costs[kernel]={};power[kernel]={}
            for block in target['G1']:
                costs[kernel][block]={};power[kernel][block]={}
                for s in mom:
                    actual='G2' if s=='G2_control' else s
                    v,p,meta=state_profile(bg[s],target[s][block],actual,kernel)
                    costs[kernel][block][s]=v;power[kernel][block][s]=p
                    support.append(dict(point=site,kernel=kernel,state=s,block=block,**meta))
                    if block in ['R2_first','R2_last','R2_full','R3_full']:
                        profiles_saved[f'{site}_state_{s}_{kernel}_{block}']=np.array([vv['cost'] for vv in v])
        for arm,(kernel,balanced,states) in ARMS.items():
            use=('G2_control',) if arm.startswith('G2_') else states
            view={block:np.array([fuse(cs,use,b,sw,balanced) for b in range(3)]) for block,cs in costs[kernel].items()}
            w=np.array([sw[s] if balanced else 1 for s in use]);w=w/w.sum()
            pp={block:float(np.dot(w,[sp[s] for s in use])) for block,sp in power[kernel].items()}
            for block,v in view.items():profiles_saved[f'{site}_{arm}_{block}']=v
            for r in [2,3]:summary.append(dict(point=site,arm=arm,kernel=kernel,states='/'.join(states),unique_antennas=3 if len(states)==1 else 12,**assess(view,pp,r)))
            for block in ['R2_first','R2_last','R2_full','R3_full']:
                for band in range(3):
                    peak,alts,margin=peaks(view[block][band])
                    for rank,v in enumerate(alts,1):candidates.append(dict(point=site,arm=arm,block=block,band=BANDS[band],rank=rank,**v))
            if arm=='SIX_WBG_R1':
                for drop in ST[:4]:
                    use=tuple(s for s in ST if s!=drop)
                    pred=peaks(fuse(costs[kernel]['R2_first'],use,0,sw,True))[0]
                    ablation.append(dict(point=site,removed_state=drop,prediction=pred,states='/'.join(use),not_independent=True))
            if arm=='G2_BG':
                # Exact baseline reproduction, including all three band costs.
                oldp=np.load(OLD/f'{site}_music_profiles.npz')
                for b,n in [('R2_first','predict'),('R2_last','held'),('R2_full','full')]:
                    assert np.allclose(view[b],oldp[f'{site}_G2_raw_phase_full_q1_{n}'],equal_nan=True,atol=1e-9),(site,b)
        print(site,'uniform prediction and empty-control checks complete',flush=True)
    save('blind_predictions.csv',summary);save('all_local_candidates.csv',candidates);save('tone_support.csv',support);save('reference_transfer.csv',refs);save('leave_G_state_out.csv',ablation)
    np.savez_compressed(ROOT/'spectra.npz',angles=ANG,**profiles_saved)
    files=['protocol.json','predict_uniform.py','R1_frozen_weights.csv','R1_weight_freeze.json','blind_predictions.csv','all_local_candidates.csv','tone_support.csv','reference_transfer.csv','leave_G_state_out.csv','spectra.npz','packet_budget.csv','source_manifest.csv','input_cache_hashes.json','software_checks.json']
    dump('prediction_freeze.json',{f:sha(ROOT/f) for f in files})
    print('Frozen truth-free outputs. No survey errors computed.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
