"""Bounded 12-element complex assembly audit; survey truth is deliberately absent."""
from pathlib import Path
import sys, json, importlib.util
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'music_subarray_gain_20261105'))
import predict_subarrays as prior
base=prior.base
spec=importlib.util.spec_from_file_location('ratnam_local',ROOT.parent/'ratnam_phase_timing_validation/core.py')
rat=importlib.util.module_from_spec(spec);spec.loader.exec_module(rat)
ST=base.STATES; F=base.F; FB=F-5.23e9; ANG=np.arange(-90.,91.)
TRAIN=np.arange(30)%3!=2
assert np.allclose(FB,rat.F)
NC=24; NP=6

def save(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding='utf-8-sig')

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=(-2,-1),keepdims=True),1e-15)

def select(n,k):
    assert n>=k
    ids=np.floor((np.arange(k)+.5)*n/k).astype(int)
    assert len(np.unique(ids))==k
    return ids

def extract(site,folder):
    cache=ROOT/f'{site}_selected.npz'
    if cache.exists():
        z=np.load(cache);return z['switched'],z['g2']
    packs=[];g2=[];ledger=[];sources=[];availability=[]
    for row in pd.read_csv(folder/'raw_runs.csv').itertuples():
        path=Path(row.source);assert base.sha(path)==row.sha256 and row.continuity_gate
        x,host,nic=base.decoder.decode(path)
        ta,_=base.frame_details(path)
        log=path.parent/'events.log'
        starts,order,_=base.decoder.align(log,row.start_ns/1e9)
        ev=base.EVENT.findall(log.read_text(encoding='utf8'))
        assert all(list(map(int,e[2].split(',')))==base.MAP[e[1]] for e in ev)
        t=(host-row.start_ns)/1e9
        idx=np.searchsorted(starts,t,side='right')-1
        vi=np.maximum(idx,0);dt=t-starts[vi]
        keep=(ta==base.APPARENT_TA)&(t>=0)&(t<10)&(idx>=0)&(dt>=.003)&(dt<.021)
        assert np.array_equal(order,np.arange(len(order))%6)
        groups={c:[np.where(keep&(idx==6*c+j))[0] for j in range(6)] for c in range(len(starts)//6)}
        good=[c for c,a in groups.items() if all(len(v)>=NP for v in a)]
        availability.append(dict(point=site,run=row.run,possible_cycles=len(groups),eligible_cycles=len(good),
                                 rejected_cycles=len(groups)-len(good),all_state_min_packets=NP))
        assert len(good)>=NC,(site,row.run,'inadequate complete cycles')
        cycles=np.array(good)[select(len(good),NC)]
        selected=[]
        for ci,c in enumerate(cycles):
            selected_states=[]
            span=[]
            for j,s in enumerate(ST):
                ids=groups[c][j][select(len(groups[c][j]),NP)]
                selected_states.append(x[ids]);span.extend(t[ids].tolist())
                for pi,k in enumerate(ids):
                    ledger.append(dict(point=site,run=row.run,arm='twelve',sample=ci,state=s,
                        cycle=int(c),packet=int(k),time_s=float(t[k]),visit_dt_ms=float(dt[k]*1000),
                        source_sha=row.sha256,part='R1_fit' if row.run==1 and ci<8 else 'held_or_foreground'))
            selected.append(selected_states)
            availability.append(dict(point=site,run=row.run,selected_cycle=int(c),sample=ci,
                                     within_cycle_span_ms=1000*(max(span)-min(span))))
        packs.append(selected)
        ids=np.where(keep&(order[vi]==1))[0]
        ids=ids[select(len(ids),NC*NP*6)]
        g2.append(x[ids].reshape(NC,NP*6,3,30))
        for pi,k in enumerate(ids):
            ledger.append(dict(point=site,run=row.run,arm='G2',sample=pi//36,state='G2',
                cycle=int(idx[k]//6),packet=int(k),time_s=float(t[k]),visit_dt_ms=float(dt[k]*1000),
                source_sha=row.sha256,part='R1_fit' if row.run==1 and pi<288 else 'held_or_foreground'))
        sources.append(dict(point=site,run=row.run,path=str(path),sha256=row.sha256,events_sha256=base.sha(log),frames=len(x)))
    p=np.asarray(packs);g=np.asarray(g2)
    assert p.shape==(3,24,6,6,3,30) and g.shape==(3,24,36,3,30)
    df=pd.DataFrame(ledger)
    assert (df.groupby(['run','arm']).size()==864).all()
    assert not df.duplicated(['run','arm','packet']).any()
    save(f'{site}_packets.csv',ledger);save(f'{site}_sources.csv',sources);save(f'{site}_cycles.csv',availability)
    np.savez_compressed(cache,switched=p,g2=g)
    print(site,'raw extraction complete',flush=True)
    return p,g

def residual(x,ref,mask=None):
    if mask is None:mask=np.ones_like(x.real)
    w=(x.conj()*ref*mask).sum(axis=0)
    denom=max(np.linalg.norm(x*mask)*np.linalg.norm(ref*mask),1e-15)
    v=w.sum();corr=abs(v)/denom
    opt=minimize_scalar(lambda z:-abs(np.sum(w*np.exp(-2j*np.pi*FB*z*1e-9)))**2,
                        bounds=(-10,10),method='bounded')
    delay=opt.x
    phase=np.angle(np.sum(w*np.exp(-2j*np.pi*FB*delay*1e-9)))
    amp=np.linalg.norm(x*mask)/max(np.linalg.norm(ref*mask),1e-15)
    return float(corr),float(abs(phase)*180/np.pi),float(abs(delay)),float(amp)

def representatives(p,site):
    reps=np.empty((3,24,6,3,3,30),complex) # run,cycle,state,all/early/late,chain,tone
    stats=[]
    cmask=np.zeros((3,30));cmask[:2,TRAIN]=1;held=1-cmask
    for r in range(3):
        for cy in range(24):
            for si,s in enumerate(ST):
                x=unit(p[r,cy,si].astype(complex));ref=x[0]
                aligned=[]
                for pi,v in enumerate(x):
                    tau,phase,corr,edge=rat.fit(v,ref)
                    u=rat.apply(v,tau,phase);aligned.append(u)
                    if pi in (1,3,5):
                        tc,pc,_,_=rat.fit(v,ref,cmask)
                        rh=residual(rat.apply(v,tc,pc),ref,held)
                        independent=[rat.fit(v[m:m+1],ref[m:m+1]) for m in range(3)]
                        ds=np.array([a[0] for a in independent]);ps=np.array([a[1] for a in independent])
                        dd=float(np.max(abs(rat.wrap_delay(ds-ds[0]))))
                        pp=float(np.max(abs(rat.wrap_phase(ps-ps[0]))))*180/np.pi
                        stats.append(dict(point=site,run=r+1,sample=cy,state=s,packet_in_visit=pi,
                            held_corr=rh[0],held_phase_deg=rh[1],held_delay_ns=rh[2],chain_delay_spread_ns=dd,
                            chain_phase_spread_deg=pp,pass_common=rh[0]>=.90 and rh[1]<=15 and rh[2]<=5 and dd<=5 and pp<=15 and not edge))
                u=np.asarray(aligned)
                reps[r,cy,si]=np.array([u.mean(0),u[:3].mean(0),u[3:].mean(0)])
        print(site,'packet reference checks run',r+1,flush=True)
    save(f'{site}_packet_reference.csv',stats)
    return reps

def edge_fit(x,ref):
    mask=np.broadcast_to(TRAIN,x.shape)
    tau,phase,corr,edge=rat.fit(x,ref,mask)
    rot=np.exp(1j*(2*np.pi*FB*tau*1e-9+phase))
    pred=x*rot
    gain=float(np.real(np.sum(pred.conj()*ref*mask))/max(np.sum(abs(pred*mask)**2),1e-15))
    gain=max(gain,1e-12)
    rh=residual(pred*gain,ref,1-mask)
    return rot*gain,dict(delay_ns=tau,phase_deg=phase*180/np.pi,gain=gain,fit_corr=corr,
        held_corr=rh[0],held_phase_deg=rh[1],held_delay_ns=rh[2],held_amplitude_ratio=rh[3],
        pass_edge=rh[0]>=.9 and rh[1]<=15 and rh[2]<=5 and .25<=gain<=4 and not edge)

def assemble(rep,bridge):
    # rot child*parent = common complex reference. A shared antenna has the same chain index.
    edges=[(0,4,0),(1,4,2),(2,4,1),(5,4,bridge),(3,5,0)]
    rot={4:np.ones(30,complex)};audit=[]
    for child,parent,m in edges:
        corr,info=edge_fit(rep[child,m:m+1],rep[parent,m:m+1])
        rot[child]=corr*rot[parent]
        audit.append(dict(child=ST[child],parent=ST[parent],shared_antenna=base.MAP[ST[child]][m],**info))
    synced=np.array([rep[j]*rot[j] for j in range(6)])
    other=1 if bridge==2 else 2
    closure=residual(synced[5,other:other+1],synced[4,other:other+1])
    phase_samples=np.abs(np.angle(synced[5,other]*synced[4,other].conj()))*180/np.pi
    closure_p90=float(np.quantile(phase_samples,.9))
    vec=np.concatenate(synced[:4],axis=0)
    info=dict(closure_antenna=base.MAP['C1'][other],closure_corr=closure[0],closure_phase_deg=closure[1],
        closure_delay_ns=closure[2],closure_phase_tone_p90_deg=closure_p90,closure_gain=closure[3],
        all_tree_edges_pass=all(a['pass_edge'] for a in audit),
        pass_closure=closure[0]>=.9 and closure[1]<=15 and closure[2]<=5 and closure_p90<=15 and .8<=closure[3]<=1.25)
    return vec,info,audit

def cov_vectors(x):
    x=unit(x)*np.sqrt(x.shape[-2]*30)
    return np.einsum('nmk,nlk->nkml',x,x.conj())

def g2_cov(x):
    x=unit(x)*np.sqrt(90)
    return np.einsum('npmk,nplk->nkml',x,x.conj())/x.shape[1]

def herm(x):return (x+x.conj().swapaxes(-1,-2))/2

def calibration(ref,y):
    _,v=np.linalg.eigh(herm(ref))
    h=v[:,:,-1]
    path=np.sqrt(3.65**2+y*y)-3.65
    los=np.exp(-2j*np.pi*F[:,None]*path[None,:]/base.C)
    z=los*np.exp(-1j*np.angle(h))
    return z

def transform(c,z):return c*z[None,:,:,None]*z.conj()[None,:,None,:]

def fbss(c,L=8):
    m=c.shape[-1]
    out=sum(c[...,i:i+L,i:i+L] for i in range(m-L+1))/(m-L+1)
    return (out+out[...,::-1,::-1].conj())/2

def music_profile(ref,test,training,y,q=1):
    m=len(y);eye=np.eye(m)
    e,u=np.linalg.eigh(herm(ref));h=u[:,:,-1]
    B=eye-h[:,:,None]*h[:,None,:].conj()
    vals,v=np.linalg.eigh(herm(B@test@B))
    U=v[:,:,-q:];N=B-U@U.conj().swapaxes(-1,-2)
    floor=np.max(np.linalg.eigvalsh(herm(B[None]@training@B[None]))[...,-1],axis=0)
    use=(e[:,-1]/np.maximum(e.sum(1),1e-15)>=.7)&(vals[:,-q]>floor+1e-10)
    a=np.exp(2j*np.pi/base.C*F[:,None,None]*y[None,:,None]*np.sin(np.radians(ANG))[None,None,:])
    num=np.maximum(np.einsum('kma,kmn,kna->ka',a.conj(),N,a).real,0)
    den=np.einsum('kma,kmn,kna->ka',a.conj(),B,a).real
    out=[]
    for ids in (np.arange(30),np.arange(15),np.arange(15,30)):
        good=ids[use[ids]]
        cost=np.full(len(ANG),np.nan)
        if len(good)*2>=len(ids):
            ns=num[good].sum(0);ds=den[good].sum(0)
            np.divide(ns,ds,out=cost,where=ds>=.10*m*len(good))
        out.append(cost)
    return np.array(out),float(np.trace(B@test@B,axis1=1,axis2=2).real.mean()/(m-1)),int(use.sum())

def peak(cost):
    return float(ANG[np.nanargmin(cost)]) if np.isfinite(cost).any() else np.nan

def background_phase(ref,test):
    _,u=np.linalg.eigh(herm(ref));_,v=np.linalg.eigh(herm(test))
    a=u[:,:,-1];b=v[:,:,-1]
    # all pair phase changes, invariant to each frequency's global eigenvector gauge
    phase=[]
    for i in range(a.shape[1]):
        for j in range(i):
            phase.extend(abs(np.angle((b[:,i]*b[:,j].conj())*(a[:,i]*a[:,j].conj()).conj()))*180/np.pi)
    return float(np.quantile(phase,.90)),float(np.mean(abs(np.sum(a.conj()*b,axis=1))))

def analyze_music(site,assembled,graw):
    spectra={};rows=[];stability=[]
    # All complex arrays kept even when synchronization gates fail, explicitly diagnostic.
    raw12={tree:np.array([cov_vectors(assembled[r,:,tree,0]) for r in range(3)]) for tree in range(2)}
    rawg=np.array([g2_cov(graw[r]) for r in range(3)])
    for tree,raw in [('G2',rawg),('tree6',raw12[0]),('tree8',raw12[1])]:
        y=base.Y[3:6] if tree=='G2' else base.Y
        ref=raw[0,:8].mean(0)
        for r,ix in [(1,np.arange(8,24)),(3,np.arange(24))]:
            ph,corr=background_phase(ref,raw[r-1,ix].mean(0))
            stability.append(dict(point=site,arm=tree,empty_run=r,phase_pair_p90_deg=ph,
                eigenvector_corr=corr,pass_reference=ph<=15 and corr>=.9))
        z=calibration(ref,y)
        for cal in ('raw_phase','R1_empirical'):
            data=raw if cal=='raw_phase' else np.array([transform(c,z) for c in raw])
            for smooth in (False,True) if tree!='G2' else (False,):
                d=fbss(data) if smooth else data
                yy=y[:8] if smooth else y
                ref=d[0,:8].mean(0);training=np.array([d[0,:4].mean(0),d[0,4:8].mean(0)])
                for q in (1,2) if tree!='G2' else (1,):
                    name=f'{tree}_{cal}_{"fbss8" if smooth else "full"}_q{q}'
                    blockdata={
                        'predict':d[1,:12].mean(0),'held':d[1,12:].mean(0),'full':d[1].mean(0),
                        'empty':d[2].mean(0),'empty_held_R1':d[0,8:].mean(0),
                        **{f'Q{k+1}':d[1,k*6:(k+1)*6].mean(0) for k in range(4)}}
                    views={};power={};usable={}
                    for block,c in blockdata.items():
                        v,p,n=music_profile(ref,c,training,yy,q)
                        views[block]=v;power[block]=p;usable[block]=n
                        spectra[f'{site}_{name}_{block}']=v
                    angle=peak(views['predict'][0]);held=peak(views['held'][0]);full=peak(views['full'][0])
                    halves=[peak(v) for v in views['held'][1:]]
                    blank=[peak(v) for v in views['empty']]
                    bgstable=np.isfinite(blank).all() and max(abs(np.array(blank)-angle))<=9
                    temporal=np.isfinite(held) and abs(angle-held)<=9
                    freq=np.isfinite(halves).all() and max(abs(np.array(halves)-angle))<=9
                    ratio=power['predict']/max(power['empty'],1e-15)
                    statistical=(np.isfinite(angle) and abs(angle)<87 and temporal and freq and not bgstable and ratio>1)
                    rows.append(dict(point=site,arm=tree,calibration=cal,smoothing='FBSS8' if smooth else 'none',
                        rank=q,prediction=angle,all_R2_peak=full,held_R2_peak=held,held_low=halves[0],held_high=halves[1],
                        empty_peak=blank[0],foreground_empty_ratio=ratio,usable_prediction_tones=usable['predict'],
                        pass_time=temporal,pass_frequency=freq,empty_same_direction=bgstable,
                        spectral_gate=bool(statistical),physical_calibration_verified=False,physical_release=False))
    save(f'{site}_music_predictions.csv',rows);save(f'{site}_reference_stability.csv',stability)
    np.savez_compressed(ROOT/f'{site}_music_profiles.npz',angle=ANG,**spectra)
    np.savez_compressed(ROOT/f'{site}_complex_covariances.npz',G2=rawg,tree6=raw12[0],tree8=raw12[1])
    return rows

def synthetic():
    rng=np.random.default_rng(807)
    # Exact single common phase+slope and a distinguishable held-chain corruption.
    h=rng.normal(size=(3,30))+1j*rng.normal(size=(3,30))
    v=h*np.exp(-1j*(2*np.pi*FB*47.3e-9+.82))
    t,p,_,_=rat.fit(v,h);assert np.linalg.norm(rat.apply(v,t,p)-h)/np.linalg.norm(h)<1e-4
    cm=np.zeros((3,30));cm[:2,TRAIN]=1
    bad=v.copy();bad[2]*=np.exp(1j*.9)
    t,p,_,_=rat.fit(bad,h,cm)
    assert residual(rat.apply(bad,t,p),h,1-cm)[1]>15
    # Recover 12 true complex entries modulo a common frequency-dependent scalar.
    true=rng.normal(size=(12,30))+1j*rng.normal(size=(12,30))
    reps=[]
    for s in ST:
        ph=rng.uniform(-np.pi,np.pi);tau=rng.uniform(-80,80)
        reps.append(true[np.array(base.MAP[s])-1]*np.exp(-1j*(ph+2*np.pi*FB*tau*1e-9))*rng.uniform(.8,1.2))
    recovered,info,_=assemble(np.array(reps),2)
    ref=true[np.array(base.MAP['C1'])-1]
    ratio=np.sum(recovered.conj()*true,axis=0)/np.maximum(np.sum(abs(recovered)**2,axis=0),1e-15)
    err=np.linalg.norm(recovered*ratio-true)/np.linalg.norm(true)
    assert err<1e-4 and info['pass_closure'],(err,info)
    # Intentionally perturb held antenna8 in C2; tree6 fitting cannot hide it.
    reps[5][1]*=np.exp(1j*.8)
    _,bad,_=assemble(np.array(reps),2)
    assert not bad['pass_closure']
    # Background-projected MUSIC uses the same projected manifold as its covariance.
    for ids in (np.arange(12),np.arange(3,6)):
        y=base.Y[ids]
        def a(theta):return np.exp(2j*np.pi/base.C*F[:,None]*y[None,:]*np.sin(np.radians(theta)))
        b=a(0);h=a(34)
        outer=lambda v:v[:,:,None]*v[:,None,:].conj()
        r0=outer(b)+.01*np.eye(len(ids));r2=outer(b+.8j*h)+.01*np.eye(len(ids))
        profile,_,_=music_profile(r0,r2,np.stack([r0,r0]),y)
        assert abs(peak(profile[0])-34)<=1,peak(profile[0])
    (ROOT/'synthetic_checks.json').write_text(json.dumps(dict(common_phase_delay=True,held_chain_failure_detected=True,
        complex_12_recovery_relative_error=err,held_shared_antenna_corruption_detected=True,projected_music_exact_angle=True,
        scope='software consistency only, not measured human accuracy'),indent=2),encoding='utf8')

def main():
    synthetic()
    all_predictions=[]
    for site in 'ABCDE':
        p,g=extract(site,prior.previous.INPUTS[site])
        repfile=ROOT/f'{site}_representatives.npz'
        if repfile.exists():rep=np.load(repfile)['representatives']
        else:
            rep=representatives(p,site);np.savez_compressed(repfile,representatives=rep)
        assembled=np.empty((3,24,2,3,12,30),complex);closures=[];edges=[]
        for r in range(3):
            for c in range(24):
                for tr,bridge in enumerate((2,1)):
                    for part in range(3):
                        v,info,ea=assemble(rep[r,c,:,part],bridge)
                        assembled[r,c,tr,part]=v
                        key=dict(point=site,run=r+1,sample=c,tree='tree6' if tr==0 else 'tree8',part=('all','early','late')[part])
                        closures.append(dict(**key,**info));edges.extend(dict(**key,**e) for e in ea)
        save(f'{site}_closure.csv',closures);save(f'{site}_edge_checks.csv',edges)
        np.savez_compressed(ROOT/f'{site}_assembled.npz',snapshots=assembled)
        all_predictions.extend(analyze_music(site,assembled,g))
        print(site,'12-element MUSIC diagnostic complete',flush=True)
    save('blind_music_predictions.csv',all_predictions)
    files=[p for p in ROOT.iterdir() if p.suffix in ('.csv','.npz','.py','.json') and p.name!='prediction_freeze.json']
    (ROOT/'prediction_freeze.json').write_text(json.dumps({p.name:base.sha(p) for p in files},indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
