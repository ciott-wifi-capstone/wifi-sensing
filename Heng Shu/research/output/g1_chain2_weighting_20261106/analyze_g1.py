"""G1 audit and weighted subspace-fit MUSIC. Human coordinates absent by design."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'coherent_array_validation_20261106'
sys.path.insert(0,str(OLD))
import validate_coherent as c
SITES='ABCDE'
PAIRS=((0,1),(0,2),(1,2))
TRAIN_PACKETS=np.array([0,2,4]); HELD_PACKETS=np.array([1,3,5])

def save(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(ROOT/name,index=False,encoding='utf-8-sig');return d

def dump(name,v):
    (ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def pair_checks(x):
    """x = (..., six packets, three chains, thirty tones). No packet phase fit."""
    out=[]
    for i,j in PAIRS:
        z=x[...,i,:]*x[...,j,:].conj()
        ok=abs(z)>0
        u=np.divide(z,abs(z),out=np.zeros_like(z,dtype=complex),where=ok)
        train=u[...,TRAIN_PACKETS,:].sum(-2)
        trainok=ok[...,TRAIN_PACKETS,:].sum(-2)>=2
        # Undefined low-resultant references are failures, not a fabricated phase.
        trainok &= abs(train)>1e-8
        ref=np.divide(train,abs(train),out=np.zeros_like(train),where=abs(train)>0)
        err=np.angle(u[...,HELD_PACKETS,:]*ref[...,None,:].conj())
        valid=ok[...,HELD_PACKETS,:]&trainok[...,None,:]
        err=np.where(valid,err,np.nan)
        out.append((i,j,err,valid))
    return out

def provenance():
    frozen=json.loads((OLD/'prediction_freeze.json').read_text())
    source=[];ledgers=[];cache=[]
    for site in SITES:
        sf=OLD/f'{site}_selected.npz'
        assert c.base.sha(sf)==frozen[sf.name]
        z=np.load(sf);p=z['switched'];g=z['g2']
        ledger=pd.read_csv(OLD/f'{site}_packets.csv')
        assert (ledger.groupby(['run','arm']).size()==864).all()
        assert not ledger.duplicated(['run','arm','packet']).any()
        for row in pd.read_csv(OLD/f'{site}_sources.csv').itertuples():
            path=Path(row.path);log=path.parent/'events.log'
            assert c.base.sha(path)==row.sha256
            assert c.base.sha(log)==row.events_sha256
            x,host,_=c.base.decoder.decode(path)
            ta,perms=c.base.frame_details(path)
            run=json.loads((path.parent/'run.json').read_text(encoding='utf-8-sig'))
            starts,order,guard=c.base.decoder.align(log,run['start_ns']/1e9)
            ev=c.base.EVENT.findall(log.read_text(encoding='utf8'))
            assert all(list(map(int,e[2].split(',')))==c.base.MAP[e[1]] for e in ev)
            t=(host-run['start_ns'])/1e9
            for arm,target in [('twelve',p[row.run-1].reshape(-1,3,30)),('G2',g[row.run-1].reshape(-1,3,30))]:
                q=ledger[(ledger.run==row.run)&(ledger.arm==arm)]
                ids=q.packet.to_numpy(int)
                assert np.array_equal(x[ids],target)
                idx=np.searchsorted(starts,t[ids],side='right')-1
                dt=t[ids]-starts[idx]
                assert np.all(ta[ids]==c.base.APPARENT_TA)
                assert np.all(dt>=.003) and np.all(dt<.021)
                assert np.array_equal(np.array(c.ST)[order[idx]],q.state.to_numpy())
                assert guard<.003,(site,row.run,guard)
                ledgers.append(dict(point=site,run=row.run,arm=arm,packets=len(ids),
                    unique_packets=len(np.unique(ids)),cache_matches_raw=True,
                    visit_dt_min_ms=1000*dt.min(),visit_dt_max_ms=1000*dt.max(),clock_guard_ms=1000*guard))
            source.append(dict(point=site,run=row.run,source=str(path),sha256=row.sha256,
                events_sha256=row.events_sha256,frames=len(x),permutations=json.dumps(dict(perms)),
                physical_mapping_verified=True,cache_matches_raw=True))
        for name in (f'{site}_selected.npz',f'{site}_representatives.npz',f'{site}_assembled.npz',
                     f'{site}_complex_covariances.npz',f'{site}_packets.csv',f'{site}_sources.csv'):
            cache.append(dict(path=str(OLD/name),sha256=c.base.sha(OLD/name)))
        print(site,'raw, mapping and packet audit passed',flush=True)
    save('source_audit.csv',source);save('packet_budget_audit.csv',ledgers)
    dump('input_hashes.json',cache)

def diagnostic():
    power=[];tone=[];pairs=[];detail=[];weights=[];reference=[]
    for site in SITES:
        p=np.load(OLD/f'{site}_selected.npz')['switched'].astype(complex)
        for r in range(3):
            for si,s in enumerate(c.ST):
                x=p[r,:,si] # cycle, packet, chain, tone
                e=abs(x)**2;pw=e.mean(-1);share=pw/np.maximum(pw.sum(-1,keepdims=True),1e-15)
                for m in range(3):
                    v=x[:,:,m,:]
                    rel=10*np.log10(np.maximum(pw[:,:,m],1e-15)/np.maximum(np.delete(pw,m,axis=-1).mean(-1),1e-15))
                    power.append(dict(point=site,run=r+1,state=s,chain=m+1,antenna=c.base.MAP[s][m],
                        median_power_share=float(np.median(share[:,:,m])),
                        relative_other_chains_db=float(np.median(rel)),raw_amplitude_median=float(np.median(abs(v))),
                        exact_zero_fraction=float(np.mean(abs(v)==0)),
                        signed8_rail_fraction=float(np.mean((v.real==-128)|(v.real==127)|(v.imag==-128)|(v.imag==127))),
                        max_abs_realimag=float(max(abs(v.real).max(),abs(v.imag).max()))))
                checks=pair_checks(x)
                for i,j,err,valid in checks:
                    vals=abs(err[valid])*180/np.pi
                    pairs.append(dict(point=site,run=r+1,state=s,pair=f'{i+1}-{j+1}',
                        samples=len(vals),possible_samples=err.size,valid_fraction=valid.mean(),
                        phase_abs_median_deg=float(np.median(vals)),phase_abs_p90_deg=float(np.quantile(vals,.9)),
                        cosine_loss=float(np.mean(1-np.cos(err[valid])))))
                if s!='G1':continue
                ratio=10*np.log10(np.maximum(e[:,:,1],1e-15)/np.maximum((e[:,:,0]+e[:,:,2])/2,1e-15))
                errors=np.stack([checks[0][2],checks[2][2]])
                count=np.isfinite(errors).sum(0)
                combined=np.divide(np.nansum(abs(errors),axis=0),count,out=np.full(errors.shape[1:],np.nan),where=count>0)*180/np.pi
                for k in range(30):
                    tone.append(dict(point=site,run=r+1,tone=k,frequency_mhz=c.F[k]/1e6,
                        relative_chain2_db=float(np.median(ratio[:,:,k])),
                        chain2_amplitude_median=float(np.median(abs(x[:,:,1,k]))),
                        held_phase_p90_deg=float(np.nanquantile(combined[:,:,k],.9)),
                        held_phase_median_deg=float(np.nanmedian(combined[:,:,k]))))
                for cy in range(24):
                    for pi,original_pi in enumerate(HELD_PACKETS):
                        for k in range(30):
                            detail.append(dict(point=site,run=r+1,cycle=cy,packet_in_visit=int(original_pi),tone=k,
                                relative_chain2_db=ratio[cy,original_pi,k],
                                chain2_amplitude=abs(x[cy,original_pi,1,k]),
                                held_phase_error_deg=combined[cy,pi,k]))
                # A reference from even packets; common correction is trained only on strong chains and training tones.
                mask=np.zeros((3,30));mask[[0,2],:]=c.TRAIN
                for cy in range(24):
                    v=c.unit(x[cy]);ref=v[0]
                    even=[]
                    for pi in TRAIN_PACKETS:
                        t,ph,_,_=c.rat.fit(v[pi],ref,mask)
                        even.append(c.rat.apply(v[pi],t,ph))
                    ref=np.mean(even,axis=0)
                    for pi in HELD_PACKETS:
                        t,ph,_,edge=c.rat.fit(v[pi],ref,mask)
                        pred=c.rat.apply(v[pi],t,ph)
                        t0,ph0,_,_=c.rat.fit(v[pi],ref)
                        for m in range(3):
                            k=np.arange(30) if m==1 else np.where(~c.TRAIN)[0]
                            a=pred[m,k];b=ref[m,k];valid=(abs(a)>0)&(abs(b)>0)
                            normcorr=abs(np.vdot(a,b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-15)
                            phase=abs(np.angle(a[valid]*b[valid].conj()))*180/np.pi
                            reference.append(dict(point=site,run=r+1,cycle=cy,packet_in_visit=int(pi),chain=m+1,
                                held_tones=len(k),held_corr=normcorr,held_phase_p90_deg=np.quantile(phase,.9) if len(phase) else np.nan,
                                held_nrmse=np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15),fit_edge=edge,
                                all_vs_strong_delay_ns=abs(float(c.rat.wrap_delay(t0-t))),
                                all_vs_strong_phase_deg=abs(float(c.rat.wrap_phase(ph0-ph)))*180/np.pi))
        ch=pair_checks(p[0,:8,0])
        losses=[float(np.nanmean(1-np.cos(e))) for _,_,e,_ in ch]
        raw=losses[1]/max((losses[0]+losses[2])/2,1e-15)
        weights.append(dict(point=site,weight2=float(np.clip(raw,.05,1)),unclipped_ratio=raw,
            V12=losses[0],V13=losses[1],V23=losses[2],fit_cycles=8,fit_raw_packets=288,
            G1_fit_packets=48,phase_valid_fraction=min(v.mean() for _,_,_,v in ch)))
        print(site,'G1 diagnostics complete; R1 weight',weights[-1]['weight2'],flush=True)
    save('chain_power.csv',power);save('g1_tone_profiles.csv',tone);save('pair_phase_held.csv',pairs)
    save('g1_packet_tone_diagnostics.csv',detail);save('g1_strong_chain_prediction.csv',reference)
    save('frozen_R1_weights.csv',weights)
    dump('weight_freeze.json',{name:c.base.sha(ROOT/name) for name in ['protocol.json','analyze_g1.py','frozen_R1_weights.csv']})
    return {z['point']:z['weight2'] for z in weights}

def projector(v):
    # v[K,M,Q], Q=1 or 2. Reduced QR with an explicit rank check.
    q,r=np.linalg.qr(v,mode='reduced')
    assert np.min(abs(np.diagonal(r,axis1=-2,axis2=-1)))>1e-10
    return np.eye(v.shape[1])-q@q.conj().swapaxes(-1,-2)

def profile(ref,test,training,y,weight2=1.,antenna_ids=None):
    m=len(y);e,v=np.linalg.eigh(c.herm(ref));h=v[:,:,-1]
    B=np.eye(m)-h[:,:,None]*h[:,None,:].conj()
    vals,u=np.linalg.eigh(c.herm(B@test@B));u=u[:,:,-1]
    floor=np.linalg.eigvalsh(c.herm(B[None]@training@B[None]))[...,-1].max(0)
    use=(e[:,-1]/np.maximum(e.sum(1),1e-15)>=.7)&(vals[:,-1]>floor+1e-10)
    w=np.ones(m)
    if antenna_ids is None:antenna_ids=np.arange(1,m+1)
    w[np.asarray(antenna_ids)==2]=weight2
    d=np.sqrt(w)
    bg=projector((h*d[None,:])[:,:,None])
    null=projector(np.stack([h,u],axis=-1)*d[None,:,None])
    a=np.exp(2j*np.pi/c.base.C*c.F[:,None,None]*y[None,:,None]*np.sin(np.radians(c.ANG))[None,None,:])
    a*=d[None,:,None]
    num=np.maximum(np.einsum('kma,kmn,kna->ka',a.conj(),null,a).real,0)
    den=np.einsum('kma,kmn,kna->ka',a.conj(),bg,a).real
    profiles=[]
    for ids in (np.arange(30),np.arange(15),np.arange(15,30)):
        good=ids[use[ids]];cost=np.full(181,np.nan)
        if len(good)*2>=len(ids):
            ns=num[good].sum(0);ds=den[good].sum(0)
            np.divide(ns,ds,out=cost,where=ds>=.10*w.sum()*len(good))
        profiles.append(cost)
    # Keep the original unweighted residual-power gate, so weighting cannot quietly lower the empty noise floor.
    power=float(np.trace(B@test@B,axis1=1,axis2=2).real.mean()/(m-1))
    return np.array(profiles),power,int(use.sum())

def rebuild(site,weight):
    p=np.load(OLD/f'{site}_selected.npz')['switched'].astype(complex)
    reps=np.load(OLD/f'{site}_representatives.npz')['representatives'].copy()
    w=np.broadcast_to(np.array([1.,weight,1.])[:,None],(3,30))
    fits=[];vectors=[];closures=[]
    for r in range(3):
        row=[]
        for cy in range(24):
            x=c.unit(p[r,cy,0]);ref=x[0];aligned=[]
            for pi,v in enumerate(x):
                t,ph,_,edge=c.rat.fit(v,ref,w)
                pred=c.rat.apply(v,t,ph);aligned.append(pred)
                t0,ph0,_,_=c.rat.fit(v,ref)
                sw=np.sqrt(w)
                corr=abs(np.sum(pred.conj()*ref*w))/max(np.linalg.norm(pred*sw)*np.linalg.norm(ref*sw),1e-15)
                fits.append(dict(point=site,run=r+1,cycle=cy,packet=pi,weight2=weight,
                    common_delay_change_ns=abs(float(c.rat.wrap_delay(t-t0))),
                    common_phase_change_deg=abs(float(c.rat.wrap_phase(ph-ph0)))*180/np.pi,
                    weighted_corr=corr,edge=edge))
            a=np.array(aligned)
            reps[r,cy,0]=np.array([a.mean(0),a[:3].mean(0),a[3:].mean(0)])
            v,info,_=c.assemble(reps[r,cy,:,0],2)
            row.append(v);closures.append(dict(point=site,run=r+1,cycle=cy,**info))
        vectors.append(row)
    vectors=np.array(vectors)
    cov=np.array([c.cov_vectors(a) for a in vectors])
    save(f'{site}_weighted_common_fit.csv',fits);save(f'{site}_weighted_closure.csv',closures)
    np.savez_compressed(ROOT/f'{site}_weighted_covariance.npz',tree6=cov)
    return cov

def analyze(site,weight):
    raw=np.load(OLD/f'{site}_complex_covariances.npz')
    changed=rebuild(site,weight);rows=[];spectra={};stability=[]
    ids11=np.delete(np.arange(12),1)
    methods=[
        ('G2_original',raw['G2'],np.arange(4,7),1.),
        ('12_original',raw['tree6'],np.arange(1,13),1.),
        ('12_R1_weighted_scan',raw['tree6'],np.arange(1,13),weight),
        ('12_R1_weighted_fit_and_scan',changed,np.arange(1,13),weight),
        ('12_fixed_w2_0.5_scan',raw['tree6'],np.arange(1,13),.5),
        ('12_fixed_w2_0.25_scan',raw['tree6'],np.arange(1,13),.25),
        ('12_R1_weighted_fit_only',changed,np.arange(1,13),1.),
        ('11_drop_antenna2',raw['tree6'][...,ids11,:][...,:,ids11],ids11+1,1.)]
    for name,dat,ants,w in methods:
        y=c.base.Y[ants-1]
        assert dat.shape[-1]==len(y)
        for label,ix,rr in [('R1_held',np.arange(8,24),0),('R3',np.arange(24),2)]:
            ph,co=c.background_phase(dat[0,:8].mean(0),dat[rr,ix].mean(0))
            stability.append(dict(point=site,method=name,check=label,phase_pair_p90_deg=ph,
                                  eigenvector_corr=co,pass_reference=ph<=15 and co>=.9))
        for cal in ('raw_phase','R1_empirical'):
            z=c.calibration(dat[0,:8].mean(0),y)
            d=dat if cal=='raw_phase' else np.array([c.transform(v,z) for v in dat])
            ref=d[0,:8].mean(0);training=np.array([d[0,:4].mean(0),d[0,4:8].mean(0)])
            views={};powers={};used={}
            blocks={'predict':d[1,:12].mean(0),'held':d[1,12:].mean(0),'full':d[1].mean(0),
                    'empty':d[2].mean(0),'R1_held':d[0,8:].mean(0),
                    **{f'Q{i+1}':d[1,i*6:(i+1)*6].mean(0) for i in range(4)}}
            for block,v in blocks.items():
                pr,po,n=profile(ref,v,training,y,w,ants)
                views[block]=pr;powers[block]=po;used[block]=n
                spectra[f'{name}_{cal}_{block}']=pr
                if w==1:
                    old,oldpo,oldn=c.music_profile(ref,v,training,y)
                    assert np.array_equal(np.isnan(pr),np.isnan(old))
                    # Cached G2 covariances are complex64; QR re-orthogonalization differs by float32 roundoff.
                    assert not np.isfinite(pr).any() or np.nanmax(abs(pr-old))<2e-6,(site,name,cal,block,float(np.nanmax(abs(pr-old))))
                    assert c.peak(pr[0])==c.peak(old[0]) or not np.isfinite(pr).any()
                    assert abs(po-oldpo)<1e-8 and n==oldn
            angle=c.peak(views['predict'][0]);held=c.peak(views['held'][0]);full=c.peak(views['full'][0])
            halves=[c.peak(v) for v in views['held'][1:]];blank=[c.peak(v) for v in views['empty']]
            bgstable=bool(np.isfinite(blank).all() and np.max(abs(np.array(blank)-angle))<=9)
            temporal=bool(np.isfinite(held) and abs(angle-held)<=9)
            freq=bool(np.isfinite(halves).all() and np.max(abs(np.array(halves)-angle))<=9)
            ratio=powers['predict']/max(powers['empty'],1e-15)
            gate=bool(np.isfinite(angle) and abs(angle)<87 and temporal and freq and not bgstable and ratio>1)
            rows.append(dict(point=site,method=name,calibration=cal,elements=len(ants),weight2=w,
                prediction=angle,all_R2_peak=full,held_R2_peak=held,held_low=halves[0],held_high=halves[1],
                empty_peak=blank[0],foreground_empty_ratio=ratio,usable_prediction_tones=used['predict'],
                pass_time=temporal,pass_frequency=freq,empty_same_direction=bgstable,spectral_gate=gate,
                physical_calibration_verified=False,physical_release=False))
    save(f'{site}_reference_stability.csv',stability)
    np.savez_compressed(ROOT/f'{site}_spectra.npz',angles=c.ANG,**spectra)
    print(site,'fixed-method MUSIC comparison complete',flush=True)
    return rows

def synthetic():
    results={}
    for ants in (np.arange(4,7),np.arange(1,13),np.delete(np.arange(1,13),1)):
        y=c.base.Y[ants-1]
        a=lambda theta:np.exp(2j*np.pi/c.base.C*c.F[:,None]*y[None,:]*np.sin(np.radians(theta)))
        b=a(0);h=a(34);outer=lambda z:z[:,:,None]*z[:,None,:].conj()
        r0=outer(b)+.01*np.eye(len(y));r2=outer(b+.8j*h)+.01*np.eye(len(y))
        train=np.stack([r0,r0])
        for w in (1.,.5,.25,.05):
            pr,_,_=profile(r0,r2,train,y,w,ants)
            assert abs(c.peak(pr[0])-34)<=1
            if w==1:
                old,_,_=c.music_profile(r0,r2,train,y)
                assert np.nanmax(abs(old-pr))<1e-8
        results[str(len(y))+'_elements']='known synthetic 34 deg recovered at all weights'
    results['scope']='software geometry and metric consistency; not measured performance'
    dump('synthetic_checks.json',results)

def main():
    synthetic();provenance();weights=diagnostic()
    rows=[]
    for site in SITES:rows+=analyze(site,weights[site])
    save('blind_predictions.csv',rows)
    names=[p for p in ROOT.iterdir() if p.suffix in ('.json','.py','.csv','.npz') and p.name!='prediction_freeze.json']
    dump('prediction_freeze.json',{p.name:c.base.sha(p) for p in names})
    print('Predictions frozen; no surveyed human coordinates read by this script.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
