from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R=Path(__file__).resolve().parent;O=R.parent
S=O/'same_frequency_spatial_background_20261104';A=O/'e1_e3_mapping_phase_timing_audit_20261104'
sys.path.insert(0,str(S));import spatial_model as sm
sys.path.insert(0,str(O/'multi_peak_branch_validation_20261105'));from branch_analysis import region
ST=list(sm.MAP);GS=ST[:4];FIRST=[0,12,21,30];PA=[(0,1),(0,2),(1,2)]
KT=np.arange(30)%3!=2;KV=~KT;C=299792458.;Y=np.array(sm.GEO['array_y_local_m'])
MODES=['identity','rf_constant','antenna_constant','antenna_tone']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(rows,name,cols=None):
    x=pd.DataFrame(rows,columns=cols);x.to_csv(R/name,index=False);return x
def freeze(name,paths):
    (R/name).write_text(json.dumps({str(Path(p).resolve()):sha(p) for p in paths},indent=2),encoding='utf8')
def verify(p):
    for path,h in json.loads(Path(p).read_text()).items():assert sha(path)==h
def unit(z):return z/np.maximum(abs(z),1e-30)
def los(s):
    yy=Y[np.array(sm.MAP[s])-1];d=np.sqrt(3.65**2+yy**2)
    return np.exp(-2j*np.pi*d[:,None]*sm.F[None,:]/C)
def pairval(x):
    z=sm.spatial(x);dg=np.real(np.diagonal(z,axis1=1,axis2=2))
    q=np.array([z[:,j,i]/np.sqrt(np.maximum(dg[:,i]*dg[:,j],1e-30)) for i,j in PA])
    return q,z
def qp(h):return np.array([h[j]*h[i].conj() for i,j in PA])
def phasecheck(x,h,mask):
    q,z=pairval(x);rel=abs(q[:,mask])>=.5;e=np.degrees(abs(np.angle(q[:,mask]*qp(h)[:,mask].conj())))
    v=e[rel];support=rel.mean();med=float(np.median(v)) if len(v) else np.nan;p90=float(np.quantile(v,.9)) if len(v) else np.nan
    return dict(phase_median_deg=med,phase_p90_deg=p90,reliable_fraction=float(support),
                phase_prediction_gate=bool(support>=.5 and med<=15 and p90<=30))

def fitgraph(edges,obs,weight,n):
    edges=np.asarray(edges);obs=unit(np.asarray(obs));w=np.maximum(np.asarray(weight),1e-6)
    H=np.zeros((n,n),complex);deg=np.zeros(n)
    for (i,j),a,b in zip(edges,obs,w):H[j,i]+=b*a;H[i,j]+=b*a.conjugate();deg[i]+=b;deg[j]+=b
    _,u=np.linalg.eigh(np.diag(deg)-H);ph=np.angle(u[:,0]);ph-=ph[0]
    def f(v):
        p=np.r_[0,v];err=(np.exp(1j*(p[edges[:,1]]-p[edges[:,0]]))-obs)*np.sqrt(w)
        return np.r_[err.real,err.imag]
    opt=least_squares(f,ph[1:],max_nfev=150,ftol=1e-11,gtol=1e-11,xtol=1e-11)
    return np.angle(np.exp(1j*np.r_[0,opt.x])),dict(success=bool(opt.success),cost=float(2*opt.cost/w.sum()),nfev=opt.nfev)

def templates(empty):
    q={s:pairval(x[:len(x)//2])[0]/qp(los(s)) for s,x in empty.items()}
    temp={'identity':{s:np.ones((3,30),complex) for s in ST}};fits=[];params=[]
    e=[];z=[];w=[]
    for s in ['G1','G2']:
        for pair,vals in zip(PA,q[s]):
            for v in vals[KT]:e.append(pair);z.append(v);w.append(abs(v))
    ph,diag=fitgraph(e,z,w,3);fits.append(dict(model='rf_constant',slot='all_training',**diag))
    temp['rf_constant']={s:np.tile(np.exp(1j*ph)[:,None],(1,30)) for s in ST}
    for j,p in enumerate(ph):params.append(dict(model='rf_constant',slot='all',element=j+1,phase_deg=np.degrees(p)))
    for mode in ['antenna_constant','antenna_tone']:
        phases=[]
        slots=[None] if mode=='antenna_constant' else list(np.where(KT)[0])
        for k in slots:
            e=[];z=[];w=[]
            for s in ST:
                ids=np.array(sm.MAP[s])-1
                for (i,j),vals in zip(PA,q[s]):
                    vv=vals[KT] if k is None else [vals[k]]
                    for v in vv:e.append([ids[i],ids[j]]);z.append(v);w.append(abs(v))
            ph,diag=fitgraph(e,z,w,12);fits.append(dict(model=mode,slot='all_training' if k is None else k,**diag));phases.append(ph)
            for j,p in enumerate(ph):params.append(dict(model=mode,slot='all' if k is None else k,element=j+1,phase_deg=np.degrees(p)))
        if mode=='antenna_constant':full=np.repeat(phases[0][:,None],30,axis=1)
        else:
            tr=np.unwrap(np.array(phases).T,axis=1)
            from scipy.interpolate import interp1d
            full=interp1d(sm.F[KT],tr,axis=1,kind='linear',fill_value='extrapolate')(sm.F)
        temp[mode]={s:np.exp(1j*full[np.array(sm.MAP[s])-1]) for s in ST}
    save(fits,'fit_diagnostics.csv');save(params,'phase_parameters.csv')
    np.savez_compressed(R/'fixed_templates.npz',**{f'{m}_{s}':h for m,t in temp.items() for s,h in t.items()})
    return temp

def invariants(x,s):
    q,z=pairval(x);eig=np.linalg.eigvalsh(z);tr=np.trace(z,axis1=1,axis2=2).real
    closure=np.degrees(abs(np.angle(q[0]*q[2]*q[1].conj())))
    reliable=(abs(q)>=.5).all(axis=0)
    yy=Y[np.array(sm.MAP[s])-1]
    pred=np.exp(2j*np.pi*sm.F[None,:,None]*(yy[np.array([j for i,j in PA])]-yy[np.array([i for i,j in PA])])[:,None,None]*np.sin(np.deg2rad(sm.ANG))[None,None,:]/C)
    residual=np.abs(np.angle(q[:,:,None]*pred.conj()));score=np.sum(abs(q)[:,:,None]*residual**2,axis=(0,1))/np.sum(abs(q))
    ix=int(np.argmin(score))
    return dict(packets=len(x),dominant_eigen_fraction_median=float(np.median(eig[:,-1]/tr)),
        pair_coherence_median=float(np.median(abs(q))),reliable_pair_fraction=float((abs(q)>=.5).mean()),
        triangle_closure_median_deg=float(np.median(closure[reliable])) if reliable.any() else np.nan,
        best_single_plane_angle_parameter_deg=float(sm.ANG[ix]),single_plane_phase_rms_deg=float(np.degrees(np.sqrt(score[ix]))))

def main():
    (R/'spectra').mkdir(exist_ok=True);sources=[]
    def load(p):
        p=Path(p);sources.append(dict(path=str(p),sha256=sha(p)));return np.load(p)['x']
    signs=json.loads((O/'empty_pi_sign_validation_20261104/frozen_signs.json').read_text())['pairs']
    evsign={'E1':np.ones(3),'E2':np.array(signs['E1_E2']['sign']),'E3':np.array(signs['E1_E3']['sign'])};evsign['E4']=evsign['E3']*np.array(signs['E3_E4']['sign'])
    empty={(cap,r,s):load(sm.B/'cache'/f'{start+r-1:03d}_{s}.npz')*evsign[cap][None,:,None] for cap,start in sm.CAP.items() for r in ([1,2,3] if cap=='E1' else [3]) for s in ST}
    temp=templates({s:empty[('E1',1,s)] for s in ST})
    phase=[];phasearrays={}
    for model,tt in temp.items():
        for (cap,r,s),x in empty.items():
            if cap=='E1' and r==1:x=x[len(x)//2:];role='source_held_time'
            elif cap=='E1' and r==2:role='development_run'
            else:role='retrospective_full_run_check'
            for scope,mask in [('withheld_frequency',KV),('all_frequency',np.ones(30,bool))]:
                phase.append(dict(model=model,capture=cap,run=r,state=s,role=role,frequency_scope=scope,packets=len(x),**phasecheck(x,tt[s]*los(s),mask)))
            phasearrays[f'{model}_{cap}_{r}_{s}']=np.angle(pairval(x)[0]*qp(tt[s]*los(s)).conj())
    phdf=save(phase,'phase_prediction.csv');np.savez_compressed(R/'phase_residual_arrays.npz',**phasearrays)
    dev=phdf[(phdf.capture=='E1')&(phdf.run==2)&(phdf.frequency_scope=='withheld_frequency')&phdf.state.isin(GS)]
    select=dev.groupby('model').agg(passed_G=('phase_prediction_gate','sum'),median_phase_deg=('phase_median_deg','median'),worst_p90_deg=('phase_p90_deg','max')).reset_index()
    passing=select[(select.model!='identity')&(select.passed_G==4)].sort_values('median_phase_deg')
    chosen=passing.model.iloc[0] if len(passing) else 'identity'
    save(select,'development_model_ranking.csv')
    (R/'development_decision.json').write_text(json.dumps(dict(candidate=chosen,physical_calibration_verified=False,rule='all four G phase gates on E1 run2 withheld frequencies, then median residual; otherwise identity'),indent=2))
    models={};th=[]
    for m,s in itertools.product(MODES,ST):
        corr=temp[m][s].conj()
        mod,td,ts=sm.model_from_source(empty[('E1',1,s)]*corr[None],empty[('E1',2,s)]*corr[None],s)
        models[m,s]=(mod,td,ts)
        np.savez_compressed(R/f'background_{m}_{s}.npz',background=mod['background'],distance_threshold=td,score_thresholds=ts)
        th.append(dict(model=m,state=s,distance_threshold=td,score_threshold=ts[0]))
    save(th,'frozen_background_thresholds.csv')
    freeze('model_freeze.json',[R/'PROTOCOL.md',R/'run_check.py',R/'fixed_templates.npz',R/'development_decision.json',R/'frozen_background_thresholds.csv']+list(R.glob('background_*.npz')))
    print('model frozen, chosen',chosen,flush=True)
    ocdec=json.loads((S/'occupied_frozen_signs.json').read_text());oc={};raw=[];qs={}
    oldsrc=O/'static_human_aoa_validation_20261101/packets'
    for first,off,s in itertools.product(FIRST,[0,1,2],ST):
        idx=first+off;p=oldsrc/f'{idx}_{s}.npz' if off==0 else A/'occupied_packets'/f'{idx}_{s}.npz'
        x=load(p)[64:]*np.array(ocdec[f'E1_{first}']['sign'])[None,:,None]
        oc[idx,s]=x;qs[idx,s]=pairval(x)[0]
        raw.append(dict(batch_first=first,index=idx,state=s,**invariants(x,s)))
    save(raw,'occupied_raw_invariants.csv')
    repeat=[];closure=[]
    for first,off,s in itertools.product(FIRST,[1,2],ST):
        a=qs[first,s];b=qs[first+off,s];mask=(abs(a)>=.5)&(abs(b)>=.5)
        err=np.degrees(abs(np.angle(a*b.conj())))[mask]
        repeat.append(dict(batch_first=first,index=first+off,state=s,phase_median_deg=np.median(err),phase_p90_deg=np.quantile(err,.9),reliable_fraction=mask.mean(),
                           spatial_distance=sm.dist(sm.spatial(oc[first,s]),sm.spatial(oc[first+off,s]))))
    for first,off in itertools.product(FIRST,[0,1,2]):
        idx=first+off;a=qs[idx,'C1'][2];b=qs[idx,'C2'][2];mask=(abs(a)>=.5)&(abs(b)>=.5)
        er=np.degrees(abs(np.angle(a*b.conj())))[mask]
        closure.append(dict(batch_first=first,index=idx,shared_pair='antenna6_minus_antenna8',median_deg=np.median(er),p90_deg=np.quantile(er,.9),reliable_fraction=mask.mean()))
    save(repeat,'within_batch_phase_stability.csv');save(closure,'shared_pair_closure.csv')
    stats=[];cand=[];diag=[];null=[];nullp=[];controls=[]
    for m,s in itertools.product(MODES,ST):
        mod,td,ts=models[m,s];corr=temp[m][s].conj()
        for first,off in itertools.product(FIRST,[0,1,2]):
            idx=first+off;x=oc[idx,s]*corr[None]
            st,pp,arr=sm.assess(x,mod,td,ts)
            stats.append(dict(model=m,batch_first=first,index=idx,state=s,**st))
            for p in pp:cand.append(dict(model=m,batch_first=first,index=idx,state=s,**p))
            np.savez_compressed(R/'spectra'/f'{m}_{idx}_{s}.npz',**arr)
            diag.append(dict(model=m,batch_first=first,index=idx,state=s,**invariants(x,s)))
        for cap in sm.CAP:
            x=empty[cap,3,s][64:]*corr[None]
            st,pp,arr=sm.assess(x,mod,td,ts)
            null.append(dict(model=m,capture=cap,state=s,packets=len(x),low_sample=len(x)<128,**st))
            for p in pp:nullp.append(dict(model=m,capture=cap,state=s,**p))
        # If steering is transformed consistently as a coordinate change, scores must stay invariant.
        b0=models['identity',s][0]['background'];x0=oc[12,s];c0=sm.spatial(x0);mod0=models['identity',s][0]
        D=np.array([np.diag(corr[:,k]) for k in range(30)])
        mt=sm.prepare(D@b0@D.conj().transpose(0,2,1),s);mt['wa']=mt['w']@(D@mod0['a'])
        wb=mt['w']@mt['background']@mt['w'].conj().transpose(0,2,1)
        mt['den']=np.real(np.sum(mt['wa'].conj()*(wb@mt['wa']),axis=1))
        e=float(abs(sm.scores(D@c0@D.conj().transpose(0,2,1),mt)-sm.scores(c0,mod0)).max())
        eig=float(abs(np.linalg.eigvalsh(sm.spatial(x0*corr[None]))-np.linalg.eigvalsh(c0)).max())
        controls.append(dict(model=m,state=s,coordinate_change_score_max_error=e,eigenvalue_max_error=eig))
        assert e<1e-8 and eig<1e-12
    save(stats,'occupied_scores.csv');save(cand,'occupied_candidates.csv');save(diag,'corrected_manifold_diagnostics.csv')
    save(null,'empty_scores.csv');save(nullp,'empty_candidates.csv');save(controls,'coordinate_controls.csv')
    save(sources,'source_manifest.csv').drop_duplicates().to_csv(R/'source_manifest.csv',index=False)
    np.savez_compressed(R/'occupied_pair_correlations.npz',**{f'{idx}_{s}':q for (idx,s),q in qs.items()})
    pred=['occupied_scores.csv','occupied_candidates.csv','corrected_manifold_diagnostics.csv','empty_scores.csv','empty_candidates.csv','occupied_raw_invariants.csv',
          'within_batch_phase_stability.csv','shared_pair_closure.csv','phase_prediction.csv','source_manifest.csv']
    freeze('spectral_predictions_frozen.json',[R/f for f in pred])
    print(select.to_string(index=False));print('spectral rows',len(stats),'null rows',len(null),flush=True)
if __name__=='__main__':
    with sm.threadpool_limits(limits=1):main()
