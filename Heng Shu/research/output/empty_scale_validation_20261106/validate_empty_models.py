"""Empty-only predictive noise-scale checks. Never processes R2 samples/truth."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'noncoherent_bc_followup_20261106'))
import pcd_kernel as pcd
old=pcd.prior
TRAIN=np.arange(30)%3!=2
HELD=~TRAIN
EYE=np.eye(3,dtype=complex)
MODELS=('NSI','NSD')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,rows):pd.DataFrame(rows).to_csv(ROOT/n,index=False,encoding='utf-8-sig')
def dump(n,obj):(ROOT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def norm_spectrum(d):return np.max(abs(np.linalg.eigvalsh(old.herm(d))),axis=-1)
def project(B,d):return old.herm(B@d@B)
def proxy(raw):return np.mean(90/np.maximum(np.sum(abs(raw)**2,axis=(-2,-1)),1e-30),axis=1)
def columns(B,model):
    basis=[EYE] if model=='NSI' else [np.diag(np.eye(3)[i]) for i in range(3)]
    return np.stack([project(B,x) for x in basis],axis=-1)
def fit_model(B,reference,fitblocks,model):
    Q=columns(B,model)
    X=[];Y=[]
    for target,u in fitblocks:
        X.append((u*Q[TRAIN]).reshape(-1,Q.shape[-1]))
        Y.append(project(B,target-reference)[TRAIN].reshape(-1))
    X=np.concatenate(X);Y=np.concatenate(Y)
    A=np.concatenate([X.real,X.imag]);b=np.concatenate([Y.real,Y.imag])
    beta,_,rank,sing=np.linalg.lstsq(A,b,rcond=None)
    cond=float(sing.max()/sing.min()) if len(sing) and sing.min()>0 else float('inf')
    valid=bool(np.all(np.isfinite(beta)) and rank==len(beta) and cond<=1e6 and np.all(beta>=0))
    N=np.einsum('kmnd,d->kmn',Q,beta)
    return beta,N,dict(fit_valid=valid,fit_rank=int(rank),fit_condition=cond,nonnegative_coefficients=bool(np.all(beta>=0)))

def math_checks():
    # Distinct known software direction, no standing-point geometry.
    r=np.random.default_rng(12);h=r.normal(size=(30,3))+1j*r.normal(size=(30,3));h=h/np.linalg.norm(h,axis=1,keepdims=True)
    B=EYE-old.outer(h);ref=old.outer(h)+.03*EYE
    outputs={}
    for model,beta in [('NSI',np.array([.024])),('NSD',np.array([.01,.025,.04]))]:
        Q=columns(B,model);N=np.einsum('kmnd,d->kmn',Q,beta)
        fit=[(ref+u*N,u) for u in [-.2,.1,.3,.5]]
        estimate,pred,meta=fit_model(B,ref,fit,model)
        assert meta['fit_valid'] and np.allclose(estimate,beta,atol=1e-12)
        assert np.allclose(pred,N,atol=1e-12)
        # Empty correction predicts previously unused u, while preserving an added coherent target.
        a=old.STEER['G2'][:,:,np.where(old.ANG==27)[0][0]]
        foreground=old.outer(h+.2*a)-old.outer(h)
        corrected=project(B,ref+.17*N+foreground-ref)-.17*pred
        expected=project(B,foreground)
        assert np.allclose(corrected,expected,atol=1e-12)
        outputs[model]=dict(max_coefficient_error=float(abs(estimate-beta).max()),added_target_preserved=True)
    # A pure coherent rank-one change has an isotropic trace component; that component is not identified noise.
    P=B[0];u=np.linalg.eigh(P)[1][:,-1];D=np.outer(u,u.conj());b=np.trace(D).real/2
    assert np.isclose(np.linalg.norm(b*P)**2/np.linalg.norm(D)**2,.5)
    outputs['oracle_trace_noise_fraction_counterexample']=True
    outputs['physical_validation']=False
    dump('mathematical_checks.json',outputs)

def main():
    prior=ROOT.parent/'noncoherent_bc_followup_20261106'
    frozen=json.loads((prior/'prediction_freeze.json').read_text('utf8'))
    assert all(sha(prior/n)==v for n,v in frozen.items())
    cache_hashes=json.loads((old.ROOT/'input_cache_hashes.json').read_text('utf8'))
    assert all(sha(n)==v for n,v in cache_hashes.items())
    dump('pre_execution_freeze.json',{n:sha(ROOT/n) for n in ['protocol.json','validate_empty_models.py']})
    dump('input_hashes.json',{'prior_frozen':frozen,'selected_cache':cache_hashes,'R2_processed':False})
    math_checks()
    params=[];checks=[];ranks=[];summary=[];envelopes=[];budgets=[];snapshots={}
    for site in 'ABCDE':
        cache=np.load(old.OLD/f'{site}_selected.npz')
        # Empty run indices only; no occupied window enters this computation.
        empty={s:{'R1':cache['switched'][0,:,j].astype(complex),'R3':cache['switched'][2,:,j].astype(complex)} for j,s in enumerate(old.ST)}
        empty['G2_control']={'R1':cache['g2'][0].astype(complex),'R3':cache['g2'][2].astype(complex)}
        ledger=pd.read_csv(old.OLD/f'{site}_packets.csv')
        for (run,arm),q in ledger.groupby(['run','arm']):
            assert len(q)==q.packet.nunique()==864
            assert len(q[q['sample']<8])==288
            budgets.append(dict(point=site,run=run,arm=arm,unique_packets=864,first8=288,model_fit=run in [1,3],R2_processed=False))
        for state,runs in empty.items():
            moments={k:old.cov(v) for k,v in runs.items()}
            a={k:proxy(v) for k,v in runs.items()}
            bg=old.background(np.stack([moments['R1'],moments['R1'],moments['R3']]))
            B=bg['B'];ref=bg['ref'];floor=pcd.training_change_floor(bg)
            a_ref=a['R1'][:8].mean()
            u=lambda run,i,j:float(a[run][i:j].mean()/a_ref-1)
            target=lambda run,i,j:moments[run][i:j].mean(0)
            fitblocks=[(target(run,i,i+2),u(run,i,i+2)) for run in ['R1','R3'] for i in range(0,8,2)]
            urange=(min(x[1] for x in fitblocks),max(x[1] for x in fitblocks))
            cal=np.stack([project(B,target('R1',i,i+4)-ref) for i in [8,12]])
            cs=norm_spectrum(cal)
            env={}
            for band,mask in [('full',HELD),('low',HELD&(np.arange(30)<15)),('high',HELD&(np.arange(30)>=15))]:
                env[band]=np.max(np.quantile(cs[:,mask],[.5,.9],axis=1),axis=1)
                envelopes.append(dict(point=site,state=state,band=band,median=float(env[band][0]),p90=float(env[band][1]),calibration='R1 samples8:16 only',PCD_floor_unchanged=True))
            for model in MODELS:
                beta,N,meta=fit_model(B,ref,fitblocks,model)
                params.append(dict(point=site,state=state,model=model,coefficients=json.dumps(beta.tolist()),u_min=urange[0],u_max=urange[1],**meta))
                snapshots[f'{site}_{state}_{model}_beta']=beta
                modelrows=[]
                for run,ids in [('R1',[16,20]),('R3',[8,12,16,20])]:
                    for i in ids:
                        actual=project(B,target(run,i,i+4)-ref);uw=u(run,i,i+4);res=old.herm(actual-uw*N)
                        raw_e=np.linalg.eigvalsh(actual);res_e=np.linalg.eigvalsh(res)
                        original_pos=(raw_e>floor[:,None]+1e-10).sum(1)
                        corrected_pos=(res_e>floor[:,None]+1e-10).sum(1)
                        for m,ev in [('STATIC',raw_e),(model,res_e)]:
                            pos=(ev>floor[:,None]+1e-10).sum(1)
                            ranks.append(dict(point=site,state=state,model=m,block=f'{run}_{i}_{i+4}',q0=int((pos==0).sum()),q1=int((pos==1).sum()),q2=int((pos==2).sum()),held_q0=int((pos[HELD]==0).sum()),held_q1=int((pos[HELD]==1).sum()),held_q2=int((pos[HELD]==2).sum()),diagnostic_only=True))
                        rawsp=norm_spectrum(actual);rsp=norm_spectrum(res)
                        for band,mask in [('full',HELD),('low',HELD&(np.arange(30)<15)),('high',HELD&(np.arange(30)>=15))]:
                            orig=np.quantile(rawsp[mask],[.5,.9]);post=np.quantile(rsp[mask],[.5,.9]);limit=env[band]
                            clauses=dict(gate_fit=meta['fit_valid'],gate_proxy=bool(urange[0]-1e-12<=uw<=urange[1]+1e-12),
                                gate_envelope=bool(np.all(post<=limit+1e-10)),gate_not_worse=bool(np.all(post<=orig+1e-10)),
                                gate_positive_not_increased=bool(np.sum(corrected_pos[mask]>0)<=np.sum(original_pos[mask]>0)))
                            row=dict(point=site,state=state,model=model,block=f'{run}_{i}_{i+4}',band=band,u=uw,
                                raw_median=float(orig[0]),corrected_median=float(post[0]),envelope_median=float(limit[0]),
                                raw_p90=float(orig[1]),corrected_p90=float(post[1]),envelope_p90=float(limit[1]),
                                pass_all=all(clauses.values()),failure=';'.join(k for k,v in clauses.items() if not v),**clauses)
                            checks.append(row);modelrows.append(row)
                summary.append(dict(point=site,state=state,model=model,pass_model=all(x['pass_all'] for x in modelrows),
                    fit_valid=meta['fit_valid'],passed_cells=sum(x['pass_all'] for x in modelrows),cells=len(modelrows),
                    envelope_failures=sum(not x['gate_envelope'] for x in modelrows),proxy_failures=sum(not x['gate_proxy'] for x in modelrows),
                    worsening_failures=sum(not x['gate_not_worse'] for x in modelrows),positive_count_failures=sum(not x['gate_positive_not_increased'] for x in modelrows)))
        print(site,'empty-only model validation complete',flush=True)
    save('model_parameters.csv',params);save('held_empty_checks.csv',checks);save('empty_rank_counts.csv',pd.DataFrame(ranks).drop_duplicates().to_dict('records'));save('state_model_gate.csv',summary);save('R1_calibration_envelopes.csv',envelopes);save('packet_budget.csv',budgets)
    sf=pd.DataFrame(summary);elig=[]
    for site in 'ABCDE':
        for model in MODELS:
            for arm,states in [('G2',('G2_control',)),('SIX',old.ST)]:
                q=sf[(sf.point==site)&(sf.model==model)&sf.state.isin(states)]
                assert len(q)==len(states)
                elig.append(dict(point=site,arm=arm,model=model,empty_model_pass=bool(q.pass_model.all()),passed_states=int(q.pass_model.sum()),required_states=len(states),failed_states='/'.join(q[~q.pass_model].state)))
    save('human_branch_eligibility.csv',elig)
    np.savez_compressed(ROOT/'model_coefficients.npz',**snapshots)
    files=['protocol.json','validate_empty_models.py','pre_execution_freeze.json','input_hashes.json','mathematical_checks.json','model_parameters.csv','held_empty_checks.csv','empty_rank_counts.csv','state_model_gate.csv','R1_calibration_envelopes.csv','packet_budget.csv','human_branch_eligibility.csv','model_coefficients.npz']
    dump('empty_stage_freeze.json',{n:sha(ROOT/n) for n in files})
    print(sf.groupby('model')[['pass_model','fit_valid']].sum().to_string())
    print(pd.DataFrame(elig).to_string(index=False))
    print('Empty decisions frozen; no occupied CSI or survey scoring executed.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
