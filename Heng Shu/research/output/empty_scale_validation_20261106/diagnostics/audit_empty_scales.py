"""Empty-only held-out diagnostics; never index occupied run 2 or score AoA."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent.parent
CACHE=OUT/'coherent_array_validation_20261106'
ST=('G1','G2','G3','G4','C1','C2')
I=np.eye(3,dtype=complex)
EVEN=np.arange(30)%2==0

def herm(x):return (x+x.conj().swapaxes(-1,-2))/2
def cov(x):return np.einsum('npmk,nplk->nkml',x,x.conj())/x.shape[1]
def norm(x):return np.linalg.norm(x,axis=(-2,-1))
def measure(D,B):
    beta=np.trace(D,axis1=-2,axis2=-1).real/2
    iso=beta[:,None,None]*B;res=herm(D-iso)
    N=np.maximum(norm(D),1e-15)
    ev=np.linalg.eigvalsh(D)
    return dict(oracle_isotropic_fraction=float(np.median(norm(iso)**2/N**2)),
      oracle_anisotropic_residual_fraction=float(np.median(norm(res)/N)),
      beta_median=float(np.median(beta)),beta_min=float(beta.min()),beta_max=float(beta.max()),
      projected_delta_norm_median=float(np.median(N)))
def affine(ref,target,ids):
    A=ref[ids];Y=target[ids]
    g=np.array([[np.vdot(A,A).real,np.trace(A,axis1=-2,axis2=-1).real.sum()],
                [np.trace(A,axis1=-2,axis2=-1).real.sum(),3*len(A)]])
    rhs=np.array([np.vdot(A,Y).real,np.trace(Y,axis1=-2,axis2=-1).real.sum()])
    return np.linalg.solve(g,rhs)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((CACHE/'prediction_freeze.json').read_text('utf8'))
    scales=[];models=[];tones=[];source=[]
    for pt in 'ABCDE':
        path=CACHE/f'{pt}_selected.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert prior[path.name]==sha
        z=np.load(path)
        # Only the two confirmed empty run slices are used below.
        p=z['switched'][[0,2]].astype(complex)
        g=z['g2'][[0,2]].astype(complex)
        raw={s:p[:,:,j] for j,s in enumerate(ST)};raw['G2_control']=g
        for s,x in raw.items():
            assert x.shape[:2]==(2,24)
            source.append(dict(point=pt,state=s,path=str(path),sha256=sha,runs_used='1,3',occupied_run_used=False,R1fit_cycles='0:8',R1held_cycles='8:24',R3dev_cycles='0:8',R3held_cycles='8:24',packets_per_cycle=x.shape[2]))
            scale=np.sqrt(90/np.mean(np.sum(abs(x[0,:8])**2,axis=(-2,-1))))
            reps={'unit_packet':x*np.sqrt(90)/np.maximum(np.linalg.norm(x,axis=(-2,-1),keepdims=True),1e-15),
                  'fixed_R1_scale_raw':x*scale}
            for rep,xx in reps.items():
                m=np.stack([cov(xx[r]) for r in range(2)])
                ref=m[0,:8].mean(0);e,v=np.linalg.eigh(herm(ref));h=v[:,:,-1]
                B=I-h[:,:,None]*h[:,None,:].conj()
                target={'R1_held':m[0,8:].mean(0),'R3_dev':m[1,:8].mean(0),'R3_held':m[1,8:].mean(0)}
                D={key:herm(B@(y-ref)@B) for key,y in target.items()}
                for win,d in D.items():
                    scales.append(dict(point=pt,state=s,representation=rep,window=win,**measure(d,B)))
                beta_tone=np.trace(D['R3_dev'],axis1=1,axis2=2).real/2
                beta_global=float(beta_tone[EVEN].mean())
                alpha,offset=affine(ref,target['R3_dev'],np.flatnonzero(EVEN))
                prediction={
                  'unchanged':ref,
                  'projected_scalar_per_tone':ref+beta_tone[:,None,None]*B,
                  'projected_scalar_shared_frequency':ref+beta_global*B,
                  'affine_aR1_bI_shared_frequency':alpha*ref+offset*I,
                  'full_empty_anchor_per_tone':target['R3_dev']}
                for name,pr in prediction.items():
                    for win in ['R3_dev','R3_held']:
                        for band,ids in [('fit_even',np.flatnonzero(EVEN)),('held_odd',np.flatnonzero(~EVEN))]:
                            d0=D[win][ids];res=herm(B@(target[win]-pr)@B)[ids]
                            fullres=herm(target[win]-pr)[ids]
                            fullbase=herm(target[win]-ref)[ids]
                            r1=np.maximum(norm(D['R1_held'][ids]),1e-15)
                            models.append(dict(point=pt,state=s,representation=rep,model=name,window=win,frequency_split=band,
                              frequency_role=('held_target_frequency' if (band=='held_odd' and name in ['projected_scalar_shared_frequency','affine_aR1_bI_shared_frequency']) else 'target_frequency_seen_in_R3dev' if name in ['projected_scalar_per_tone','full_empty_anchor_per_tone'] else 'fit_frequency' if band=='fit_even' else 'no_R3_fit_baseline'),
                              projected_residual_ratio=float(np.median(norm(res)/np.maximum(norm(d0),1e-15))),
                              projected_residual_over_R1held=float(np.median(norm(res)/r1)),
                              full_residual_ratio=float(np.median(norm(fullres)/np.maximum(norm(fullbase),1e-15))),
                              alpha=alpha if name.startswith('affine') else np.nan,
                              offset=offset if name.startswith('affine') else beta_global if name.endswith('shared_frequency') else np.nan,
                              minimum_predicted_eigenvalue=float(np.linalg.eigvalsh(herm(pr)).min())))
                for k in range(30):
                    dh=D['R3_held'][k];bh=np.trace(dh).real/2
                    vals=np.linalg.eigvalsh(v[k,:,:2].conj().T@dh@v[k,:,:2])
                    tones.append(dict(point=pt,state=s,representation=rep,tone_index=k,
                      beta_R3dev=beta_tone[k],beta_R3held=bh,small_eigenvalue=vals[0],large_eigenvalue=vals[1],
                      normalized_gap=abs(vals[1]-vals[0])/max(abs(vals).sum(),1e-15)))
        print(pt,'empty R1/R3 audit complete',flush=True)
    for name,rows in [('empty_scalar_decomposition.csv',scales),('empty_model_predictions.csv',models),('empty_tone_eigenvalues.csv',tones),('empty_source_audit.csv',source)]:
        pd.DataFrame(rows).to_csv(ROOT/name,index=False,encoding='utf-8-sig')
    d=pd.DataFrame(scales)
    q=pd.DataFrame(models)
    print(d[(d.representation=='unit_packet')&(d.window=='R3_held')][['point','state','oracle_isotropic_fraction','oracle_anisotropic_residual_fraction']].to_string(index=False))
    print(q[(q.representation=='unit_packet')&(q.window=='R3_held')&(q.frequency_split=='held_odd')].groupby('model')[['projected_residual_ratio','projected_residual_over_R1held','full_residual_ratio']].median().to_string())
    (ROOT/'audit_scope.json').write_text(json.dumps(dict(occupied_data_used=False,truth_used=False,threshold_tuning=False,run_roles={'R1':'first8reference,last16held','R3':'first8development,last16held'},representations=list(reps),affine_model_note='Shared real coefficients fitted across even tones in R3 dev; odd tones and later cycles held. Unconstrained diagnostic fit; no guarantee nonnegative noise or calibrated hardware.',oracle_note='Per-window isotropic decomposition is a descriptive oracle projection, not a fitted correction licensed for human data.'),indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
