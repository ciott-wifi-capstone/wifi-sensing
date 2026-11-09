"""Empty-only background whitening. No cross-session or human fitting."""
from audit_inputs import *
from scipy.ndimage import maximum_filter
ALPHAS=[.1,.03,.3]
def ratio(cov,background,a,alpha):
 n=len(background);reg=(1-alpha)*background+alpha*np.trace(background).real/n*np.eye(n)
 eig,u=np.linalg.eigh(reg);W=(u/np.sqrt(np.maximum(eig,1e-15)))@u.conj().T
 b=W@a;cr=W@cov@W.conj().T;br=W@background@W.conj().T
 num=np.real(np.sum(b.conj()*(cr@b),axis=0));den=np.real(np.sum(b.conj()*(br@b),axis=0))
 return (num/np.maximum(den,1e-15)).reshape(len(rat.ANG),-1)
def main():
 rat.TAU=rat.TAU[:-1]
 rows=[];peaks=[];checks=[]
 for ci,cap in enumerate(['E1','E2','E3','E4']):
  ids=[78+3*ci+j for j in range(3)]
  for state in rat.MAP:
   zz=[np.load(R/'cache'/f'{i:03d}_{state}.npz') for i in ids]
   bg=zz[0]['covariance'];a=rat.steering(state)
   dev=[rat.covariance(t) for t in np.array_split(zz[1]['x'],4)]
   val=[rat.covariance(t) for t in np.array_split(zz[2]['x'],4)]
   maps={}
   for alpha in ALPHAS:
    unity=ratio(bg,bg,a,alpha);assert np.max(abs(unity-1))<1e-8
    threshold=max(1.,max(float(ratio(c,bg,a,alpha).max()) for c in dev))
    for block,cov in [('full',zz[2]['covariance'])]+[(str(i),c) for i,c in enumerate(val)]:
     score=ratio(cov,bg,a,alpha);mask=(score==maximum_filter(score,size=(5,5),mode=('nearest','wrap')))&(score>threshold)
     ij=np.argwhere(mask);best=np.unravel_index(score.argmax(),score.shape)
     rows.append(dict(capture=cap,state=state,alpha=alpha,validation_run=ids[2],block=block,threshold=threshold,max_ratio=score.max(),false_candidate_peaks=len(ij),has_false_candidate=bool(len(ij)),best_angle_parameter_deg=rat.ANG[best[0]],best_delay_parameter_ns=rat.TAU[best[1]],cross_session_aligned=False))
     for i,j in ij:peaks.append(dict(capture=cap,state=state,alpha=alpha,block=block,angle_parameter_deg=rat.ANG[i],delay_parameter_ns=rat.TAU[j],ratio=score[i,j],threshold=threshold,occupancy_truth='empty'))
     if block=='full':maps[f'ratio_{alpha}']=score
    checks.append(dict(capture=cap,state=state,alpha=alpha,identity_ratio_error=float(abs(unity-1).max())))
   maps['baseline_music']=rat.music(zz[2]['covariance'],a);maps['background_music']=rat.music(bg,a)
   np.savez_compressed(R/'cache'/f'background_scores_{cap}_{state}.npz',angles=rat.ANG,delay_parameter_ns=rat.TAU,**maps)
  print('background validated',cap,flush=True)
 df=pd.DataFrame(rows);df.to_csv(R/'empty_false_candidates.csv',index=False)
 pd.DataFrame(peaks).to_csv(R/'empty_candidate_peaks.csv',index=False)
 pd.DataFrame(checks).to_csv(R/'whitening_identity_checks.csv',index=False)
 sm=df[df.block=='full'].groupby(['capture','alpha']).agg(states=('state','size'),states_with_false_candidates=('has_false_candidate','sum'),false_peak_count=('false_candidate_peaks','sum'),median_max_ratio=('max_ratio','median'),median_threshold=('threshold','median'))
 sm.to_csv(R/'empty_false_candidate_summary.csv');print(sm.to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
