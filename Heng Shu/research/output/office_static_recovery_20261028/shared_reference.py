"""Freeze one reference per state from R1 prefix; validate R2/R3 without retuning.
No state-to-state alignment, no chain-specific corrections, no human geometry fit.
"""
import analyze as a
import numpy as np
import pandas as pd
import json
from threadpoolctl import threadpool_limits
def main():
 rows=[];held=[];pairs=[];spec={};covs={};checks=[]
 for state in a.STATES:
  z=np.load(a.O/'packets'/f'R1_{state}.npz');ref=a.rat.unit(z['corrected'][:64].mean(axis=0));train,test=a.rat.crossfit_masks()
  for run in [1,2,3]:
   z=np.load(a.O/'packets'/f'R{run}_{state}.npz');x=z['x'];g=a.rat.common_gate(np.concatenate([ref[None],x[:63]]))
   out=x.copy();accepted=[]
   for i in range(len(x)):
    t,p,c,edge=a.fit(x[i],ref);ok=g['common_model_pass'] and c>=.9 and not edge;accepted.append(ok)
    if ok:out[i]=a.rat.apply(x[i],t,p)
    if i>=64 and i%8==0:
     tt,pp,cc,ee=a.fit(x[i],ref,train);prop=a.rat.apply(x[i],tt,pp)
     tb,pb,_,_=a.fit(x[i],ref,test);ta,pa,_,_=a.fit(prop,ref,test)
     held.append(dict(run=run,state=state,packet=i,raw_test_tau_ns=tb,proposed_test_residual_tau_ns=ta,raw_test_phase_rad=pb,proposed_test_residual_phase_rad=pa,real_before=a.rat.score(x[i],ref,test)[0],real_after=a.rat.score(prop,ref,test)[0],magnitude_after=a.rat.score(prop,ref,test)[1]))
   r=a.cov(out);s=a.music(r,a.steer(state));covs[(run,state)]=r;spec[(run,state)]=s
   np.savez_compressed(a.O/'packets'/f'shared_R{run}_{state}.npz',corrected=out,record_index=z['record_index'],reference=ref,accepted=np.array(accepted))
   rows.append(dict(run=run,state=state,**g,accepted_fraction=float(np.mean(accepted)),peak_angle_deg=a.peak(s)[0],peak_tau_ns=a.peak(s)[1]))
   checks.append(a.rat.pair_error(x,out))
  for r1,r2 in [(1,2),(1,3),(2,3)]:pairs.append(dict(run_a=r1,run_b=r2,state=state,shared_cov_distance=a.dist(covs[(r1,state)],covs[(r2,state)]),shared_spectrum_rms=float(np.sqrt(np.mean((spec[(r1,state)]-spec[(r2,state)])**2)))))
 a.csv('shared_reference_gate.csv',rows);a.csv('heldout_reference_validation.csv',held);a.csv('shared_reference_between_runs.csv',pairs)
 a.O.joinpath('results/shared_reference_checks.json').write_text(json.dumps({'pair_phase_max_change_deg':max(checks),'reference_source':'R1 first 64 sampled packets, separately per state','no_retuning':True,'gate_pass':sum(r['common_model_pass'] for r in rows)},indent=2))
 np.savez_compressed(a.O/'results/shared_reference_spectra.npz',angle_deg=a.ANG,tau_ns=a.TAU,**{f'R{r}_{s}':p for (r,s),p in spec.items()})
 fig,ax=a.plt.subplots(3,6,figsize=(18,10),layout='constrained')
 for ri in range(3):
  for si,s in enumerate(a.STATES):
   im=ax[ri,si].imshow(10*np.log10(np.maximum(spec[(ri+1,s)],1e-4)),origin='lower',aspect='auto',extent=[a.TAU[0],a.TAU[-1],a.ANG[0],a.ANG[-1]],vmin=-25,vmax=0,cmap='magma');ax[ri,si].set_title(f'R{ri+1} {s}');ax[ri,si].set_xlabel('Uncalibrated delay (ns)');ax[ri,si].set_ylabel('Uncalibrated angle (deg)')
 fig.colorbar(im,ax=ax,shrink=.7,label='Per-map normalized MUSIC (dB)');fig.suptitle('Office HT20 | one frozen R1 reference per state | R2/R3 validation\n28 mm | q=3 | relative reference recovered; physical axes and human path unvalidated',fontsize=15);a.savefig(fig,'04_music_shared_reference')
 h=pd.DataFrame(held);summary=[]
 for (run,state),df in h.groupby(['run','state']):
  summary.append(dict(run=int(run),state=state,raw_test_delay_iqr_ns=a.rat.iqr(df.raw_test_tau_ns),corrected_test_delay_iqr_ns=a.rat.iqr(df.proposed_test_residual_tau_ns),raw_test_phase_sd_deg=a.rat.circ_sd(df.raw_test_phase_rad)*180/np.pi,corrected_test_phase_sd_deg=a.rat.circ_sd(df.proposed_test_residual_phase_rad)*180/np.pi,test_real_before=float(df.real_before.median()),test_real_after=float(df.real_after.median()),test_magnitude_after=float(df.magnitude_after.median())))
 a.csv('heldout_reference_summary.csv',summary)
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))
 fig,ax=a.plt.subplots(1,2,figsize=(12,4.5),layout='constrained');d=pd.DataFrame(summary);ix=np.arange(len(d));labels=[f'R{r.run}-{r.state}' for r in d.itertuples()]
 ax[0].plot(ix,d.raw_test_delay_iqr_ns,'o-',label='Raw');ax[0].plot(ix,d.corrected_test_delay_iqr_ns,'s-',label='Held-out residual after common fit');ax[0].set_ylabel('Timing IQR on omitted entries (ns)');ax[0].legend()
 original=pd.read_csv(a.O/'results/between_runs.csv');merged=original.merge(pd.DataFrame(pairs),on=['run_a','run_b','state'])
 for col,label in [('raw_cov_distance','A raw'),('corrected_cov_distance','C separate references'),('shared_cov_distance','D frozen R1 references')]:ax[1].plot(np.arange(18),merged[col],'o-',label=label)
 ax[1].set_ylabel('Inter-run relative covariance distance');ax[1].legend();ax[1].set_xticks(np.arange(18),[f'{r.run_a}/{r.run_b}-{r.state}' for r in merged.itertuples()],rotation=90,fontsize=7);ax[0].set_xticks(ix,labels,rotation=90,fontsize=7)
 for aa in ax:aa.grid(alpha=.25)
 fig.suptitle('Reference recovery: withheld-entry prediction and whole-run comparison');a.savefig(fig,'05_shared_reference_validation')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
