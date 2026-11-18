from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101'
f=pd.concat([pd.read_csv(R/'fused_peaks.csv'),pd.read_csv(R/'initial_phase_fused_peaks.csv')],ignore_index=True)
m=pd.read_csv(S/'input_manifest.csv')[['index','label']];g=pd.read_csv(S/'results/evaluation_geometry.csv')[['label','theory_angle_deg']];f=f.merge(m,on='index',validate='many_to_one').merge(g,on='label',validate='many_to_one');f['unassigned_peak_abs_deviation_deg']=abs((f.angle_deg-f.theory_angle_deg+180)%360-180);f['valid_human_output']=False
f.to_csv(R/'post_prediction_peak_deviations.csv',index=False)
rows=[]
for (stage,branch),a in f.groupby(['stage','branch']):
 e=a.unassigned_peak_abs_deviation_deg
 rows.append(dict(stage=stage,branch=branch,records=len(a),unassigned_peak_mean_deviation_deg=float(e.mean()),median=float(e.median()),p90=float(e.quantile(.9)),max=float(e.max()),fraction_peak_within8=float((e<=8).mean()),physical_human_MAE=None,validated_human_outputs=0))
(R/'evaluation_summary.json').write_text(json.dumps(rows,indent=2))
print(f[f.stage=='evaluation'][['index','label','branch','angle_deg','theory_angle_deg','unassigned_peak_abs_deviation_deg']].to_string(index=False));print(json.dumps([r for r in rows if r['stage']=='evaluation'],indent=2))
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
fig,axs=plt.subplots(2,2,figsize=(10.2,7),sharex=True,sharey=True)
for ax,key,label in zip(axs.flat,[3,15,24,33],['+1 m','-1 m','+2 m','-2 m']):
 z=np.load(R/f'spectra_{key}.npz');v=np.load(R/f'initial_phase_spectra_{key}.npz');ang=z['angle_deg'];truth=float(f[f['index']==key].iloc[0].theory_angle_deg)
 ax.plot(ang,10*np.log10(np.maximum(z['fused_baseline'],1e-4)),label='Baseline',lw=1.8)
 ax.plot(ang,10*np.log10(np.maximum(v['fused_conditional_initial_phase'],1e-4)),label='Unverified initial-phase correction',lw=1.6)
 ax.axvline(truth,color='black',ls=':',label='Human direction (evaluation only)');ax.set_title(f'{label} | record {key}');ax.grid(alpha=.25);ax.set_ylim(-20,1)
for ax in axs[-1]:ax.set_xlabel('RX-local angle parameter (deg)')
for ax in axs[:,0]:ax.set_ylabel('Normalized fused score (dB)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=1,bbox_to_anchor=(.5,-.015));fig.suptitle('Four predefined evaluation records: unassigned peaks, not calibrated human AoA');fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(R/'four_point_comparison.png',dpi=150);fig.savefig(R/'four_point_comparison.pdf');plt.close(fig)
# Separate frequency variation from cross-development-record phase inconsistency.
geo=json.loads((S/'geometry.json').read_text(encoding='utf8'));freq=5.23e9+np.arange(-58,59,4)*312500.;ds=[]
for state,ants in geo['state_map'].items():
 y=np.array([geo['array_y_local_m'][a-1] for a in ants]);d=np.sqrt(3.65**2+y*y);geom=-2*np.pi*(d[1:]-d[0])[:,None]*freq[None,:]/299792458.;per=[]
 for key in [0,12,21,30]:
  x=np.load(S/'packets'/f'{key}_{state}.npz')['x'][:64];p=x[:,1:]*x[:,:1].conj();u=p/np.maximum(abs(p),1e-30);mean=(u*np.exp(-1j*geom)[None]).mean(0);per.append(np.exp(1j*np.angle(mean)))
 rho=abs(np.mean(per,axis=0))
 for ci in range(2):ds.append(dict(state=state,chain=ci+2,cross_record_per_tone_resultant_median=float(np.median(rho[ci])),cross_record_per_tone_resultant_min=float(rho[ci].min())))
pd.DataFrame(ds).to_csv(R/'template_cross_record_per_tone.csv',index=False);print(pd.DataFrame(ds).to_string(index=False))
