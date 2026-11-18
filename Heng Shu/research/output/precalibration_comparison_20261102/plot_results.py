from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
labels=['Constant','Affine frequency','Per-tone interpolation']
a=pd.read_csv(R/'evaluation_phase_summary.csv').set_index('model').loc[['pi_constant','pi_affine','pi_tone']];b=pd.read_csv(R/'office_relative_summary.csv').set_index('model').loc[['office_relative_constant','office_relative_affine','office_relative_tone']]
fig,axs=plt.subplots(1,2,figsize=(10.8,4.4))
for ax,df,title in zip(axs,[a,b],['Four-point HT40: held records','Office HT20: R1 reference to R2']):
 x=np.arange(3);v=df.median_block_median_deg.to_numpy();w=df.median_block_p90_deg.to_numpy();ax.bar(x-.18,v,width=.36,label='Median of block medians');ax.bar(x+.18,w,width=.36,label='Median of block P90s',alpha=.7)
 for i,num in enumerate(v):ax.text(i-.18,num+.6,f'{num:.2f}',ha='center',fontsize=9)
 ax.set_xticks(x,labels,rotation=15);ax.set_ylabel('Phase-prediction residual (deg)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_ylim(0,max(w)*1.18)
axs[0].legend(fontsize=8);fig.suptitle('Reference prediction improves; these are not human AoA errors',fontsize=12);fig.tight_layout();fig.savefig(R/'phase_model_comparison.png',dpi=160);fig.savefig(R/'phase_model_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10.8,7.2),sharex=True,sharey=True)
e=pd.read_csv(R/'post_prediction_deviations.csv')
for ax,key,label in zip(axs.flat,[3,15,24,33],['+1 m','-1 m','+2 m','-2 m']):
 z=np.load(R/f'spectra_{key}.npz');p=np.load(R/f'projected_spectra_{key}.npz');ang=z['angle_deg'];truth=e[e.record==key].iloc[0].theory_angle_deg
 for model,title,color in [('baseline','Baseline','C0'),('pi_tone','Pi + per-tone template','C1')]:ax.plot(ang,10*np.log10(np.maximum(z['fused_'+model],1e-4)),label=title,color=color,lw=1.6)
 q=p['fused_pi_tone_tx_projection'];ax.plot(ang,np.where(p['observable_angle'],10*np.log10(np.maximum(q,1e-4)),np.nan),label='Template + TX projection',color='C2',ls='--')
 ax.axvline(truth,color='black',ls=':',label='Human direction: evaluation only');ax.set_title(f'{label}: record {key}');ax.set_ylim(-20,1);ax.grid(alpha=.2)
for ax in axs[-1]:ax.set_xlabel('RX-local angle parameter (deg)')
for ax in axs[:,0]:ax.set_ylabel('Normalized fused score (dB)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.025),ncol=2);fig.suptitle('Unassigned spectra: phase stability does not establish human-path identity',fontsize=12);fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(R/'spectral_comparison.png',dpi=150);fig.savefig(R/'spectral_comparison.pdf');plt.close(fig)
r=pd.read_csv(R/'record_rotations.csv');r=r[(r.model=='pi_tone')&(r.chain==2)].copy();r['phase_wrapped_360']=r.phase_change_deg%360;r.loc[r.phase_wrapped_360>300,'phase_wrapped_360']-=360
fig,ax=plt.subplots(figsize=(8,3.6));ax.scatter(np.arange(8),r.phase_wrapped_360,c=['C0']*4+['C1']*4,s=45);ax.axhline(0,color='gray',ls='--');ax.axhline(180,color='gray',ls='--');ax.set_xticks(np.arange(8),r.record.astype(str));ax.set(xlabel='Record index: development then retrospective evaluation',ylabel='Chain 2/1 phase change (deg)',title='Global chain-2 change is close to a 0 / 180-degree branch');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'chain2_pi_branches.png',dpi=150);fig.savefig(R/'chain2_pi_branches.pdf');plt.close(fig)
print('three figures saved')
