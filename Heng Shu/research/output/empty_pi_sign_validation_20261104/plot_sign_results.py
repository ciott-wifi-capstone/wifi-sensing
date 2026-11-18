from validate_sign import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
d=pd.read_csv(O/'comparisons.csv');p=pd.read_csv(O/'phase_residuals.csv')
sel=d[d.run==3];pairs=['E1_E3','E1_E2','E3_E4'];states=ST
fig,axs=plt.subplots(1,2,figsize=(10.6,4.2))
x=np.arange(3)
for j,(mode,label,color) in enumerate([('raw','Original','#8a93a6'),('fixed_pi','Frozen pi signs','#268a83')]):
 a=sel[sel['mode']==mode].groupby('pair').covariance_distance.median().reindex(pairs)
 b=sel[sel['mode']==mode].groupby('pair').false_peaks.sum().reindex(pairs)
 for ax,vals in zip(axs,[a,b]):
  bars=ax.bar(x+(j-.5)*.34,vals,.34,label=label,color=color)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.3f}' if ax is axs[0] else f'{v:.0f}',ha='center',va='bottom',fontsize=9)
axs[0].set_ylabel('Median relative covariance distance');axs[1].set_ylabel('Empty local peaks above fixed threshold')
for ax in axs:
 ax.set_xticks(x,['E1 → E3','E1 → E2','E3 → E4']);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.15)
axs[0].legend();fig.suptitle('Complete third-run validation: one frozen sign vector per capture pair')
fig.tight_layout();fig.savefig(O/'pi_effect_run3.png',dpi=170);fig.savefig(O/'pi_effect_run3.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
for ax,pair in zip(axs,pairs):
 a=p[(p.pair==pair)&(p.run==3)&(p.frequency_role=='withheld_frequency')]
 for mode,label in [('raw','Original'),('fixed_pi','Frozen pi signs')]:
  vals=a[a['mode']==mode].set_index('state').loc[ST].pair_phase_median_deg
  ax.plot(range(6),vals,'o-',label=label)
 ax.set_title(pair.replace('_',' → '));ax.set_xticks(range(6),ST);ax.grid(alpha=.2)
axs[0].set_ylabel('Pair-phase residual median (deg)');axs[0].legend()
fig.suptitle('Held-out frequency slots; G3/G4/C1/C2 also excluded from sign fitting')
fig.tight_layout();fig.savefig(O/'held_phase_residuals.png',dpi=170);fig.savefig(O/'held_phase_residuals.pdf');plt.close(fig)
summary=p[(p.run==3)&(p.frequency_role=='withheld_frequency')].groupby(['pair','mode']).pair_phase_median_deg.median()
summary.to_csv(O/'held_phase_summary.csv');print(summary.to_string())
