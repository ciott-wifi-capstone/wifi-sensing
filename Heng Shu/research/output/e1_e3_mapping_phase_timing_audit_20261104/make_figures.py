from audit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})
pairs=['E1_E3','E1_E2','E3_E4'];methods=['raw','pi','constant_phase','phase_slope','phase_slope_gain']
p=pd.read_csv(R/'phase_model_validation.csv').query('run==3 and frequency_scope=="all"')
g=pd.read_csv(R/'gain_phase_validation.csv').query('run==3');g['model']='phase_slope_gain'
combined=pd.concat([p,g],ignore_index=True)
summary=combined.groupby(['pair','model'])[['phase_median_deg','amplitude_distance','spatial_distance']].median()
counts=pd.read_csv(R/'empty_candidate_recheck.csv').query('run==3').groupby(['pair','model']).stable_candidates.sum()
oldcounts=pd.read_csv(O/'same_frequency_spatial_background_20261104/empty_summary.csv').query('run==3 and mode=="raw"')
for row in oldcounts.itertuples():counts.loc[(row.source+'_'+row.target,'raw')]=row.stable_false_peaks
fig,axs=plt.subplots(1,3,figsize=(12,4))
colors=['#b54431','#3470ad','#18886c']
for pair,color in zip(pairs,colors):
 z=summary.loc[pair].reindex(methods)
 axs[0].plot(range(5),z.spatial_distance,'o-',label=pair.replace('_',' -> '),c=color)
 axs[1].plot(range(5),z.phase_median_deg,'o-',c=color)
 axs[2].plot(range(5),counts.loc[pair].reindex(methods),'o-',c=color)
axs[0].axhline(.25,color='black',ls=':',lw=1)
axs[0].set_ylabel('Median spatial covariance distance');axs[0].legend(fontsize=8)
axs[1].set_ylabel('Median pair-phase residual (deg)');axs[1].axhline(15,color='black',ls=':',lw=1)
axs[2].set_ylabel('Stable empty direction candidates')
for ax in axs:
 ax.set_xticks(range(5),['Raw','pi sign','Constant\nphase','Phase +\nslope','+ chain\ngain']);ax.grid(alpha=.15)
fig.suptitle('Third-run validation: one RF-chain response does not explain E1 -> E3')
fig.tight_layout();fig.savefig(R/'phase_transfer_validation.png',dpi=180);fig.savefig(R/'phase_transfer_validation.pdf');plt.close(fig)
ps=pd.read_csv(R/'phase_by_state.csv').query('pair=="E1_E3" and run==3')
heat=ps.pivot(index='state',columns='pair_ch',values='relative_phase_deg').reindex(ST)
t=pd.read_csv(R/'timing_sensitivity.csv').query('index==86 and model=="phase_slope"')
tm=t.groupby('variant')[['amplitude_distance','spatial_distance']].median().reindex(['baseline','guard4','guard6','guard9','guard6_shift-3','guard6_shift+3'])
fig,axs=plt.subplots(1,2,figsize=(10,4))
im=axs[0].imshow(heat,cmap='twilight_shifted',vmin=-180,vmax=180,aspect='auto')
axs[0].set_xticks(range(3),heat.columns);axs[0].set_yticks(range(6),ST);axs[0].set_title('E1 -> E3 relative pair phases (deg)')
for i in range(6):
 for j in range(3):axs[0].text(j,i,f'{heat.iloc[i,j]:.0f}',ha='center',va='center',color=('black' if abs(heat.iloc[i,j])<80 else 'white'),fontsize=8)
fig.colorbar(im,ax=axs[0],fraction=.05)
axs[1].plot(range(6),tm.spatial_distance,'o-',label='Spatial distance')
axs[1].plot(range(6),tm.amplitude_distance,'s-',label='Amplitude distance')
axs[1].set_xticks(range(6),['Original','4 ms','6 ms','9 ms','6 ms\n-3 ms shift','6 ms\n+3 ms shift'])
axs[1].set_ylim(0,1);axs[1].set_title('Timing sensitivity; frozen phase/slope model')
axs[1].set_ylabel('Median distance, third run');axs[1].legend();axs[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'phase_state_and_timing.png',dpi=180);fig.savefig(R/'phase_state_and_timing.pdf');plt.close(fig)
wi=pd.read_csv(R/'occupied_within_invariants.csv')
agg=wi.groupby(['batch_first','index'])[['amplitude_distance','spatial_distance','phase_median_deg']].median()
fig,axs=plt.subplots(1,3,figsize=(10,4))
for ax,col,title in zip(axs,['amplitude_distance','spatial_distance','phase_median_deg'],['Amplitude distance','Spatial covariance distance','Pair-phase residual (deg)']):
 a=np.array([[agg.loc[(first,first+off),col] for off in [1,2]] for first in [0,12,21,30]])
 ax.imshow(a,cmap='Blues',vmin=0,vmax=a.max()*1.25,aspect='auto')
 ax.set_xticks([0,1],['Run 2','Run 3']);ax.set_yticks(range(4),['+1 m','-1 m','+2 m','-2 m']);ax.set_title(title)
 for i in range(4):
  for j in range(2):ax.text(j,i,(f'{a[i,j]:.1f}' if col=='phase_median_deg' else f'{a[i,j]:.3f}')+('*' if i==0 and j==1 else ''),ha='center',va='center',fontsize=10,color=('white' if a[i,j]>.6*a.max() else 'black'))
fig.suptitle('Within-batch occupied repeatability relative to run 1')
fig.text(.5,.01,'* +1 m run 3 is truncated (~3.15 s); values are repeatability metrics, not human AoA errors.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(R/'within_batch_repeatability.png',dpi=180);fig.savefig(R/'within_batch_repeatability.pdf');plt.close(fig)
summary.to_csv(R/'phase_transfer_summary.csv')
print('figures written')
