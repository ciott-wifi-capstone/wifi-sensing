"""Only this post-prediction module reads withheld human geometry."""
from fit_inverse import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'scaled'
frozen=json.loads((O/'frozen_predictions.json').read_text(encoding='utf8'))
for name,digest in frozen.items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==digest
pred=pd.read_csv(O/'fused_predictions.csv')
local_y={0:-1,3:-1,12:1,15:1,21:-2,24:-2,30:2,33:2}
pred['reference_angle_deg']=pred.record.map({k:np.degrees(np.arctan2(y,2.01)) for k,y in local_y.items()})
pred['unassigned_peak_deviation_deg']=abs(pred.angle_deg-pred.reference_angle_deg)
pred['split']=pred.record.map({k:('calibration' if k in CAL else 'same_position_repeat' if k in [3,15] else 'withheld_position') for k in KEYS})
pred['physical_AoA_MAE_deg']=np.nan
pred.to_csv(O/'post_prediction_comparison.csv',index=False)
summaries=[]
for split,keys in [('calibration',CAL),('same_position_repeat',[3,15]),('withheld_position',[21,24,30,33]),('all_noncalibration',[3,15,21,24,30,33]),('previous_four_record_comparison',[3,15,24,33])]:
 for mode,g in pred[pred.record.isin(keys)].groupby('model',sort=False):
  e=g.unassigned_peak_deviation_deg.to_numpy()
  summaries.append(dict(split=split,model=mode,records=len(e),mean_unassigned_peak_deviation_deg=np.mean(e),median_deviation_deg=np.median(e),p90_deviation_deg=np.quantile(e,.9),max_deviation_deg=max(e),fraction_within8=np.mean(e<=8),spectral_output_rate=1.,human_validated_output_rate=0.,physical_AoA_MAE_deg=np.nan))
summary=pd.DataFrame(summaries);summary.to_csv(O/'descriptive_summary.csv',index=False)
conf=pd.read_csv(O/'dominant_between_positions.csv')
conf.groupby('hypothesis').between_calibration_conflict_deg.agg(['median','mean','max']).to_csv(O/'dominant_conflict_summary.csv')
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})
labels={'baseline':'Baseline','pi_only':'Pi correction only','dominant_tx':'Dominant TX phase','dominant_human':'Dominant human phase','tx_only':'TX subspace fit','human_only':'Human subspace fit','tx_human':'TX + human fit','tx_mirror_placebo':'TX + mirror placebo'}
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,split,title in zip(axs,['same_position_repeat','withheld_position'],['Unfitted repeats at calibration positions (n=2)','Positions excluded from calibration (n=4)']):
 g=summary.query('split==@split').set_index('model').loc[list(labels)]
 ax.barh(np.arange(len(g)),g.mean_unassigned_peak_deviation_deg,color=['#527a98' if k=='baseline' else '#ba6380' if k=='tx_mirror_placebo' else '#4c9d91' for k in labels])
 ax.set_yticks(np.arange(len(g)),labels.values());ax.invert_yaxis()
 for i,val in enumerate(g.mean_unassigned_peak_deviation_deg):ax.text(val+.7,i,f'{val:.1f}',va='center',fontsize=8)
 ax.set_xlim(0,max(g.mean_unassigned_peak_deviation_deg)+10);ax.set_xlabel('Mean unassigned-peak deviation from human angle (deg)');ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
fig.suptitle('Shared phase calibration: retrospective diagnostics, not validated human AoA',fontsize=12)
fig.tight_layout();fig.savefig(O/'inverse_comparison.png',dpi=170);fig.savefig(O/'inverse_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(11,6.2),sharex=True,sharey=True)
sel=['baseline','tx_only','human_only','tx_human','tx_mirror_placebo'];colors=['#444444','#3082ae','#b48a2b','#27866a','#b64983']
for ax,key in zip(axs.flat,[3,15,21,24,30,33]):
 z=np.load(O/f'predictions_{key}.npz');angles=z['angle_deg']
 for mode,color in zip(sel,colors):ax.plot(angles,10*np.log10(np.maximum(z['fused_'+mode],1e-8)),label=labels[mode],color=color,lw=1.2,alpha=.9)
 truth=float(pred[pred.record==key].reference_angle_deg.iloc[0]);ax.axvline(truth,color='#777777',ls=':',lw=1.5)
 ax.set_title(f'Record {key}: reference {truth:+.2f} deg');ax.set_ylim(-15,1);ax.grid(alpha=.15)
for ax in axs[-1]:ax.set_xlabel('Angle parameter (deg)')
for ax in axs[:,0]:ax.set_ylabel('Normalized fused spectrum (dB)')
handles,ls=axs[0,0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
fig.suptitle('G1-G4 noncoherent fusion; dotted reference added only after prediction',fontsize=12)
fig.tight_layout(rect=[0,.08,1,.96]);fig.savefig(O/'inverse_spectra.png',dpi=170);fig.savefig(O/'inverse_spectra.pdf');plt.close(fig)
fits=pd.read_csv(O/'fit_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(11,3.7))
ph=pd.read_csv(O/'shared_phase_candidates.csv')
for mode,color in zip(list(MODES),colors[1:]):
 g=ph[ph.model==mode];axs[0].plot(g.antenna,g.phase_deg,marker='o',label=labels[mode],color=color)
axs[0].set(xlabel='Physical antenna number',ylabel='Fitted phase relative to antenna 1 (deg)',xticks=np.arange(1,13),title='Hypothesis-dependent phase candidates')
axs[0].legend(fontsize=7,ncol=2);axs[0].grid(alpha=.2)
for mode,g in pd.read_csv(O/'multistart_fits.csv').groupby('model'):
 i=list(MODES).index(mode);axs[1].scatter(np.full(len(g),i),g.cost,alpha=.6,s=25)
axs[1].set_xticks(range(4),['TX','Human','TX + human','TX + mirror'],rotation=15);axs[1].set(ylabel='Mean normalized noise-subspace loss',title='Eight frozen initializations per hypothesis');axs[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'inverse_identifiability.png',dpi=170);fig.savefig(O/'inverse_identifiability.pdf');plt.close(fig)
print(summary[summary.split.isin(['withheld_position','all_noncalibration'])].to_string(index=False))
print('Conflicts:',conf.groupby('hypothesis').between_calibration_conflict_deg.agg(['median','max']).to_string())
print(pred.pivot(index='record',columns='model',values='angle_deg').to_string())
