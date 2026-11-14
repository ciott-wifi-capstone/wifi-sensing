from spatial_model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})
d=pd.read_csv(R/'empty_validation.csv')
choices=[('E1','E1','within'),('E2','E2','within'),('E3','E3','within'),('E4','E4','within'),('E1','E2','fixed_pi'),('E3','E4','fixed_pi'),('E1','E3','fixed_pi')]
a=[];labels=[];false=[]
for src,tar,mode in choices:
 z=d[(d.source==src)&(d.target==tar)&(d.run==3)&(d['mode']==mode)].set_index('state').loc[ST]
 a.append((z.distance/z.distance_threshold).to_numpy());labels.append(src+' → '+tar+(' (pi)' if mode=='fixed_pi' else ''))
 false.append(int(z.stable_candidates.sum()))
fig,axs=plt.subplots(1,2,figsize=(10.5,4.5),gridspec_kw={'width_ratios':[3,2]})
arr=np.array(a);im=axs[0].imshow(arr,cmap='viridis',norm=LogNorm(vmin=.3,vmax=max(3,arr.max())),aspect='auto')
axs[0].set_xticks(range(6),ST);axs[0].set_yticks(range(len(labels)),labels);axs[0].set_title('Spatial distance / frozen threshold')
for i in range(len(labels)):
 for j in range(6):axs[0].text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white',fontsize=8)
axs[1].barh(range(len(labels)),false,color='#238a83');axs[1].invert_yaxis();axs[1].set_yticks(range(len(labels)),labels);axs[1].set_xlabel('Stable empty directional candidates');axs[1].set_xlim(0,max(false)+3)
for i,v in enumerate(false):axs[1].text(v+.2,i,str(v),va='center')
fig.suptitle('Third-run valid windows: covariance change need not yield a direction')
fig.tight_layout();fig.savefig(R/'empty_stability.png',dpi=170);fig.savefig(R/'empty_stability.pdf');plt.close(fig)
thresholds=pd.read_csv(R/'occupied_source_thresholds.csv').query('source=="E1"').set_index('state')
labels_by_key={0:'+1 m',3:'+1 m',12:'-1 m',15:'-1 m',21:'+2 m',24:'+2 m',30:'-2 m',33:'-2 m'}
keys=[0,3,12,15,21,24,30,33];fig,axs=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True)
for ax,key in zip(axs.flat,keys):
 for state in ['G1','G2','G3','G4']:
  z=np.load(R/'spectra'/f'occupied_E1_{key}_{state}_pi_hypothesis.npz')
  y=10*np.log10(np.maximum(z['scores'][0]/thresholds.loc[state,'score_threshold'],1e-12))
  ax.plot(ANG,y,label=state,lw=1.3)
 ax.axhline(0,color='black',ls=':',lw=1);ax.set_title(labels_by_key[key]+' | record '+str(key));ax.grid(alpha=.15);ax.set_xticks([-90,-45,0,45,90])
for ax in axs[-1]:ax.set_xlabel('Uncalibrated direction parameter (deg)')
for ax in axs[:,0]:ax.set_ylabel('Score / empty threshold (dB)')
handles,lab=axs[0,0].get_legend_handles_labels();fig.legend(handles,lab,ncol=4,loc='lower center')
fig.suptitle('E1 reference: repeated occupied changes do not agree across G subarrays',fontsize=12)
fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(R/'occupied_subarray_scores.png',dpi=170);fig.savefig(R/'occupied_subarray_scores.pdf');plt.close(fig)
c=pd.read_csv(R/'occupied_candidates.csv')
v=c[(c.source=='E1')&(c['mode']=='pi_hypothesis')&c.stable&c.state.str.startswith('G')]
table=v.groupby(['record','state']).angle_parameter_deg.apply(lambda t:','.join(f'{a:+.0f}' for a in t)).unstack().reindex(index=keys,columns=['G1','G2','G3','G4']).fillna('none')
table.to_csv(R/'stable_G_peaks_E1.csv');print(table.to_string())
# Same-position repeats vs other occupied records: diagnostic of change specificity, no classification.
model=np.load(R/'occupied_frozen_signs.json') if False else json.loads((R/'occupied_frozen_signs.json').read_text())
reps={(0,3),(12,15),(21,24),(30,33)};mat={}
for key in keys:
 sign=np.array(model[f'E1_{key}']['sign'])
 for state in ['G1','G2','G3','G4']:
  x=np.load(S/'packets'/f'{key}_{state}.npz')['x'][64:]*sign[None,:,None];mat[(key,state)]=spatial(x)
rows=[]
for i,a in enumerate(keys):
 for b in keys[i+1:]:
  for state in ['G1','G2','G3','G4']:
   rows.append(dict(record_a=a,record_b=b,state=state,same_position_repeat=(a,b) in reps,distance=dist(mat[(a,state)],mat[(b,state)])))
pd.DataFrame(rows).to_csv(R/'occupied_pair_spatial_distances.csv',index=False)
print(pd.DataFrame(rows).groupby('same_position_repeat').distance.agg(['median','min','max']).to_string())
