from analyze import *

df=pd.read_csv(OUT/'runs.csv');z=np.load(OUT/'power_profiles.npz');fs=z['profiles'];ids=z['run_indices']
conflict=df.duplicated('sha256',keep=False)&(df.label=='kong')
df['analysis_group']=df.label
df.loc[(df.label=='kong')&df.batch.str.startswith('20261003'),'analysis_group']='empty Oct03'
df.loc[(df.label=='kong')&df.batch.str.startswith('20261018'),'analysis_group']='empty Oct18'
df.loc[conflict,'analysis_group']='duplicate A (not empty)'
df['use_for_comparison']=~conflict
df['complete_window']=df.trailing_s<=.1
csv('curated_runs.csv',df.to_dict('records'))
clean=df[~conflict];labels=list(clean.analysis_group.unique())
group=clean.groupby('analysis_group',sort=False).agg(runs=('run','size'),batches=('batch','nunique'),frames=('frames','sum'),duration_s=('duration_s','median'),pps=('window_pps','median'),phase_sd_deg=('phase_sd_deg','median'),shape_cv=('shape_cv','median'),truncated=('complete_window',lambda s:int((~s).sum())))
group.to_csv(OUT/'curated_groups.csv',encoding='utf-8-sig')
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for ax,col,title in zip(axes,['window_pps','phase_sd_deg','shape_cv'],['Received packets / scheduled second','Within-burst phase SD (degrees)','Within-burst normalized power CV']):
    ax.boxplot([clean[clean.analysis_group==s][col].dropna() for s in labels],tick_labels=labels)
    ax.set_title(title);ax.tick_params(axis='x',rotation=60);ax.grid(axis='y',alpha=.25)
fig.suptitle('Curated diagnostics: duplicate assigned to A; empty room split by date');fig.tight_layout();fig.savefig(OUT/'quality_curated.png',dpi=180);plt.close(fig)
bursts=pd.read_csv(OUT/'bursts.csv')
counts=bursts.groupby(['label','batch','run','state']).size().unstack(fill_value=0).reset_index()
coverage=df.merge(counts,on=['label','batch','run']);coverage=coverage[coverage.use_for_comparison].copy()
coverage['min_state_windows']=coverage[STATES].min(axis=1)
csv('state_window_counts.csv',coverage.to_dict('records'))
fdf=df.iloc[ids].reset_index(drop=True)
fdf=fdf.merge(counts,on=['label','batch','run'],how='left')
keep=fdf.use_for_comparison.to_numpy()&(fdf[STATES].min(axis=1).to_numpy()>=10)
fs=fs[keep];fdf=fdf[keep].reset_index(drop=True)
csv('profile_included_runs.csv',fdf.to_dict('records'))
flat=fs.reshape(len(fs),-1);distance=np.sqrt(np.mean((flat[:,None]-flat[None,:])**2,axis=2))
fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(distance,cmap='viridis',vmin=0);centers=[];pos=0
featurelabels=list(fdf.analysis_group.unique())
for label in featurelabels:
    count=int((fdf.analysis_group==label).sum());centers.append(pos+(count-1)/2);pos+=count;ax.axhline(pos-.5,color='white',lw=.6);ax.axvline(pos-.5,color='white',lw=.6)
ax.set_xticks(centers,featurelabels,rotation=60);ax.set_yticks(centers,featurelabels);ax.set_title('Normalized power-profile RMS distance (>=10 windows per state)\nA excluded: insufficient G4 coverage; not localization error');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'profile_distance_curated.png',dpi=180);plt.close(fig)
gcounts=coverage.groupby('analysis_group',sort=False)[STATES].median()
fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(gcounts.to_numpy(),vmin=0,vmax=100,cmap='YlGnBu')
for i in range(len(gcounts)):
    for j in range(6):ax.text(j,i,f'{gcounts.iloc[i,j]:g}',ha='center',va='center',color='white' if gcounts.iloc[i,j]>60 else 'black')
ax.set_xticks(range(6),STATES);ax.set_yticks(range(len(gcounts)),gcounts.index);ax.set_title('Median usable windows per run / switching state\nOld: about 66 per state; new: about 100; minimum 5 packets per window');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'state_coverage.png',dpi=180);plt.close(fig)
pairs=[]
for i in range(len(fs)):
    for j in range(i):
        a,b=fdf.iloc[i],fdf.iloc[j]
        typ='same_point_same_batch' if a.analysis_group==b.analysis_group and a.batch==b.batch else 'same_point_other_batch' if a.analysis_group==b.analysis_group else 'different_point'
        pairs.append(dict(group1=a.analysis_group,group2=b.analysis_group,type=typ,rms=float(distance[i,j])))
csv('curated_pair_distances.csv',pairs)
# Sensitivity: central 5 ms windows across alignment offsets for one complete run per group.
sens=[]
for label in labels:
    r=clean[clean.analysis_group==label].iloc[0];p=Path(r.path);c,host,nic=decode(p);t=(host-int(r.start_ns))/1e9
    a=align(p.parent/'events.log',int(r.start_ns)/1e9);starts,states,guard=a
    for offset in [-10,-5,0,5,10]:
        stats=[]
        for start in starts[:-1]:
            lo=start+.010+offset/1000;hi=lo+.005
            if lo<0 or hi>r.duration_s:continue
            i,j=np.searchsorted(t,[lo,hi])
            if j-i<3:continue
            x=c[i:j];u=x[:,0]*x[:,1].conj();good=np.all(np.abs(u)>0,axis=0);u=u/(np.abs(u)+1e-12)
            if good.any():stats.append(float(np.median(np.sqrt(-2*np.log(np.clip(np.abs(u.mean(0))[good],1e-9,1)))*180/np.pi)))
        sens.append(dict(group=label,offset_ms=offset,phase_sd_deg=float(np.median(stats)),windows=len(stats)))
csv('timing_sensitivity.csv',sens)
print(group.to_string());print(pd.DataFrame(sens).pivot(index='group',columns='offset_ms',values='phase_sd_deg').round(2).to_string())
# User-supplied geometry; no estimated positions.
fig,axs=plt.subplots(1,2,figsize=(12,5));ax=axs[0]
ax.plot([0,3.65],[0,0],'kD-',label='RX / TX');ax.text(0,-.2,'RX');ax.text(3.65,-.2,'TX')
for k,xy in COORD.items():
    if xy:ax.scatter(*xy,c='tab:blue' if k.startswith('old') else 'tab:orange');ax.annotate(k,xy,xytext=(5,5),textcoords='offset points')
ax.set(xlabel='x (m)',ylabel='y (m)',title='Confirmed measurement coordinates',xlim=(-.3,4),ylim=(-2.4,2.4));ax.set_aspect('equal');ax.grid(alpha=.25)
geo=pd.read_csv(OUT/'geometry_theory_only.csv');axs[1].bar(geo.label,geo.excess_ns,color=['tab:blue']*4+['tab:orange']*4);axs[1].set(ylabel='Excess path delay (ns)',title='Geometry prediction ONLY: (|P-RX|+|P-TX|-3.65)/c');axs[1].tick_params(axis='x',rotation=45);axs[1].grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'geometry.png',dpi=180);plt.close(fig)
