import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(R/'figures'/f'{name}.png',dpi=160);fig.savefig(R/'figures'/f'{name}.pdf');plt.close(fig)

def main():
    ids=[78,79,80,84,85,86];states=list(rat.MAP)
    ev=pd.read_csv(R/'results/events.csv');man=pd.read_csv(R/'empty_manifest.csv').set_index('index')
    fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    axs[0].axis('off');axs[0].text(.02,.80,'E1 (3 x 10 s)  →  E3 (3 x 15 s)',fontsize=16,weight='bold');axs[0].text(.02,.49,'Same scene and hardware session: user confirmed\nFilename dates do not describe the actual acquisition interval.',fontsize=12)
    axs[0].text(.02,.10,'Four occupied points: DIFFERENT hardware session\nE2 / E4: relationship unconfirmed; provenance audit only.',fontsize=12,color='#a13722')
    for j,i in enumerate(ids):
        z=ev[(ev['index']==i)&(ev.host_relative_s>=0)&(ev.host_relative_s<.65)]
        for k,s in enumerate(states):
            t=z[z.state==s].host_relative_s;axs[1].scatter(t,[j]*len(t),s=25,color=f'C{k}',label=s if j==0 else None)
    axs[1].set(yticks=range(6),yticklabels=[f'{man.loc[i,"capture_id"]}/{man.loc[i,"run"]}' for i in ids],xlabel='Within-run relative event time (s)',title='First 0.65 s: logged READY events; no claim of simultaneous observation')
    axs[1].legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,-.23));save(fig,'01_acquisition_and_timeline')

    p=pd.read_csv(R/'results/empty_pair_comparisons.csv');p=p[p.variant=='full']
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col,title in zip(axs,['spatial_distance','amplitude_distance','covariance_distance'],['Same-frequency spatial covariance','Normalized amplitude','Frequency-stacked covariance']):
        groups=['within_E1','within_E3','E1_E3'];ax.boxplot([p[p.comparison==g][col] for g in groups],labels=['Within E1','Within E3','E1 vs E3'],showfliers=True);ax.set(title=title,ylabel='Symmetric relative Frobenius distance');ax.tick_params(axis='x',rotation=15)
    fig.suptitle('All run pairs x 6 states; comparisons share recordings and are not independent trials');save(fig,'02_empty_invariant_comparison')

    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,state in zip(axs.flat,states):
        for i in ids:
            v=np.load(R/'cache'/f'{i:03d}_{state}.npz');z=v['pair_angle'][0];co=v['pair_coherence'][0];ax.plot(rat.F/1e6,np.where(co>=.5,np.rad2deg(z),np.nan),color='C0' if i<84 else 'C1',alpha=.65,label=man.loc[i,'capture_id'] if i in [78,84] else None)
        ax.set(title=state,xlabel='Subcarrier frequency offset (MHz)',ylabel='Pair 1-2 mean phase (deg)',ylim=(-185,185));ax.legend(fontsize=8)
    fig.suptitle('Per-state chain-pair phase: coherence >= 0.5; no phase alignment or unwrap');save(fig,'03_chain_phase_by_frequency')

    d=pd.read_csv(R/'results/reference_checks.csv');sel=load(R/'results/development_selection.json')['selected_variant'];z=d[d.variant==sel]
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col,limit,title in zip(axs,['chain_delay_p90_ns','chain_phase_p90_deg','prefix_predictive_correlation'],[5,15,.9],['Chain delay disagreement (ns)','Chain phase disagreement (deg)','Held-feature prefix correlation']):
        mat=z.pivot(index='index',columns='state',values=col).reindex(index=ids,columns=states).to_numpy();im=ax.imshow(mat,aspect='auto',cmap='viridis');ax.set(xticks=range(6),xticklabels=states,yticks=range(6),yticklabels=[str(i) for i in ids],title=title+'\nGate '+('>= ' if col.endswith('correlation') else '<= ')+str(limit));fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Selected representative-packet reference: 0/36 complete model gates pass');save(fig,'04_common_model_failures')

    cross=pd.read_csv(R/'results/cross_record_prediction.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for mode,col in [('A baseline','future_pred_corr_A'),('C candidate only','future_pred_corr_C_candidate')]:
        axs[0].plot(range(len(cross)),cross[col],'.',label=mode,alpha=.7)
    axs[0].axhline(.9,color='k',ls='--',label='0.90 reference gate');axs[0].set(xlabel='Cross-record state comparison',ylabel='Future held-feature correlation',title='48 cross-record diagnostics; no C alignment accepted');axs[0].legend(fontsize=8)
    b=d[d.variant==sel];axs[1].plot(range(len(b)),b.B_cov_change,'o',label='B - A');axs[1].plot(range(len(b)),b.C_cov_change,'x',label='Applied C - A');axs[1].set(xlabel='Run-state record',ylabel='Relative covariance difference',title='B is covariance-invariant; rejected C retains A');axs[1].legend();save(fig,'05_ABC_and_cross_prediction')

    nn=pd.read_csv(R/'results/empty_null_spectral_changes.csv');fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,g in enumerate(['within_E1','within_E3','E1_E3']):
        z=nn[nn.comparison==g];axs[0].scatter(np.full(len(z),k),z.excess_max/z.threshold,alpha=.5,s=20)
    axs[0].axhline(1,color='k',ls='--');axs[0].set(xticks=range(3),xticklabels=['Within E1','Within E3','E1 vs E3'],ylabel='Maximum positive spectral change / dev limit',title='Empty-only diagnostic; NOT a detector false-positive rate')
    v=np.load(R/'parameters/empty_spectra_G1.npz');delta=np.maximum(v['spectra'][3]-v['spectra'][0],0);im=axs[1].imshow(delta,origin='lower',extent=[-400,400,-90,90],aspect='auto',cmap='magma',vmin=0,vmax=1);axs[1].set(xlabel='Uncalibrated delay parameter (ns)',ylabel='Angle parameter (deg)',title='Fixed example: E3/run1 - E1/run1, G1');fig.colorbar(im,ax=axs[1],label='Positive normalized spectral change');save(fig,'06_empty_null_changes')

    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    sens=pd.read_csv(R/'results/empty_pair_comparisons.csv');sens=sens[sens.comparison=='E1_E3']
    for ax,col in zip(axs,['spatial_distance','covariance_distance']):
        means=sens.groupby('variant')[col].median();ax.barh(means.index,means.values);ax.set(xlabel='Median symmetric relative distance',title=col.replace('_',' '),xlim=(0,1.5))
    save(fig,'07_window_sensitivity')
    h=pd.read_csv(R/'results/human_stability_summary.csv');fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=['old+1','old-1','old+2','old-2'];pos=np.arange(4)
    for ax,same,title in zip(axs,[True,False],['Same-batch run pairs','Different-batch run pairs']):
        for j,mode in enumerate(['A','C']):
            z=h[(h.same_batch==same)&(h.method==mode)].set_index('point').reindex(labels);ax.bar(pos+(j-.5)*.35,z.median_spectrum_rms,width=.35,label=mode)
        ax.set(xticks=pos,xticklabels=labels,ylabel='Median normalized spectral RMS difference',title=title);ax.legend()
    fig.suptitle('Reuse of all 39 occupied runs: spectral stability is not human-path identification');save(fig,'08_human_existing_stability')
    print('Wrote eight diagnostic figures',flush=True)
if __name__=='__main__':main()
