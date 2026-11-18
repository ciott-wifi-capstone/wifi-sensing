"""Scientific plots from recorded results. No truth-based peak selection."""
import json,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
GROUPS=['old+1','old-1','old+2','old-2'];LABELS=['+1 m','-1 m','+2 m','-2 m']
S=pd.read_csv(R/'results/state_summary.csv');T=pd.read_csv(R/'results/temporal_paired.csv')
def save(fig,name):
    fig.savefig(R/'figures'/f'{name}.png',dpi=170);fig.savefig(R/'figures'/f'{name}.svg');plt.close(fig)
def main():
    h=S[(S.role=='held_out')&S.group.isin(GROUPS)]
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,cols,title,ylabel in [(axs[0],['pred_delay_iqr_A_ns','pred_delay_iqr_C_ns'],'Relative timing on withheld entries','Residual delay IQR (ns)'),
                                 (axs[1],['pred_phase_sd_B_deg','pred_phase_sd_C_deg'],'Phase: joint correction vs CPE-only','Circular SD (degrees)')]:
        for j,(col,color,lab) in enumerate(zip(cols,['#8a969f','#007f87'],['A / B timing' if ax==axs[0] else 'B: CPE only','C candidate: joint'])):
            for i,g in enumerate(GROUPS):
                vals=h[h.group==g][col].to_numpy();xx=np.full(len(vals),i+(j-.5)*.28)
                ax.scatter(xx,vals,s=17,alpha=.45,color=color,label=lab if i==0 else None)
                ax.plot([xx[0]-.09,xx[0]+.09],[np.median(vals)]*2,color=color,lw=3)
        ax.set(xticks=range(4),xticklabels=LABELS,title=title,ylabel=ylabel,yscale='log');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('72 run-state checks in 12 held-out occupied runs; disjoint fit/test entries\nConditional-model consistency, NOT physical timing accuracy',fontsize=12)
    save(fig,'01_withheld_feature_validation')
    groups=GROUPS+['empty Oct03','A','B','C','D','empty Oct18']
    rates=[];labels=[]
    for g in groups:
        z=S[S.group==g];ok=int((z.common_model_pass==True).sum());computed=int((z.status=='computed').sum())
        rates.append(ok/max(computed,1));labels.append(f'{g}\n{ok}/{computed}')
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');ax.bar(range(len(groups)),rates,color=['#007f87']*4+['#a3adb5']*6)
    ax.set(xticks=range(len(groups)),xticklabels=labels,ylim=(0,1.12),ylabel='Fraction passing prefix common-model check',title='Applicability: primary four points vs empty / supplementary captures')
    ax.text(.5,.94,'A: 3 missing G4 states recorded separately; failed states remain uncorrected',transform=ax.transAxes,ha='center',fontsize=9)
    save(fig,'02_applicability')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    t=T[(T.role=='held_out')&T.group.isin(GROUPS)]
    for ax,col,title in zip(axs,['block_cov_median_ratio','block_spec_median_ratio'],['Temporal covariance variation','Temporal MUSIC spectrum variation']):
        vals=[t[t.group==g][col].to_numpy() for g in GROUPS];ax.boxplot(vals,tick_labels=LABELS,showfliers=True)
        ax.axhline(1,color='#b55b46',ls='--');ax.set(title=title,ylabel='C / A, lower is more stable');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Four chronological blocks per run/state; mixed state-level effects\nA spectral stability change is not a positioning result',fontsize=12)
    save(fig,'03_temporal_stability')
    # Equal-weight median of run-normalised spectra from all three held-out G1 runs per point.
    fig,axs=plt.subplots(4,3,figsize=(11,12),layout='constrained')
    for i,g in enumerate(GROUPS):
        ids=h[(h.group==g)&(h.state=='G1')]['index'].astype(int)
        for j,met in enumerate(['A','B','C']):
            sp=np.median([np.load(R/'packets'/f'{idx:03d}_G1.npz')['spectrum_'+met] for idx in ids],axis=0)
            im=axs[i,j].pcolormesh(TAU,ANG,10*np.log10(np.maximum(sp/sp.max(),1e-6)),vmin=-15,vmax=0,cmap='viridis',shading='auto')
            axs[i,j].set(title=f'{LABELS[i]}, {met}',xlabel='Uncalibrated / run-gauge delay (ns)',ylabel='RX-local angle parameter (deg)')
    fig.colorbar(im,ax=axs,shrink=.65,label='Median normalised MUSIC (dB)')
    fig.suptitle('A baseline; B CPE only; C gated joint correction\nPer-run references differ. No human path or position assigned.',fontsize=12)
    save(fig,'04_music_comparison')
    x=pd.read_csv(R/'results/between_run_comparisons.csv');x=x[(x.same_batch==False)&x.group.isin(GROUPS)]
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,col,title in zip(axs,['cov_distance','spectrum_rms'],['Cross-batch covariance distance','Cross-batch spectrum RMS difference']):
        ax.boxplot([x[x.method==m][col].to_numpy() for m in ['A','B','C']],tick_labels=['A','B','C'],showfliers=False)
        ax.set(title=title,ylabel='Dimensionless distance');ax.grid(axis='y',alpha=.2)
    fig.suptitle('All primary same-point run pairs from different batches\nDescriptive dependent pairs; no gauge alignment or independent-trial claim',fontsize=12)
    save(fig,'05_cross_batch_limits')
    # Representative chosen by chronology, not a flattering spectrum or measured truth.
    idx=3;state='G1';v=np.load(R/'packets'/f'{idx:03d}_{state}.npz');d=json.loads((R/'results'/f'run_{idx:03d}.json').read_text())
    sp=importlib.util.spec_from_file_location('dec',R/'provenance/analyze.py');dec=importlib.util.module_from_spec(sp);sp.loader.exec_module(dec)
    raw,_,_=dec.decode(Path(d['path']));a=unit(raw[v['packet_id']].astype(complex));b=apply(a,v['B_tau_ns'],v['B_psi_rad']);c=a.copy();ok=v['C_applied'];c[ok]=apply(a[ok],v['C_tau_ns'][ok],v['C_psi_rad'][ok])
    time=(v['host_ns']-v['host_ns'][0])/1e9;pred=v['prediction'];times=time[pred[:,0].astype(int)]
    fig,axs=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    for met,z,color in [('A',a,'#8a969f'),('B',b,'#d68d34'),('C',c,'#007f87')]:
        # Phase on ONE observed chain/tone, not the optimised correlation itself.
        axs[0].scatter(time,np.angle(z[:,0,0]*z[0,0,0].conj(),deg=True),s=5,alpha=.55,color=color,label=met)
    axs[0].set(ylabel='Tone phase vs first packet (deg)',xlabel='Actual elapsed time (s)',title='Representative +1 m held-out run 1, G1, chain 1 / first frequency group');axs[0].legend(ncols=3)
    axs[1].scatter(times,pred[:,2],s=10,color='#8a969f',label='A/B: independently fitted test-entry timing')
    axs[1].scatter(times,pred[:,3],s=10,color='#007f87',label='After train-entry joint correction: test residual')
    axs[1].set(xlabel='Actual elapsed time (s)',ylabel='Relative delay parameter (ns)');axs[1].legend(fontsize=8)
    fig.suptitle('Real packet timestamps; no interpolation or simultaneous-array synthesis',fontsize=12);save(fig,'06_packet_example')
    empty=pd.read_csv(R/'results/between_run_comparisons.csv')
    empty=empty[empty.group.str.startswith('empty')]
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,col,title in zip(axs,['cov_distance','spectrum_rms'],['Empty vs empty: covariance','Empty vs empty: MUSIC spectrum']):
        vals=[empty[(empty.group==g)&(empty.same_batch==same)&(empty.method=='A')][col].to_numpy() for g in ['empty Oct03','empty Oct18'] for same in [True,False]]
        ax.boxplot(vals,tick_labels=['Oct03\nwithin batch','Oct03\ncross batch','Oct18\nwithin batch','Oct18\ncross batch'])
        ax.set(title=title,ylabel='Dimensionless distance');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Empty captures fail the common-model gate: actual C equals A\nDates remain separate; dependent pairs, no physical ground truth',fontsize=12)
    save(fig,'07_empty_reference_limits')
    rows=[]
    for n,src,note in [
        ('01_withheld_feature_validation','state_summary.csv; packets/* prediction','Disjoint features on future packets; candidate model test, not true delay error'),
        ('02_applicability','state_summary.csv','All processed records and failures; A-D only selected first batches'),
        ('03_temporal_stability','temporal_paired.csv','Exploratory holdout, four chronological blocks'),
        ('04_music_comparison','packets/* spectrum_A/B/C','G1 median over three held-out runs; independent arbitrary run gauges'),
        ('05_cross_batch_limits','between_run_comparisons.csv','Dependent descriptive run pairs, no phase or delay alignment'),
        ('06_packet_example','run_003.json; packets/003_G1.npz; original CSI','Chronological representative; one actual tone and withheld-feature timing'),
        ('07_empty_reference_limits','between_run_comparisons.csv','Separate dates, dependent pairs; failed C states identical to A')]:
        rows.append(dict(figure=n,source=src,limit=note))
    pd.DataFrame(rows).to_csv(R/'figures/figure_sources.csv',index=False)
if __name__=='__main__':main()
