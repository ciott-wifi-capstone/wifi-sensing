"""Reuse existing spectra, never select a human path or fit using ground truth."""
import itertools
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *

def main():
    man=pd.read_csv(R/'primary_manifest.csv');old=pd.read_csv(OLD/'results/state_summary.csv');rows=[];pairs=[];sources=[];point=[]
    root=R/'figures/human_atlas';root.mkdir(exist_ok=True)
    coord={'old+1':(1.64,1),'old-1':(1.64,-1),'old+2':(1.64,2),'old-2':(1.64,-2)}
    for group,mm in man.groupby('label',sort=False):
        spectra={};covs={};peaks={}
        pdfpath=root/(group.replace('+','plus').replace('-','minus')+'.pdf')
        with PdfPages(pdfpath) as pdf:
            for row in mm.itertuples(index=False):
                # Raw hashes establish lineage; only stored outputs are processed here.
                assert sha(row.path)==row.sha256
                fig,axes=plt.subplots(6,2,figsize=(10,13),sharex=True,sharey=True,layout='constrained')
                for k,state in enumerate(rat.MAP):
                    path=OLD/'packets'/f'{row.index:03d}_{state}.npz';v=np.load(path)
                    sources.append(dict(index=row.index,state=state,raw_path=row.path,raw_sha256=row.sha256,existing_output=str(path),existing_output_sha256=sha(path),reuse='original cached A/C spectra; no raw re-estimation'))
                    for j,mode in enumerate(['A','C']):
                        sp=v['spectrum_'+mode];spectra[(row.index,state,mode)]=sp;covs[(row.index,state,mode)]=v['cov_'+mode];peak=rat.peak(sp);peaks[(row.index,state,mode)]=peak
                        ax=axes[k,j];im=ax.imshow(10*np.log10(np.maximum(sp,1e-3)),origin='lower',extent=[rat.TAU[0],rat.TAU[-1],rat.ANG[0],rat.ANG[-1]],aspect='auto',vmin=-30,vmax=0,cmap='viridis',interpolation='nearest')
                        ax.set_title(f'{state} | {mode}: '+('baseline' if mode=='A' else 'existing gated correction'),fontsize=9)
                        if j==0:ax.set_ylabel('Angle parameter (deg)',fontsize=8)
                        ax.tick_params(labelsize=7)
                        rows.append(dict(index=row.index,point=group,batch=row.batch,run=row.run,state=state,method=mode,global_max_angle_parameter_deg=peak[0],global_max_delay_parameter_ns=peak[1],human_candidate_identified=False,physical_calibration=False,background_status='A: no trusted reference',meaning='global diagnostic maximum, not a selected human path'))
                for ax in axes[-1]:ax.set_xlabel('Uncalibrated delay parameter (ns)')
                fig.colorbar(im,ax=axes.ravel().tolist(),label='Per-panel normalized MUSIC power (dB)',shrink=.5,pad=.02)
                fig.suptitle(f'{group} | source {row.index:03d} | {row.batch} / {row.run}\nUncalibrated per-state spectra; no human-path label or position estimate',fontsize=12)
                pdf.savefig(fig,dpi=120)
                if row.index==int(mm['index'].min()):fig.savefig(root/(group.replace('+','plus').replace('-','minus')+'_first_run.png'),dpi=120)
                plt.close(fig)
        for a,b in itertools.combinations(mm.itertuples(index=False),2):
            for state in rat.MAP:
                for mode in ['A','C']:
                    ka=(a.index,state,mode);kb=(b.index,state,mode)
                    pairs.append(dict(point=group,index_a=a.index,index_b=b.index,same_batch=a.batch==b.batch,state=state,method=mode,cov_distance=distance(covs[ka],covs[kb]),spectrum_rms=float(np.sqrt(np.mean((spectra[ka]-spectra[kb])**2))),angle_max_difference_deg=abs(peaks[ka][0]-peaks[kb][0]),delay_max_circular_difference_ns=abs(float(rat.wrap_delay(peaks[ka][1]-peaks[kb][1]))),meaning='global-maximum stability does not identify a human reflector'))
        zz=old[old['index'].isin(mm['index'])];x,y=coord[group]
        point.append(dict(point=group,x=x,y=y,runs=len(mm),states=len(zz),old_common_model_passes=int(zz.common_model_pass.sum()),status='A',status_text='未能建立可信背景参考',human_path_verified=False,position_verified=False,atlas=str(pdfpath)))
        print('Reused',group,len(mm),'runs',flush=True)
    pd.DataFrame(rows).to_csv(R/'results/human_diagnostic_peaks.csv',index=False)
    df=pd.DataFrame(pairs);df.to_csv(R/'results/human_spectral_stability_pairs.csv',index=False)
    df.groupby(['point','same_batch','method']).agg(comparisons=('state','size'),median_cov_distance=('cov_distance','median'),median_spectrum_rms=('spectrum_rms','median'),median_max_angle_difference_deg=('angle_max_difference_deg','median'),median_max_delay_difference_ns=('delay_max_circular_difference_ns','median')).reset_index().to_csv(R/'results/human_stability_summary.csv',index=False)
    pd.DataFrame(sources).to_csv(R/'results/human_reuse_provenance.csv',index=False)
    pd.DataFrame(point).to_csv(R/'results/point_status.csv',index=False)
if __name__=='__main__':main()
