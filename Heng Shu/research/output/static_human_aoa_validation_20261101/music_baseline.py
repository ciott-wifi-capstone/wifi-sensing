"""Diagnostic MUSIC and noncoherent angular fusion; never uses human coordinates."""
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
ANG=np.arange(-90.,91.,3.)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'savefig.bbox':'tight'})

def steering(rat,state):
 ys=np.array([(6.5-m)*.028 for m in MAP[state]]);nf=len(rat.windows[0]);f=rat.F[rat.windows[0]]
 # Frequency smoothing discards the HT20 cross-block half-spacing relation.
 period=1e9/float(np.gcd.reduce(np.round(np.diff(f)).astype(int)))
 tau=np.arange(-period/2,period/2,4.)
 spatial=rat.fc/299792458*np.repeat(-ys[:,None]*np.sin(np.deg2rad(ANG))[None,:],nf,axis=0)
 a=np.exp(-2j*np.pi*(spatial[:,:,None]+np.tile(f,3)[:,None,None]*tau[None,None,:]*1e-9))
 return a.reshape(3*nf,-1),tau,period

def spectrum(cov,a,q,nang,ntau):
 _,v=np.linalg.eigh(cov);den=np.sum(abs(v[:,:-q].conj().T@a)**2,axis=0);p=(1/np.maximum(den,1e-12)).reshape(nang,ntau)
 return p/p.max()

def run():
 results=[];peaks=[];fusions=[];invariants=[]
 for meta in sorted((R/'results').glob('*_complete.json')):
  info=json.loads(meta.read_text(encoding='utf-8'));key=str(info['index']);rat=engine(info['band']);saves={};angspec={};weights=[]
  for state in STATES:
   z=np.load(R/'packets'/f'{key}_{state}.npz');a,tau,period=steering(rat,state)
   base=spectrum(z['cov_A'],a,3,len(ANG),len(tau));b=spectrum(z['cov_B'],a,3,len(ANG),len(tau));invariants.append(dict(index=key,state=state,A_B_spectrum_max_difference=float(abs(base-b).max())))
   for method in ['A','B','C','D']:
    if f'cov_{method}' not in z:continue
    for q in [2,3,4]:
     p=spectrum(z[f'cov_{method}'],a,q,len(ANG),len(tau));ii,jj=np.unravel_index(p.argmax(),p.shape)
     results.append(dict(index=key,stage=info['stage'],band=info['band'],state=state,method=method,q=q,peak_angle_deg=ANG[ii],peak_delay_parameter_ns=tau[jj],spectrum_delay_period_ns=period,raw_fit_delay_period_ns=rat.PERIOD,physical_angle_calibrated=False,human_associated=False,source=info['source'],raw_sha256=info['raw_sha256']))
     if q!=3:continue
     saves[f'{state}_{method}']=p;angspec[(state,method)]=p.max(axis=1)
     loc=(p==maximum_filter(p,size=(5,5),mode=('nearest','wrap')));ids=np.argwhere(loc);ids=sorted(ids,key=lambda v:-p[tuple(v)])[:5]
     for rank,(i,j) in enumerate(ids,1):peaks.append(dict(index=key,state=state,method=method,rank=rank,angle_parameter_deg=ANG[i],delay_parameter_ns=tau[j],relative_power=float(p[i,j]),classification='unassigned_spectral_peak',human_valid=False))
   if state.startswith('G'):
    x=z['out_D'] if 'out_D' in z else z['out_C'];mid=len(x)//2;change=distance(rat.covariance(x[:mid]),rat.covariance(x[mid:]));weights.append((state,1/(1+change),change))
  np.savez_compressed(R/'results'/f'{key}_spectra.npz',angle_deg=ANG,tau_ns=tau,**saves)
  fig,axs=plt.subplots(2,3,figsize=(10.4,5.4),sharex=True,sharey=True);method='D' if key.startswith('office') else 'C'
  for ax,state in zip(axs.flat,STATES):
   im=ax.pcolormesh(tau,ANG,10*np.log10(np.maximum(saves[f'{state}_{method}'],1e-4)),vmin=-25,vmax=0,cmap='viridis',rasterized=True);ax.set_title(f'{state}: {MAP[state]}')
  for ax in axs[-1]:ax.set_xlabel('Uncalibrated delay parameter (ns)')
  fig.supylabel('Uncalibrated angle parameter (deg)',x=.035)
  fig.suptitle(f'Record {key} | {info["band"]} | {method}, q=3 | unassigned paths',fontsize=12);fig.subplots_adjust(right=.89,hspace=.25,wspace=.1,top=.86);fig.colorbar(im,cax=fig.add_axes([.91,.17,.015,.61]),label='Normalized spectrum (dB)')
  fig.savefig(R/'figures'/f'music_{key}.pdf');fig.savefig(R/'figures'/f'music_{key}.png',dpi=150);plt.close(fig)
  fig,ax=plt.subplots(figsize=(7,3.4))
  for method in ['A','B','C','D']:
   if ('G1',method) not in angspec:continue
   # Marginalize unknown state-specific delay gauges BEFORE fusion. G states only: no duplicated C antennas.
   fused=sum(w*angspec[(s,method)] for s,w,change in weights)/sum(w for s,w,change in weights);fused/=fused.max();peak=ANG[fused.argmax()]
   for s,w,change in weights:fusions.append(dict(index=key,stage=info['stage'],state=s,method=method,weight=w,temporal_covariance_distance=change,fused_peak_angle_parameter_deg=peak,human_valid=False,physical_angle_valid=False))
   ax.plot(ANG,10*np.log10(np.maximum(fused,1e-4)),label=method,ls='--' if method=='B' else '-')
  ax.set(xlabel='Uncalibrated angle parameter (deg)',ylabel='Normalized fused score (dB)',title=f'Record {key}: G1--G4 noncoherent angular fusion',ylim=(-20,1));ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'figures'/f'fusion_{key}.pdf');fig.savefig(R/'figures'/f'fusion_{key}.png',dpi=150);plt.close(fig)
  print('spectra',key,flush=True)
 for name,rows in [('music_sensitivity',results),('unassigned_peaks',peaks),('noncoherent_fusion',fusions),('music_invariants',invariants)]:pd.DataFrame(rows).to_csv(R/'results'/f'{name}.csv',index=False)
 assert max(r['A_B_spectrum_max_difference'] for r in invariants)<1e-8
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
