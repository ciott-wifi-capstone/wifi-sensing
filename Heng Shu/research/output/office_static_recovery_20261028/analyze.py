"""Bounded Office HT20 reanalysis. No empty subtraction or array concatenation.
Run: python output/office_static_recovery_20261028/analyze.py
"""
from pathlib import Path
import importlib.util, json, hashlib, shutil, itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=Path(__file__).resolve().parent; ROOT=O.parents[1]
D=ROOT/'office_broadside_raw6state_20260907'
for p in ['results','figures','snapshots','packets']: (O/p).mkdir(exist_ok=True)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
for src,dst in [(ROOT/'output/experiment_analysis_20261010/analyze.py','decoder.py'),(ROOT/'output/ratnam_phase_timing_validation/core.py','core.py'),(ROOT/'output/ratnam_phase_timing_validation/config.json','config.json')]:
 if not (O/'snapshots'/dst).exists():shutil.copy2(src,O/'snapshots'/dst)
dec=module('office_decoder',O/'snapshots/decoder.py');rat=module('office_ratnam',O/'snapshots/core.py')
cfg=json.loads((D/'config.json').read_text()); IDX=np.array(cfg['array']['subcarrier_indices']); F=IDX*312500.; STATES=dec.STATES
MAP=cfg['switching']['chain_to_antenna']; Y=(6.5-np.arange(1,13))*.028
rat.F=F;rat.PERIOD=3200.;rat.CFG['delay_edge_ns']=1598.
ANG=np.arange(-90.,91.,3); TAU=np.arange(-800.,801.,8); M=10
# Ten translated, uniformly spaced windows; never smooth across the irregular centre.
WINDOWS=[np.arange(a,a+M) for a in range(5)]+[np.arange(a,a+M) for a in range(15,20)]
assert all(np.all(np.diff(IDX[w])==2) for w in WINDOWS)
def fit(x,ref,mask=None,refine=True):
 w=x.conj()*ref
 if mask is not None:w=w*mask
 w=w.sum(axis=0);b=np.zeros(800,complex);b[IDX-IDX.min()]=w
 tau=float(rat.wrap_delay(np.argmax(abs(np.fft.fft(b)))*4.))
 def val(t):return np.sum(w*np.exp(-2j*np.pi*F*t*1e-9))
 if refine:
  op=minimize_scalar(lambda t:-abs(val(t))**2,bounds=(max(-1600.,tau-4),min(1600.,tau+4)),method='bounded',options={'xatol':.001})
  if op.success and abs(val(op.x))>=abs(val(tau)):tau=float(op.x)
 mask=np.ones_like(x.real) if mask is None else mask
 return tau,float(np.angle(val(tau))),float(abs(val(tau))/max(np.linalg.norm(x*mask)*np.linalg.norm(ref*mask),1e-15)),abs(tau)>=1598.
rat.fit=fit
def cov(x):
 return sum((z:=x[:,:,w].reshape(len(x),-1)).T@z.conj()/len(x) for w in WINDOWS)/len(WINDOWS)
def spatial(x):return np.einsum('nmk,nlk->kml',x,x.conj())/len(x)
def dist(a,b):return float(np.linalg.norm(a-b)/max((np.linalg.norm(a)+np.linalg.norm(b))/2,1e-15))
def phase(x):return np.array([np.mean(x[:,m]*x[:,n].conj(),axis=0) for m,n in [(0,1),(0,2),(1,2)]])
def steer(state):
 ys=Y[np.array(MAP[state])-1];delta=-ys[:,None]*np.sin(np.deg2rad(ANG))[None,:]
 return np.exp(-2j*np.pi*(5.2e9/299792458*np.repeat(delta,M,axis=0)[:,:,None]+np.tile(np.arange(M)*625000.,3)[:,None,None]*TAU[None,None,:]*1e-9)).reshape(3*M,-1)
def music(r,a,q=3):
 _,v=np.linalg.eigh(r);s=1/np.maximum(np.sum(abs(v[:,:-q].conj().T@a)**2,axis=0),1e-12);s=s.reshape(len(ANG),len(TAU));return s/s.max()
def peak(s):
 i,j=np.unravel_index(np.argmax(s),s.shape);return float(ANG[i]),float(TAU[j])
def csv(name,rows):pd.DataFrame(rows).to_csv(O/'results'/name,index=False,encoding='utf-8-sig')
def savefig(fig,name):
 fig.savefig(O/'figures'/f'{name}.png',dpi=160);fig.savefig(O/'figures'/f'{name}.svg');plt.close(fig)

def main():
 hashes=[]
 for e in json.loads((D/'delivery_manifest.json').read_text())['files']:
  p=D/e['path'];h=hashlib.sha256(p.read_bytes()).hexdigest();hashes.append(dict(file=e['path'],sha256=h,match=h==e['sha256'],bytes=p.stat().st_size))
 csv('source_hashes.csv',hashes);assert all(r['match'] for r in hashes)
 config={'spacing_m':.028,'spacing_source':'User confirmation 2026-10-28 overrides original 0.54 lambda','occupancy':'Static human at one position beside midpoint; not an empty room','exact_positions':None,'paired_empty':'unconfirmed','bandwidth':'HT20','frequency_hz':F.tolist(),'state_map':MAP,'samples_per_run_state_max':768,'selection':'uniform indices of all guarded valid packets; independent of amplitude or spectrum','prefix_packets':64,'gate_delay_p90_ns':5,'gate_phase_p90_deg':15,'gate_prediction_corr_median':.9,'packet_corr_min':.9,'phase_timing_period_ns':3200,'phase_timing_grid_ns':4,'music_angle_grid_deg':ANG.tolist(),'music_delay_grid_ns':TAU.tolist(),'music_order':3,'order_sensitivity':[2,3,4],'smoothing_windows_indices':[w.tolist() for w in WINDOWS],'evaluation':'R1 development diagnostics; R2 and R3 held out; no parameter tuning. Within record prefix excluded from predictive evaluation.','geometry':'RX-local broadside toward TX; y left from behind RX. Angle axis remains uncalibrated.','post_processing':'No phase sign inversion, hardware calibration, complex empty subtraction, or coherent concatenation.'}
 (O/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 c,h,nic,dn,dc,perm=dec.decode(D/'session.dat');host=dec.host_times(D,'2026-09-07');assert len(c)==len(host) and np.all(np.diff(host)>=0)
 audit={'records':len(c),'logger_span_s':float(nic[-1]),'permutations':np.unique(perm,axis=0).tolist(),'rate_flags':np.unique(h[:,18].astype(int)+256*h[:,19]).tolist(),'all_nonzero_rssi':bool(np.all(h[:,10:13]>0))}
 # Exact-frequency numerical recovery check, independent of Office measurements.
 rng=np.random.default_rng(28);ref=rng.normal(size=(3,30))+1j*rng.normal(size=(3,30));checks=[]
 for t,p in [(31.7,.8),(-217.3,-1.4),(1100.,2.1)]:
  x=rat.apply(ref,-t,-p);tt,pp,co,edge=fit(x,ref);checks.append({'tau':t,'tau_error_ns':float(abs(rat.wrap_delay(tt-t))),'phase_error_rad':float(abs(rat.wrap_phase(pp-p))),'correlation':co})
 assert max(q['tau_error_ns'] for q in checks)<.01
 audit['synthetic_fit']=checks
 rows=[]; align=[];pairrows=[];sens=[];closure=[];maps={};xs={};rs={};traces=[];packets=[]
 for run in [1,2,3]:
  issued,starts,states,guard,info=dec.events(D/f'run{run}_events.log');align.append({'run':run,**info});ids={s:[] for s in STATES}
  for ei in range(len(starts)-1):
   if starts[ei]+.025>issued+30:continue
   i,j=np.searchsorted(host,[starts[ei]+guard,starts[ei]+.025-guard])
   if j-i>=10:ids[STATES[states[ei]]].extend(range(i,j))
  for state in STATES:
   full=np.array(ids[state]);sel=full[np.unique(np.linspace(0,len(full)-1,min(768,len(full))).astype(int))];x=rat.unit(c[sel].astype(complex));gate=rat.common_gate(x)
   # A medoid initialization is diagnostic only; never substitutes a failed baseline gate.
   features=(x[:64,:,None,:]*x[:64,None,:,:].conj()).reshape(64,-1);med=int(np.argmin(np.linalg.norm(features-features.mean(0),axis=1)))
   xm=x.copy();xm[[0,med]]=xm[[med,0]];mg=rat.common_gate(xm)
   b,bi=rat.correct(x,'B');y,ci=rat.correct(x,'C',gate['common_model_pass'])
   rr=cov(x);rb=cov(b);rc=cov(y);a=steer(state)
   maps[(run,state,'A')]=music(rr,a);maps[(run,state,'C')]=music(rc,a);xs[(run,state)]=x;rs[(run,state)]=(rr,rc)
   # Predictions fit on two chains and two thirds of tones; test on omitted entries.
   scores=[];posttau=[];rawtau=[];rawpsi=[];postpsi=[]
   for pi in range(64,len(x),8):
    t,p,ed,pre,post=rat.prediction(x[pi],x[0]);scores.append([pre[0],post[0],post[1]])
    t0,p0,_,_=fit(x[pi],x[0]);t1,p1,_,_=fit(y[pi],y[0]);rawtau.append(t0);posttau.append(t1);rawpsi.append(p0);postpsi.append(p1)
    traces.append(dict(run=run,state=state,time_s=float(host[sel[pi]]-issued),raw_tau_ns=t0,corrected_tau_ns=t1,raw_phase_rad=p0,corrected_phase_rad=p1))
   scores=np.array(scores)
   row=dict(run=run,state=state,valid_packets=len(full),analyzed_packets=len(x),**gate,medoid_index=med,medoid_gate_pass=mg['common_model_pass'],medoid_delay_p90_ns=mg['chain_delay_p90_ns'],medoid_phase_p90_deg=mg['chain_phase_p90_deg'],accepted_fraction=float(ci['accepted'][64:].mean()),raw_tau_iqr_ns=rat.iqr(rawtau),corrected_tau_iqr_ns=rat.iqr(posttau),raw_phase_sd_deg=rat.circ_sd(rawpsi)*180/np.pi,corrected_phase_sd_deg=rat.circ_sd(postpsi)*180/np.pi,predictive_real_before=float(np.median(scores[:,0])),predictive_real_after_proposed=float(np.median(scores[:,1])),predictive_magnitude_after_proposed=float(np.median(scores[:,2])),cpe_cov_relative_error=dist(rr,rb),chain_phase_change_deg=rat.pair_error(x,y),raw_peak_angle_deg=peak(maps[(run,state,'A')])[0],raw_peak_tau_ns=peak(maps[(run,state,'A')])[1],corrected_peak_angle_deg=peak(maps[(run,state,'C')])[0],corrected_peak_tau_ns=peak(maps[(run,state,'C')])[1])
   rows.append(row);print(json.dumps(row),flush=True)
   for i in range(len(x)):packets.append(dict(run=run,state=state,record_index=int(sel[i]),time_s=float(host[sel[i]]-issued),accepted=bool(ci['accepted'][i]),fit_corr=float(ci['correlation'][i]),edge=bool(ci['edge'][i]),tau_correction_ns=float(ci['tau_ns'][i])))
   np.savez_compressed(O/'packets'/f'R{run}_{state}.npz',record_index=sel,x=x,corrected=y,time_s=host[sel]-issued)
   for q in [2,3,4]:
    for mode,r in [('A',rr),('C',rc)]:
     s=music(r,a,q);an,ta=peak(s);sens.append(dict(run=run,state=state,mode=mode,q=q,angle_deg=an,tau_ns=ta))
  # Redundant observation: physical pair 8/6 uses chain 2/3 in both C states.
  p1=phase(xs[(run,'C1')])[2];p2=phase(xs[(run,'C2')])[2];df=np.angle(p1*p2.conj())
  closure.append(dict(run=run,pair='antenna8/6',median_abs_phase_deg=float(np.median(abs(df))*180/np.pi),circular_mean_phase_deg=float(np.angle(np.mean(np.exp(1j*df)))*180/np.pi),tone_concentration=float(abs(np.mean(np.exp(1j*df))))))
 for a,b in itertools.combinations([1,2,3],2):
  for s in STATES:
   xa,xb=xs[(a,s)],xs[(b,s)];pa,pb=phase(xa),phase(xb)
   phases=[float(np.median(abs(np.angle(pa[i]*pb[i].conj())))*180/np.pi) for i in range(3)]
   pairrows.append(dict(run_a=a,run_b=b,state=s,spatial_distance=dist(spatial(xa),spatial(xb)),amplitude_distance=dist(np.mean(abs(xa),axis=0),np.mean(abs(xb),axis=0)),chain12_phase_deg=phases[0],chain13_phase_deg=phases[1],chain23_phase_deg=phases[2],raw_cov_distance=dist(rs[(a,s)][0],rs[(b,s)][0]),corrected_cov_distance=dist(rs[(a,s)][1],rs[(b,s)][1]),raw_spectrum_rms=float(np.sqrt(np.mean((maps[(a,s,'A')]-maps[(b,s,'A')])**2))),corrected_spectrum_rms=float(np.sqrt(np.mean((maps[(a,s,'C')]-maps[(b,s,'C')])**2)))))
 csv('state_results.csv',rows);csv('alignment.csv',align);csv('between_runs.csv',pairrows);csv('shared_pair_closure.csv',closure);csv('model_order.csv',sens);csv('timing_traces.csv',traces);csv('packet_acceptance.csv',packets)
 audit['cpe_covariance_max_relative_error']=max(r['cpe_cov_relative_error'] for r in rows);audit['spatial_pair_max_change_deg']=max(r['chain_phase_change_deg'] for r in rows)
 assert audit['cpe_covariance_max_relative_error']<1e-12 and audit['spatial_pair_max_change_deg']<1e-8
 audit['manifest_pass']=True;audit['state_count']=len(rows);audit['common_gate_pass_count']=sum(r['common_model_pass'] for r in rows)
 (O/'results/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 np.savez_compressed(O/'results/music_spectra.npz',angle_deg=ANG,tau_ns=TAU,**{f'R{r}_{s}_{m}':a for (r,s,m),a in maps.items()})
 for mode in ['A','C']:
  fig,ax=plt.subplots(3,6,figsize=(18,10),layout='constrained')
  for ri in range(3):
   for si,s in enumerate(STATES):
    im=ax[ri,si].imshow(10*np.log10(np.maximum(maps[(ri+1,s,mode)],1e-4)),origin='lower',aspect='auto',extent=[TAU[0],TAU[-1],ANG[0],ANG[-1]],vmin=-25,vmax=0,cmap='magma');ax[ri,si].set_title(f'R{ri+1} {s}');ax[ri,si].set_xlabel('Uncalibrated delay (ns)');ax[ri,si].set_ylabel('Uncalibrated angle (deg)')
  fig.colorbar(im,ax=ax,shrink=.7,label='Per-map normalized MUSIC (dB)');fig.suptitle(f'Office HT20 | {mode}: '+('raw normalized CSI' if mode=='A' else 'gated common phase/timing correction; failures remain raw')+'\n28 mm | q=3 | separate state spectra; no human-path identification',fontsize=15);savefig(fig,f'01_music_{mode}')
 df=pd.DataFrame(rows);fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');labels=[f'R{r.run}-{r.state}' for r in df.itertuples()];ii=np.arange(len(df))
 for aa,pre,post,title in [(ax[0],'raw_tau_iqr_ns','corrected_tau_iqr_ns','Relative timing IQR (ns)'),(ax[1],'raw_phase_sd_deg','corrected_phase_sd_deg','Common phase circular SD (deg)')]:
  aa.plot(ii,df[pre],'o-',label='A: raw');aa.plot(ii,df[post],'s-',label='C: gated');aa.set_title(title);aa.legend()
 ax[2].plot(ii,df.predictive_real_before,'o-',label='Before');ax[2].plot(ii,df.predictive_real_after_proposed,'s-',label='Proposed correction');ax[2].set_title('Held-out entries: real correlation\n(proposal score, including failed gates)');ax[2].legend()
 for aa in ax:aa.set_xticks(ii,labels,rotation=90,fontsize=7);aa.grid(alpha=.25)
 savefig(fig,'02_reference_stability')
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');pp=pd.DataFrame(pairrows)
 for k,p in enumerate(['chain12_phase_deg','chain13_phase_deg','chain23_phase_deg']):ax[0].plot(np.arange(18),pp[p],'o-',label=p.replace('_phase_deg',''))
 ax[0].set_xticks(np.arange(18),[f'R{r.run_a}/R{r.run_b}\n{r.state}' for r in pp.itertuples()],rotation=90,fontsize=7);ax[0].set_ylabel('Median absolute inter-run phase difference (deg)');ax[0].legend();ax[0].grid(alpha=.3)
 ax[1].plot(np.arange(18),pp.raw_cov_distance,'o-',label='A: raw');ax[1].plot(np.arange(18),pp.corrected_cov_distance,'s-',label='C: separate record references');ax[1].set_xticks(np.arange(18),[f'{r.run_a}/{r.run_b}-{r.state}' for r in pp.itertuples()],rotation=90,fontsize=7);ax[1].set_ylabel('Symmetric relative covariance distance');ax[1].legend();ax[1].grid(alpha=.3);fig.suptitle('Repeatability diagnostics; no sign flips or cross-record fitting');savefig(fig,'03_between_runs')
 print('COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
