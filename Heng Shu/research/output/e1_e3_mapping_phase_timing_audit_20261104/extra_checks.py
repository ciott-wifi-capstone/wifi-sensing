from audit import *
def payloads(p):
 raw=Path(p).read_bytes();pos=0;out=[]
 while pos<len(raw):
  size,_,_,ns=struct.unpack_from('<IIHB',raw,pos);s=pos+11
  for _ in range(ns):
   length,n=struct.unpack_from('<IB',raw,s);name=raw[s+5:s+5+n].rstrip(b'\0').decode();b=s+7+n
   if name=='CSI':out.append((raw[b+32],raw[b+39:b+231]))
   s+=length+4
  pos+=size+4
 return out
def parser_checks():
 rows=[]
 for idx in [78,79,80,84,85,86]:
  p=Path(MAN[MAN['index']==idx].iloc[0].raw_path);px=payloads(p);z=np.load(R/'raw_cache'/f'{idx}.npz')
  # Independent scalar bit extraction plus assignment to antenna IDs (not argsort).
  for i in np.unique(np.linspace(0,len(px)-1,64).astype(int)):
   ant,payload=px[i];perm=[(ant>>(2*j))&3 for j in range(3)]
   bits=int.from_bytes(payload,'little');ref=np.zeros((3,30),complex)
   for k in range(30):
    for j in range(3):
     bit=51*k+3+16*j;re=(bits>>bit)&255;im=(bits>>(bit+8))&255
     ref[perm[j],k]=complex(re-256 if re>=128 else re,im-256 if im>=128 else im)
   err=float(abs(z['x'][i]-ref).max());assert err==0
   rows.append(dict(index=idx,packet=int(i),permutation=str(tuple(perm)),max_CSI_difference=err))
 pd.DataFrame(rows).to_csv(R/'independent_decoder_checks.csv',index=False)
def gain_check():
 mw=np.load(R/'frozen_phase_models.npz');params=[];models={};allrows=[]
 for ca,cb in [('E1','E3'),('E1','E2'),('E3','E4')]:
  train=np.arange(30)%3!=2
  src=np.array([inv(load(CAP[ca],s))[1][:,train] for s in ['G1','G2']])
  tar=np.array([inv(load(CAP[cb],s))[1][:,train] for s in ['G1','G2']])
  gain=np.exp(np.median(np.log(np.maximum(src,1e-30)/np.maximum(tar,1e-30)),axis=(0,2)))
  gain/=np.exp(np.log(gain).mean())
  w=mw[ca+'_'+cb+'_phase_slope']*gain[:,None]
  models[ca+'_'+cb]=w
  params.append(dict(pair=ca+'_'+cb,gain1=gain[0],gain2=gain[1],gain3=gain[2]))
 np.savez_compressed(R/'frozen_gain_phase_models.npz',**models)
 (R/'gain_model_freeze.json').write_text(json.dumps(dict(model_hash=hashfile(R/'frozen_gain_phase_models.npz'),fit='run1 G1/G2 training frequencies; exploratory gain extension of predeclared phase model, no occupied data'),indent=2))
 for ca,cb in [('E1','E3'),('E1','E2'),('E3','E4')]:
  for off in [1,2]:
   for s in ST:
    allrows.append(dict(pair=ca+'_'+cb,run=off+1,state=s,**metrics(load(CAP[ca],s),load(CAP[cb]+off,s)*models[ca+'_'+cb][None])))
 pd.DataFrame(params).to_csv(R/'gain_parameters.csv',index=False)
 pd.DataFrame(allrows).to_csv(R/'gain_phase_validation.csv',index=False)
 print(pd.DataFrame(allrows).query('run==3').groupby('pair').agg(phase=('phase_median_deg','median'),amp=('amplitude_distance','median'),spatial=('spatial_distance','median'),passed=('gate','sum')).to_string())
def empty_candidates():
 # No new thresholds: source-only run1/run2 models with prior frozen settings.
 from importlib import import_module
 vo=import_module('validate_occupied')
 models=np.load(R/'frozen_phase_models.npz');gmodels=np.load(R/'frozen_gain_phase_models.npz')
 rows=[];cands=[];groups=[]
 for ca,cb in [('E1','E3'),('E1','E2'),('E3','E4')]:
  for run in [2,3]:
   for method in ['pi','constant_phase','phase_slope','phase_slope_gain']:
    by={}
    for s in ST:
     model,td,ts=sm.model_from_source(load(CAP[ca],s),load(CAP[ca]+1,s),s)
     w=gmodels[ca+'_'+cb] if method=='phase_slope_gain' else models[ca+'_'+cb+'_'+method]
     stat,pp,_=sm.assess(load(CAP[cb]+run-1,s)*w[None],model,td,ts)
     by[s]=pp;rows.append(dict(pair=ca+'_'+cb,run=run,model=method,state=s,**stat))
     for p in pp:cands.append(dict(pair=ca+'_'+cb,run=run,model=method,state=s,**p))
    groups.append(dict(pair=ca+'_'+cb,run=run,model=method,cross_G_clusters=len(vo.clusters(by))))
 pd.DataFrame(rows).to_csv(R/'empty_candidate_recheck.csv',index=False)
 pd.DataFrame(cands).to_csv(R/'empty_candidate_details.csv',index=False)
 pd.DataFrame(groups).to_csv(R/'empty_candidate_clusters.csv',index=False)
 print(pd.DataFrame(rows).query('run==3').groupby(['pair','model']).stable_candidates.sum().to_string())
def occupied_decision():
 # Record explicitly why correction is NOT transferred to the different occupied session.
 # Check existing frozen predictions and retain their exact scores/candidates.
 old=O/'same_frequency_spatial_background_20261104'
 d=pd.read_csv(old/'occupied_validation.csv');ks=[0,3,12,15,21,24,30,33];rows=[]
 for name,expected in json.loads((old/'frozen_predictions.json').read_text()).items():
  assert hashfile(old/name)==expected
 for k in ks:
  for cap in ['E1','E2']:
   z=d[(d.record==k)&(d.source==cap)&(d['mode']=='pi_hypothesis')]
   rows.append(dict(record=k,reference=cap,state_count=len(z),stable_state_peaks=int(z.stable_candidates.sum()),cross_G_clusters=0,new_continuous_chain_correction_applied=False,phase_parameter_source='none',decision='retain prior spectra; no evidenced general mapping/timing correction and E1-E3 phase/gain model fails empty validation; occupied is different uncalibrated session',human_AoA_deg=np.nan,position_error_m=np.nan))
 pd.DataFrame(rows).to_csv(R/'occupied_release_decisions.csv',index=False)
 (R/'occupied_baseline_references.json').write_text(json.dumps({str(old/k):v for k,v in json.loads((old/'frozen_predictions.json').read_text()).items()},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):
  parser_checks();gain_check();empty_candidates();occupied_decision()
