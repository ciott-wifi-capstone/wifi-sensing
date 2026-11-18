from pathlib import Path
import sys,json,hashlib,re,struct,itertools,importlib.util
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar,linear_sum_assignment
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;O=R.parent
sys.path.insert(0,str(O/'same_frequency_spatial_background_20261104'))
import spatial_model as sm
ST=sm.ST;F=sm.F-sm.F.mean();PA=[(0,1),(0,2),(1,2)]
CAP={'E1':78,'E2':81,'E3':84,'E4':87}
spec=importlib.util.spec_from_file_location('decoder',O/'ratnam_phase_timing_validation/provenance/analyze.py')
dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
MAN=pd.read_csv(O/'empty_background_validation_20261103/source_manifest.csv')
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(i,s):return sm.loadempty(i,s)
def qmean(x):
 return np.array([(x[:,i]*x[:,j].conj()/np.maximum(abs(x[:,i]*x[:,j].conj()),1e-30)).mean(0) for i,j in PA])
def inv(x):
 z=sm.unit(x);cov=sm.spatial(z)
 amp=np.sqrt(np.real(np.diagonal(cov,axis1=1,axis2=2))).T
 coh=cov/np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2).real[:,:,None]*np.diagonal(cov,axis1=1,axis2=2).real[:,None,:],1e-30))
 return cov,amp,coh,qmean(z)
def phase_fit(xs,xt,linear):
 qs=np.array([qmean(xs[s]) for s in ['G1','G2']])
 qt=np.array([qmean(xt[s]) for s in ['G1','G2']])
 mask=np.arange(30)%3!=2
 params=[];w=np.ones((3,30),complex)
 for chain in [1,2]:
  z=(qt[:,chain-1]*qs[:,chain-1].conj()).sum(0);z=z[mask];f=F[mask]
  def obj(t):return -abs(np.sum(z*np.exp(-2j*np.pi*f*t*1e-9)))
  if linear:
   grid=np.arange(-400.,400.);best=grid[np.argmin([obj(t) for t in grid])]
   tau=minimize_scalar(obj,bounds=(best-1,best+1),method='bounded').x;tau=(tau+400)%800-400
  else:tau=0.
  phase=np.angle(np.sum(z*np.exp(-2j*np.pi*f*tau*1e-9)))
  w[chain]=np.exp(1j*(phase+2*np.pi*F*tau*1e-9))
  params.append(dict(chain=chain+1,phase_deg=float(np.degrees(phase)),relative_slope_ns=float(tau)))
 return w,params
def metrics(xs,xt,mask=None):
 a,aa,ca,qa=inv(xs);b,ab,cb,qb=inv(xt)
 reliable=(abs(qa)>=.5)&(abs(qb)>=.5)
 if mask is None:mask=np.ones(30,bool)
 reliable &= mask[None,:]
 ph=abs(np.angle(qb*qa.conj()))*180/np.pi
 support=float(reliable.sum()/(3*mask.sum()))
 med=float(np.median(ph[reliable])) if reliable.any() else np.nan
 p90=float(np.quantile(ph[reliable],.9)) if reliable.any() else np.nan
 amp=sm.dist(aa,ab);dst=sm.dist(a,b)
 return dict(amplitude_distance=amp,spatial_distance=dst,coherence_magnitude_distance=sm.dist(abs(ca),abs(cb)),phase_median_deg=med,phase_p90_deg=p90,reliable_support=support,gate=bool(support>=.5 and med<=15 and p90<=30 and amp<=.2 and dst<=.25))
def header_summary(path):
 raw=Path(path).read_bytes();pos=0;ants=[];rssi=[];records=0
 while pos<len(raw):
  size,magic,ver,ns=struct.unpack_from('<IIHB',raw,pos);end=pos+size+4;s=pos+11
  for _ in range(ns):
   length,n=struct.unpack_from('<IB',raw,s);name=raw[s+5:s+5+n].rstrip(b'\0').decode();b=s+7+n
   if name=='CSI':ants.append(raw[b+32])
   s+=length+4
  pos=end;records+=1
 u,c=np.unique(ants,return_counts=True)
 return dict(zip([str(tuple((int(v)>>(2*j))&3 for j in range(3))) for v in u],map(int,c)))
def audit_raw():
 (R/'raw_cache').mkdir(exist_ok=True);rows=[];sources=[];mapping=[];timings=[];baseline=[]
 for i in range(78,90):
  row=MAN[MAN['index']==i].iloc[0];p=Path(row.raw_path);meta=dec.read(p.parent/'run.json')
  events=(p.parent/'events.log').read_text()
  ev=re.findall(r'([\d.]+)\s+READY (G[1-4]|C[12]) chains=([\d,]+) t=(\d+) seq_idx=(\d)/6',events)
  bad=sum(list(map(int,e[2].split(',')))!=sm.MAP[e[1]] or int(e[4])!=ST.index(e[1]) for e in ev)
  dif=[k for k in meta['before'] if meta['before'][k]!=meta['after'][k]]
  rows.append(dict(index=i,capture=row.capture,run=row.run,boot_id=meta['before']['boot_id'],ifindex=meta['before']['ifindex'],phy=meta['before']['phy'],radio=meta['before']['radio'],within_run_identity_changed_fields=','.join(dif),state_mapping_errors=bad,ready_count=len(ev),esp_first_us=int(ev[0][3]),esp_last_us=int(ev[-1][3]),header_permutations=json.dumps(header_summary(p)),duration=meta['requested_seconds']))
  for f in [p,p.parent/'run.json',p.parent/'events.log',p.parent/'serial_transcript.log']:
   sources.append(dict(path=str(f),sha256=hashfile(f)))
  if row.capture not in ['E1','E3']:continue
  x,host,nic=dec.decode(p);ts=(host-meta['start_ns'])/1e9
  starts,states,guard=dec.align(p.parent/'events.log',meta['start_ns']/1e9)
  hs=np.array([float(e[0]) for e in ev])-meta['start_ns']/1e9
  dev=np.array([int(e[3]) for e in ev]);delta=(dev-dev[0])/1e6
  td=(nic-nic[0])/1e6;th=(host-host[0])/1e9
  coef=np.polyfit(td,th,1);res=th-np.polyval(coef,td)
  timings.append(dict(index=i,capture=row.capture,run=row.run,frames=len(x),host_nonmonotonic=int((np.diff(host)<=0).sum()),nic_nonmonotonic=int((np.diff(nic)<=0).sum()),nic_first=int(nic[0]),nic_last=int(nic[-1]),host_vs_nic_slope_ppm=(coef[0]-1)*1e6,host_nic_residual_p01_ms=float(np.quantile(res,.01)*1000),host_nic_residual_p99_ms=float(np.quantile(res,.99)*1000),event_guard_ms=guard*1000,first_ready_est_ms=starts[0]*1000,serial_delay_p01_ms=float(np.quantile(hs-starts,.01)*1000),serial_delay_p99_ms=float(np.quantile(hs-starts,.99)*1000),event_step_min_ms=float(np.diff(delta).min()*1000),event_step_max_ms=float(np.diff(delta).max()*1000)))
  # Exact reconstruction of baseline state IDs from original selection code.
  ids={s:[] for s in ST};wins={s:[] for s in ST}
  for e,(st,si) in enumerate(zip(starts[:-1],states[:-1])):
   if st+.025>meta['requested_seconds'] or .025-2*guard<.005:continue
   l,r=np.searchsorted(ts,[st+guard,st+.025-guard]);ids[ST[si]].extend(range(l,r));wins[ST[si]].extend([e]*(r-l))
  for s in ST:
   old=np.load(sm.B/'cache'/f'{i:03d}_{s}.npz');same=np.array_equal(ids[s],old['packet_id'])
   err=float(np.max(abs(sm.unit(x[ids[s]].astype(complex))-old['x'])))
   baseline.append(dict(index=i,state=s,packet_count=len(ids[s]),packet_ids_match=same,normalized_CSI_max_error=err))
   assert same and err<1e-12
  np.savez_compressed(R/'raw_cache'/f'{i}.npz',x=x,host=host,nic=nic,time_s=ts,starts=starts,states=states,guard=guard,duration=meta['requested_seconds'],device_delta=delta,event_host=hs)
  print('raw audit',i,flush=True)
 for name,data in [('metadata_audit',rows),('source_hashes',sources),('timing_audit',timings),('baseline_reconstruction',baseline)]:pd.DataFrame(data).to_csv(R/f'{name}.csv',index=False)
def phase_models():
 signs=json.loads((O/'empty_pi_sign_validation_20261104/frozen_signs.json').read_text())['pairs']
 models={};params=[];results=[];statefit=[]
 for ca,cb in [('E1','E3'),('E1','E2'),('E3','E4')]:
  ia,ib=CAP[ca],CAP[cb];xs={s:load(ia,s) for s in ST};xt={s:load(ib,s) for s in ST}
  mw={'raw':np.ones((3,30),complex),'pi':np.broadcast_to(np.array(signs[ca+'_'+cb]['sign'])[:,None],(3,30)).copy()}
  for mode,linear in [('constant_phase',False),('phase_slope',True)]:
   w,pr=phase_fit(xs,xt,linear);mw[mode]=w
   for v in pr:params.append(dict(pair=ca+'_'+cb,model=mode,**v))
  for mode,w in mw.items():
   models[ca+'_'+cb+'_'+mode]=w
  for off in range(3):
   for s in ST:
    xx=load(ib+off,s)
    for mode,w in mw.items():
     for scope,mask in [('all',None),('held_frequency',np.arange(30)%3==2)]:
      results.append(dict(pair=ca+'_'+cb,run=off+1,state=s,model=mode,frequency_scope=scope,state_used_in_fit=s in ['G1','G2'],**metrics(xs[s],xx*w[None],mask)))
  # Per-state inferred relative constant phase: diagnostic only, never applied.
  for off in range(3):
   for s in ST:
    a=qmean(xs[s]);b=qmean(load(ib+off,s));zz=b*a.conj()
    for k,(u,v) in enumerate(PA):
     statefit.append(dict(pair=ca+'_'+cb,run=off+1,state=s,pair_ch=f'{u+1}-{v+1}',relative_phase_deg=float(np.angle(zz[k].sum())*180/np.pi),circular_resultant=float(abs(zz[k].sum())/max(abs(zz[k]).sum(),1e-30))))
 np.savez_compressed(R/'frozen_phase_models.npz',**models)
 pd.DataFrame(params).to_csv(R/'phase_model_parameters.csv',index=False)
 pd.DataFrame(results).to_csv(R/'phase_model_validation.csv',index=False)
 pd.DataFrame(statefit).to_csv(R/'phase_by_state.csv',index=False)
 (R/'phase_model_freeze.json').write_text(json.dumps({'model_hash':hashfile(R/'frozen_phase_models.npz'),'protocol_hash':hashfile(R/'PROTOCOL.md'),'fit':'run1 G1/G2 k%3!=2; no occupied data'},indent=2))
 df=pd.DataFrame(results);print(df.query('run==3 and frequency_scope=="all"').groupby(['pair','model']).agg(phase=('phase_median_deg','median'),p90=('phase_p90_deg','median'),amp=('amplitude_distance','median'),distance=('spatial_distance','median'),pass_states=('gate','sum')).to_string(),flush=True)
def mappings():
 rows=[];decisions=[]
 for ca,cb in [('E1','E3'),('E1','E2'),('E3','E4')]:
  ia,ib=CAP[ca],CAP[cb];aa={s:inv(load(ia,s))[1] for s in ST};bb={s:inv(load(ib,s))[1] for s in ST}
  for perm in itertools.permutations(range(3)):
   cost=np.array([[sm.dist(aa[s],bb[t][list(perm)]) for t in ST] for s in ST])
   ri,ci=linear_sum_assignment(cost)
   for rr in range(3):
    target={s:inv(load(ib+rr,s))[1] for s in ST}
    diag=[sm.dist(aa[s],target[s][list(perm)]) for s in ST]
    matched=[sm.dist(aa[ST[i]],target[ST[j]][list(perm)]) for i,j in zip(ri,ci)]
    decisions.append(dict(pair=ca+'_'+cb,run=rr+1,chain_permutation=str(perm),assigned_target_states=','.join(ST[j] for j in ci),identity_state_cost=float(np.mean(diag)),assigned_state_cost=float(np.mean(matched))))
   if perm==(0,1,2):
    for i,s in enumerate(ST):
     for j,t in enumerate(ST):rows.append(dict(pair=ca+'_'+cb,source_state=s,target_state=t,amplitude_distance=cost[i,j]))
 pd.DataFrame(rows).to_csv(R/'state_mapping_costs.csv',index=False)
 pd.DataFrame(decisions).to_csv(R/'mapping_hypotheses.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  audit_raw();phase_models();mappings()
