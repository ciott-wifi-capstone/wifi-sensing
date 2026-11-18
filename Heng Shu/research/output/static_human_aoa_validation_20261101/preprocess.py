"""Bounded exact-frequency common-error audit. Never reads human coordinates."""
import argparse,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import *
def prepare_primary(row):
 dec=module('decoder40',R/'provenance/decoder_ht40.py');p=Path(row.path);assert sha(p)==row.sha256
 meta=dec.read(p.parent/'run.json');x,host,nic=dec.decode(p);t=(host-meta['start_ns'])/1e9;starts,states,guard=dec.align(p.parent/'events.log',meta['start_ns']/1e9)
 out={}
 for s in STATES:
  ids=[];wins=[]
  for ei,(st,si) in enumerate(zip(starts[:-1],states[:-1])):
   if dec.STATES[si]!=s or st+.025>meta['requested_seconds']:continue
   l,r=np.searchsorted(t,[st+guard,st+.025-guard]);ids.extend(range(l,r));wins.extend([ei]*(r-l))
  ids=np.array(ids,dtype=int);w=np.array(wins,dtype=int);pick=np.unique(np.linspace(0,max(len(ids)-1,0),min(len(ids),CFG['max_packets_per_state'])).astype(int));ids,w=ids[pick],w[pick]
  out[s]=(x[ids],ids,t[ids],w,host[ids])
 return out,dict(source=str(p),raw_sha256=row.sha256,band='HT40',index=int(row['index']),run=row.run,cohort=row.label,stage=row.bounded_stage,complete_window=bool(row.complete_window),guard_ms=guard*1000)
def prepare_office(run):
 dec=module('decoder20',R/'provenance/decoder_ht20.py');p=ROOT/'office_broadside_raw6state_20260907';x,h,nic,dn,dc,perm=dec.decode(p/'session.dat');host=dec.host_times(p,'2026-09-07');issued,starts,states,guard,align=dec.events(p/f'run{run}_events.log');out={}
 for s in STATES:
  ids=[];wins=[]
  for ei in range(len(starts)-1):
   if dec.STATES[states[ei]]!=s or starts[ei]+.025>issued+30:continue
   l,r=np.searchsorted(host,[starts[ei]+guard,starts[ei]+.025-guard])
   if r-l>=10:ids.extend(range(l,r));wins.extend([ei]*(r-l))
  ids=np.array(ids);w=np.array(wins);pick=np.unique(np.linspace(0,len(ids)-1,min(len(ids),CFG['max_packets_per_state'])).astype(int));ids,w=ids[pick],w[pick]
  out[s]=(x[ids],ids,host[ids]-issued,w,np.round(host[ids]*1e9).astype(np.int64))
 return out,dict(source=str(p/'session.dat'),raw_sha256=sha(p/'session.dat'),band='HT20',index=f'office_R{run}',run=f'R{run}',cohort='Office',stage='development' if run==1 else 'evaluation',complete_window=True,guard_ms=guard*1000)
def process(data,info):
 rat=engine(info['band']);rows=[];pred=[];params=[];blocks=[];key=str(info['index']);start=time.perf_counter()
 for state,(raw,ids,ts,win,host) in data.items():
  x=rat.unit(raw.astype(complex));n=len(x)
  if n<128:rows.append(dict(**info,state=state,status='insufficient',n=n));continue
  gate=rat.common_gate(x);b,pb=rat.correct(x,'B');c,pc=rat.correct(x,'C',gate['common_model_pass']);ref=rat.unit(c[:64].mean(0)) if gate['common_model_pass'] else x[0]
  variants={'A':x,'B':b,'C':c};methodgate={'C':gate};aux={'C':pc};refs={'C':ref}
  if info['cohort']=='Office':
   if info['run']=='R1':dref=ref
   else:dref=np.load(R/'packets'/f'office_R1_{state}.npz')['reference_C']
   dg=rat.common_gate(np.concatenate([dref[None],x[:63]]));d=x.copy();dt=[];dp=[];dco=[];de=[];da=[]
   for z in x:
    t,p,co,ed=rat.fit(z,dref);ok=dg['common_model_pass'] and co>=.9 and not ed;dt.append(t);dp.append(p);dco.append(co);de.append(ed);da.append(ok)
   for i,ok in enumerate(da):
    if ok:d[i]=rat.apply(x[i],dt[i],dp[i])
   variants['D']=d;aux['D']=dict(tau_ns=np.array(dt),psi_rad=np.array(dp),correlation=np.array(dco),edge=np.array(de),accepted=np.array(da));refs['D']=dref;methodgate['D']=dg
  cov={k:rat.covariance(z) for k,z in variants.items()};assert distance(cov['A'],cov['B'])<1e-10
  for method in ['C','D']:
   if method not in variants:continue
   ag=methodgate[method];au=aux[method];reference=refs[method];reference_id=(f'office_R1:{state}:prefix64' if method=='D' else f'{key}:{state}:prefix64_after_C')
   assert rat.pair_error(x,variants[method])<1e-8
   row=dict(**info,state=state,method=method,status='computed',n=n,reference=reference_id,**ag,accepted_fraction=float(au['accepted'][64:].mean()),covariance_change=distance(cov['A'],cov[method]),spatial_pair_change_deg=rat.pair_error(x,variants[method]))
   rows.append(row)
   for i in range(n):params.append(dict(source=info['source'],source_sha256=info['raw_sha256'],index=key,run=info['run'],stage=info['stage'],state=state,method=method,packet_index=int(ids[i]),time_s=float(ts[i]),window=int(win[i]),reference=reference_id,model_gate=ag['common_model_pass'],tau_ns=float(au['tau_ns'][i]),psi_rad=float(au['psi_rad'][i]),correlation=float(au['correlation'][i]),boundary=bool(au['edge'][i]),accepted=bool(au['accepted'][i]),applied=bool(au['accepted'][i] and (method=='D' or i>0)),warmup=i<64,failure_reason='' if au['accepted'][i] else 'common_model' if not ag['common_model_pass'] else 'boundary' if au['edge'][i] else 'low_correlation'))
   for name,train,test in masks():
    for i in range(64,n,CFG['prediction_stride']):
     t,p,co,ed=rat.fit(x[i],reference,train);proposed=rat.apply(x[i],t,p);ok=ag['common_model_pass'] and co>=.9 and not ed;actual=proposed if ok else x[i]
     t0,p0,_,_=rat.fit(x[i],reference,test);t1,p1,_,_=rat.fit(actual,reference,test)
     pred.append(dict(index=key,run=info['run'],stage=info['stage'],state=state,method=method,source=info['source'],source_sha256=info['raw_sha256'],packet_index=int(ids[i]),time_s=float(ts[i]),window=int(win[i]),reference=reference_id,mask=name,time_block=min(3,int(4*i/n)),model_gate=ag['common_model_pass'],training_corr=co,boundary=ed,accepted=ok,applied=ok,actual_full_packet_applied=bool(au['accepted'][i]),raw_test_delay_ns=t0,actual_test_residual_delay_ns=t1,raw_test_phase_rad=p0,actual_test_residual_phase_rad=p1,raw_test_real=rat.score(x[i],reference,test)[0],proposed_test_real=rat.score(proposed,reference,test)[0],actual_test_real=rat.score(actual,reference,test)[0],test_magnitude=rat.score(actual,reference,test)[1],failure_reason='' if ok else 'model_or_corr_or_edge'))
  for method,z in variants.items():
   for bi,ix in enumerate(np.array_split(np.arange(64,n),4)):
    blocks.append(dict(index=key,state=state,method=method,block=bi,n=len(ix),start_s=float(ts[ix[0]]),end_s=float(ts[ix[-1]]),covariance_to_prefix=distance(rat.covariance(z[ix]),rat.covariance(z[:64]))))
  np.savez_compressed(R/'packets'/f'{key}_{state}.npz',x=x,packet_id=ids,time_s=ts,window_id=win,host_ns=host,**{f'out_{k}':v for k,v in variants.items()},**{f'cov_{k}':v for k,v in cov.items()},**{f'reference_{k}':v for k,v in refs.items()},**{f'{k}_applied':v['accepted'] for k,v in aux.items()})
  print(key,state,'gate',gate['common_model_pass'],'n',n,flush=True)
 for name,rr in [('states',rows),('predictions',pred),('packet_parameters',params),('temporal_blocks',blocks)]:pd.DataFrame(rr).to_csv(R/'results'/f'{key}_{name}.csv',index=False)
 dump(R/'results'/f'{key}_complete.json',dict(**info,seconds=time.perf_counter()-start,config_sha256=sha(R/'config.json'),state_count=len(data)))
def main(stage):
 assert sha(R/'config.json')==(R/'config.sha256').read_text().strip()
 if stage=='evaluation':assert (R/'results/development_review.json').exists()
 m=pd.read_csv(R/'input_manifest.csv');sel=m[m.bounded_stage==stage]
 for _,row in sel.iterrows():
  key=str(row['index'])
  if (R/'results'/f'{key}_complete.json').exists():continue
  data,info=prepare_primary(row);process(data,info)
 run=1 if stage=='development' else 2
 if not (R/'results'/f'office_R{run}_complete.json').exists():
  data,info=prepare_office(run);process(data,info)
 print('STAGE COMPLETE',stage,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','evaluation']);args=ap.parse_args()
 with threadpool_limits(limits=1):main(args.stage)
