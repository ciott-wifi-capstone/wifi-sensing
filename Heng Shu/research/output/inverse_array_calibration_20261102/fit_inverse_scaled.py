"""Bounded shared initial-phase inverse fit; no held-out human coordinates."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize,least_squares
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent
S=R.parent/'static_human_aoa_validation_20261101'
R=R/'scaled'
sys.path.insert(0,str(S))
from common import engine,MAP,GEO
ST=list(MAP);CAL=[0,12];KEYS=[0,12,3,15,21,24,30,33]
POS={0:np.array([2.01,-1.]),12:np.array([2.01,1.])}
C=299792458.;rat=engine('HT40');F=rat.fc+rat.F
YS=np.array(GEO['array_y_local_m']);ANT=np.array([[0.,y] for y in YS]);TX=np.array([3.65,0.])
IDS={s:np.array(MAP[s])-1 for s in ST}
NF=len(rat.windows[0]);DF=np.tile(rat.F[rat.windows[0]],3)
RNG=np.random.default_rng(1102)
def unit(z):return z/np.maximum(abs(z),1e-30)
def lengths(key,s,kind):
 a=ANT[IDS[s]]
 if kind=='tx':return np.linalg.norm(a-TX,axis=1)
 p=POS[key].copy()
 if kind=='mirror':p[1]*=-1
 return np.linalg.norm(p-TX)+np.linalg.norm(a-p,axis=1)
def load_all():
 xs={};us={};manifest=[]
 for key in KEYS:
  xs[key]={};us[key]=[]
  for s in ST:
   path=S/'packets'/f'{key}_{s}.npz';z=np.load(path)
   xs[key][s]=z['out_C'];us[key].append(unit(z['x'][:,1:]*z['x'][:,:1].conj()))
   info=json.loads((S/'results'/f'{key}_complete.json').read_text(encoding='utf8'))
   manifest.append(dict(record=key,state=s,cache=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_source=info['source'],raw_sha256=info['raw_sha256'],role='calibration' if key in CAL else 'retrospective_validation'))
  us[key]=np.array(us[key])
 anchor=us[0][:,:64].mean(1);rows=[];rot={}
 for key in KEYS:
  v=us[key][:,:64].mean(1);c=(v[:,:,::2]*anchor[:,:,::2].conj()).mean(axis=(0,2))
  rot[key]=np.where(c.real>=0,1.,-1.)
  for j in range(2):rows.append(dict(record=key,chain=j+2,phase_change_deg=np.angle(c[j])*180/np.pi,rotation_deg=0 if rot[key][j]>0 else 180,coherence=abs(c[j])))
 pd.DataFrame(rows).to_csv(R/'branch_choices.csv',index=False)
 pd.DataFrame(manifest).to_csv(R/'source_manifest.csv',index=False)
 return xs,us,rot
def graph_fit(edges,obs):
 def residual(v):
  ph=np.r_[0,v];d=np.exp(1j*(ph[edges[:,1]]-ph[edges[:,0]]))-obs
  return np.r_[d.real,d.imag]
 fits=[least_squares(residual,np.zeros(11) if k==0 else RNG.uniform(-np.pi,np.pi,11),max_nfev=300) for k in range(8)]
 opt=min(fits,key=lambda a:np.sum(a.fun*a.fun))
 return np.angle(np.exp(1j*np.r_[0,opt.x]))
def dominant(us,rot):
 solutions={};rows=[];conflicts=[]
 for kind in ['tx','human']:
  edges=[];obs=[];details=[]
  for key in CAL:
   for si,s in enumerate(ST):
    L=lengths(key,s,kind);geo=np.exp(-2j*np.pi*(L[1:]-L[0])[:,None]*F[None,:]/C)
    q=us[key][si,64:]/rot[key][None,:,None]
    candidate=(q/geo[None]).mean(axis=(0,2))
    for j in range(2):
     edges.append([IDS[s][0],IDS[s][j+1]]);obs.append(candidate[j]);details.append((key,s,j+2,candidate[j]))
  edges=np.array(edges);obs=np.array(obs);ph=graph_fit(edges,obs);solutions[kind]=ph
  for edge,(key,s,j,z) in zip(edges,details):
   err=np.angle(np.exp(1j*(ph[edge[1]]-ph[edge[0]]))*np.conj(unit(z)))*180/np.pi
   rows.append(dict(hypothesis=kind,record=key,state=s,chain=j,observed_phase_deg=np.angle(z)*180/np.pi,resultant=abs(z),fit_error_deg=err))
  for i in range(12):conflicts.append(dict(hypothesis=kind,state=details[i][1],chain=details[i][2],between_calibration_conflict_deg=abs(np.angle(obs[i]*obs[i+12].conj()))*180/np.pi))
 pd.DataFrame(rows).to_csv(R/'dominant_phase_constraints.csv',index=False)
 pd.DataFrame(conflicts).to_csv(R/'dominant_between_positions.csv',index=False)
 return solutions
MODES={'tx_only':['tx'],'human_only':['human'],'tx_human':['tx','human'],'tx_mirror_placebo':['tx','mirror']}
def components(block,kind):
 L=lengths(block['key'],block['state'],kind)
 return np.repeat(np.exp(-2j*np.pi*rat.fc*(L-L[0])/C),NF)*np.exp(-2j*np.pi*DF*(L[0]/C))/np.sqrt(3*NF)
def setup(xs,rot):
 blocks=[];saved={}
 for key in KEYS:
  for s in ST:
   x=xs[key][s][64:]/np.r_[1.,rot[key]][None,:,None]
   cov=rat.covariance(x);saved[f'{key}_{s}']=cov
   if key in CAL:
    vals,vecs=np.linalg.eigh(cov);b=dict(key=key,state=s,ids=IDS[s],E=vecs[:,-3:],spectrum=vals)
    b['bases']={kind:components(b,kind) for kind in ['tx','human','mirror']};blocks.append(b)
 np.savez_compressed(R/'branch_corrected_covariances.npz',**saved)
 return blocks
def objective(v,blocks,kinds):
 ph=np.r_[0.,v[:11]];tot=0.;g=np.zeros_like(v);n=len(blocks)*len(kinds)
 for bi,b in enumerate(blocks):
  base=np.array([b['bases'][kind] for kind in kinds])
  z=base*np.exp(1j*np.repeat(ph[b['ids']],NF))[None,:]*np.exp(-2j*np.pi*DF*v[11+bi]*1e-9)[None,:]
  E=b['E'];proj=z@E.conj();tot+=np.sum(1-np.sum(abs(proj)**2,axis=1))
  for j,ant in enumerate(b['ids']):
   if ant==0:continue
   dz=np.zeros_like(z);dz[:,j*NF:(j+1)*NF]=1j*z[:,j*NF:(j+1)*NF]
   g[ant-1]+=-2*np.real(np.sum(proj.conj()*(dz@E.conj())))
  dz=(-2j*np.pi*DF*1e-9)[None,:]*z
  g[11+bi]+=-2*np.real(np.sum(proj.conj()*(dz@E.conj())))
 return float(tot/n),g/n
def initialize(ph,blocks,kinds):
 v=np.r_[ph[1:],np.zeros(len(blocks))];grid=np.arange(-400.,400.,8.)
 for bi,b in enumerate(blocks):
  base=np.array([b['bases'][kind] for kind in kinds])
  z=base[:,None,:]*np.exp(1j*np.repeat(ph[b['ids']],NF))[None,None,:]*np.exp(-2j*np.pi*grid[:,None]*DF[None,:]*1e-9)[None,:,:]
  score=np.sum(abs(z@b['E'].conj())**2,axis=(0,2));v[11+bi]=grid[np.argmax(score)]
 return v
def numerical_checks(blocks,dom):
 b=blocks[:2];kinds=['tx','human'];v=initialize(dom['tx'],b,kinds)
 value,g=objective(v,b,kinds);eps=1e-5;fd=[]
 for i in range(len(v)):
  dv=np.zeros_like(v);dv[i]=eps;fd.append((objective(v+dv,b,kinds)[0]-objective(v-dv,b,kinds)[0])/(2*eps))
 err=float(np.max(abs(g-np.array(fd))));assert err<1e-7,err
 edges=np.array([[IDS[s][0],a] for s in ST for a in IDS[s][1:]])
 true=np.r_[0.,np.linspace(-1.1,1.3,11)]
 estimate=graph_fit(edges,np.exp(1j*(true[edges[:,1]]-true[edges[:,0]])))
 recovery=float(np.max(abs(np.angle(np.exp(1j*(estimate-true))))));assert recovery<1e-5,recovery
 return dict(analytic_gradient_max_error=err,synthetic_connected_graph_max_phase_error_rad=recovery)
def main():
 start=time.time();xs,us,rot=load_all();dom=dominant(us,rot);blocks=setup(xs,rot)
 checks=numerical_checks(blocks,dom);(R/'numerical_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 starts=[np.zeros(12),dom['tx'],dom['human']]+[np.r_[0.,RNG.uniform(-np.pi,np.pi,11)] for _ in range(5)]
 pd.DataFrame({'antenna':np.arange(1,13),'tx_dominant_phase_deg':dom['tx']*180/np.pi,'human_dominant_phase_deg':dom['human']*180/np.pi}).to_csv(R/'dominant_phase_candidates.csv',index=False)
 allrows=[];perblock=[];solutions={};summary=[]
 for mode,kinds in MODES.items():
  opts=[]
  for si,ph in enumerate(starts):
   v=initialize(ph,blocks,kinds)
   scale=np.r_[np.ones(11),np.full(len(blocks),20.)]
   def scaled_objective(u):
    f,g=objective(u*scale,blocks,kinds);return f,g*scale
   opt=minimize(scaled_objective,v/scale,jac=True,method='L-BFGS-B',bounds=[(None,None)]*11+[(-20,20)]*len(blocks),options=dict(maxiter=300,ftol=1e-12,gtol=1e-7,maxls=30))
   opt.x=opt.x*scale
   opts.append(opt);allrows.append(dict(model=mode,start=si,cost=opt.fun,success=opt.success,iterations=opt.nit,message=str(opt.message)))
   print(mode,si,round(opt.fun,6),opt.success,opt.nit,flush=True)
  best=int(np.argmin([o.fun for o in opts]));v=opts[best].x
  phases=np.angle(np.exp(1j*np.r_[0.,v[:11]]));deltas=v[11:]
  costs=np.array([o.fun for o in opts]);near=np.where(costs<=costs.min()+max(.01*costs.min(),1e-4))[0]
  diffs=[np.sqrt(np.mean(np.angle(np.exp(1j*(o.x[:11]-v[:11])))**2))*180/np.pi for i,o in enumerate(opts) if i in near]
  solutions[mode]=phases
  summary.append(dict(model=mode,best_start=best,cost=float(costs[best]),near_optimal_starts=len(near),max_near_optimal_phase_rms_deg=max(diffs),solver_success=bool(opts[best].success)))
  np.savez_compressed(R/f'fit_{mode}.npz',phase_rad=phases,nuisance_delay_ns=deltas,all_parameters=np.array([o.x for o in opts]),all_costs=costs,calibration_records=CAL)
  for bi,b in enumerate(blocks):
   for kind in kinds:
    value,_=objective(np.r_[v[:11],v[11+bi]],[b],[kind])
    perblock.append(dict(model=mode,record=b['key'],state=b['state'],path_hypothesis=kind,subspace_loss=value,nuisance_delay_ns=deltas[bi]))
 pd.DataFrame(allrows).to_csv(R/'multistart_fits.csv',index=False);pd.DataFrame(summary).to_csv(R/'fit_summary.csv',index=False);pd.DataFrame(perblock).to_csv(R/'calibration_path_losses.csv',index=False)
 rows=[dict(model=k,antenna=m+1,phase_deg=float(p*180/np.pi),physical_calibration_verified=False) for k,ps in solutions.items() for m,p in enumerate(ps)]
 pd.DataFrame(rows).to_csv(R/'shared_phase_candidates.csv',index=False)
 meta=dict(protocol_sha256=hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest(),seconds=time.time()-start,calibration_records=CAL,evaluation_truth_loaded=False,checks=checks)
 (R/'fit_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
 print(pd.DataFrame(summary).to_string(index=False),flush=True);print('Seconds',meta['seconds'],flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
