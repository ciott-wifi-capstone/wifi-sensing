from pathlib import Path
import importlib.util,hashlib,json
import numpy as np
from scipy.optimize import minimize_scalar
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
CFG=json.loads((R/'config.json').read_text(encoding='utf-8'));GEO=json.loads((R/'geometry.json').read_text(encoding='utf-8'));MAP=GEO['state_map'];STATES=list(MAP)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def engine(band):
 rat=module('rat_'+band,R/'provenance/ratnam/core.py')
 idx=np.arange(-58,59,4) if band=='HT40' else np.array([-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28])
 rat.F=idx*312500.;base=int(np.gcd.reduce(idx-idx[0]))*312500.;rat.PERIOD=1e9/base;grid=2. if band=='HT40' else 4.;edge=rat.PERIOD/2-grid
 rat.CFG['delay_edge_ns']=edge
 def fit(x,ref,mask=None,refine=True):
  w=x.conj()*ref
  if mask is not None:w=w*mask
  w=w.sum(axis=0);n=int(round(rat.PERIOD/grid));v=np.zeros(n,complex);v[np.round((rat.F-rat.F[0])/base).astype(int)]=w
  tau=float(rat.wrap_delay(np.argmax(abs(np.fft.fft(v)))*grid))
  def value(t):return np.sum(w*np.exp(-2j*np.pi*rat.F*t*1e-9))
  if refine:
   op=minimize_scalar(lambda t:-abs(value(t))**2,bounds=(max(-rat.PERIOD/2,tau-grid),min(rat.PERIOD/2,tau+grid)),method='bounded',options={'xatol':.001})
   if op.success and abs(value(op.x))>=abs(value(tau)):tau=float(op.x)
  mm=np.ones_like(x.real) if mask is None else mask
  corr=float(abs(value(tau))/max(np.linalg.norm(x*mm)*np.linalg.norm(ref*mm),1e-15))
  return tau,float(np.angle(value(tau))),corr,abs(tau)>=edge
 rat.fit=fit;rat.band=band;rat.fc=5.23e9 if band=='HT40' else 5.2e9
 rat.windows=[np.arange(j,j+15) for j in range(16)] if band=='HT40' else [np.arange(j,j+10) for j in list(range(5))+list(range(15,20))]
 def covariance(x):
  x=rat.unit(x);return sum((z:=x[:,:,w].reshape(len(x),-1)).T@z.conj()/len(x) for w in rat.windows)/len(rat.windows)
 rat.covariance=covariance
 return rat
def distance(a,b):return float(np.linalg.norm(a-b)/max((np.linalg.norm(a)+np.linalg.norm(b))/2,1e-15))
def masks():
 for m in range(3):
  test=np.zeros((3,30));test[m]=1;yield f'leave_chain_{m+1}',1-test,test
 for b in range(3):
  test=np.zeros((3,30));test[:,b*10:(b+1)*10]=1;yield f'leave_frequency_block_{b+1}',1-test,test
