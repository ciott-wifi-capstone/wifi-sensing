"""Initial-phase models. No human coordinates or labels are used here."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.interpolate import interp1d
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101';sys.path.insert(0,str(S))
from common import engine,GEO,MAP
DEV=[0,12,21,30];EVAL=[3,15,24,33];ST=list(MAP);F=5.23e9+np.arange(-58,59,4)*312500.;KTR=np.arange(0,30,2);KTE=np.arange(1,30,2)
GEOM=[]
for state,ants in MAP.items():
 y=np.array([GEO['array_y_local_m'][a-1] for a in ants]);d=np.sqrt(3.65**2+y*y);GEOM.append(np.exp(-2j*np.pi*(d[1:]-d[0])[:,None]*F[None,:]/299792458.))
GEOM=np.array(GEOM)
def unit(z):return z/np.maximum(abs(z),1e-30)
def load(key):
 X=[];U=[]
 for state in ST:
  z=np.load(S/'packets'/f'{key}_{state}.npz');x=z['x'];X.append(z['out_C']);U.append(unit(x[:,1:]*x[:,:1].conj()))
 return X,np.array(U)
def prefix(u):return u[:,:64].mean(1)
def rotation(v,anchor,mode='pi'):
 # One choice per relative RF chain, shared across every switching state.
 c=np.mean(v[:,:,KTR]*anchor[:,:,KTR].conj(),axis=(0,2));phi=np.angle(c)
 chosen=np.where(c.real>=0,0.,np.pi) if mode=='pi' else phi
 return np.exp(1j*chosen),dict(phase_deg=phi*180/np.pi,chosen_deg=chosen*180/np.pi,coherence=abs(c),binary_margin=2*abs(c.real))
def fit(train,us,model,mode='pi'):
 vs=[prefix(us[k]) for k in train];anchor=vs[0];aligned=[];details=[]
 for key,v in zip(train,vs):
  rot,diag=rotation(v,anchor,mode);aligned.append(v/rot[None,:,None]);details.append((key,diag))
 mean=np.mean(aligned,axis=0);g=mean/GEOM;gtrain=g[:,:,KTR];ghat=np.empty_like(g,dtype=complex)
 for s in range(6):
  for j in range(2):
   a=np.unwrap(np.angle(gtrain[s,j]));freq=(F-F.mean())/1e6
   if model=='constant':p=np.full(30,np.angle(np.mean(gtrain[s,j])))
   elif model=='affine':p=np.polyval(np.polyfit(freq[KTR],a,1),freq)
   elif model=='tone':p=interp1d(freq[KTR],a,kind='linear',fill_value='extrapolate')(freq)
   else:raise ValueError(model)
   ghat[s,j]=np.exp(1j*p)
 return dict(ghat=ghat,anchor=mean,mode=mode,model=model,train=train,details=details)
def predict_phase(u,t):
 rot,diag=rotation(prefix(u),t['anchor'],t['mode']);return GEOM*t['ghat']*rot[None,:,None],rot,diag

def residuals(key,u,t,tag):
 pred,rot,diag=predict_phase(u,t);rows=[]
 for si,state in enumerate(ST):
  for j in range(2):
   e=abs(np.angle(u[si,64:,j][:,KTE]*pred[si,j,KTE].conj()))*180/np.pi
   for block,ix in enumerate(np.array_split(np.arange(len(e)),4)):
    q=e[ix];rows.append(dict(record=key,state=state,chain=j+2,model=tag,block=block,median_deg=float(np.median(q)),p90_deg=float(np.quantile(q,.9)),fraction_within15=float((q<=15).mean()),phase_choice_deg=float(diag['chosen_deg'][j]),unquantized_change_deg=float(diag['phase_deg'][j]),choice_coherence=float(diag['coherence'][j]),choice_margin=float(diag['binary_margin'][j]),samples=q.size,physical_calibration_valid=False))
 return rows
