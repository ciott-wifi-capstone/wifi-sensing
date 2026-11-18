from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.ndimage import maximum_filter1d
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent
B=R.parent/'empty_background_validation_20261103'
P=R.parent/'empty_pi_sign_validation_20261104'
S=R.parent/'static_human_aoa_validation_20261101'
sys.path.insert(0,str(P))
from validate_sign import fit_sign
GEO=json.loads((S/'geometry.json').read_text(encoding='utf8'))
MAP=GEO['state_map'];ST=list(MAP);ANG=np.arange(-90.,91.,3.);F=5.23e9+np.arange(-58,59,4)*312500.
CAP={'E1':78,'E2':81,'E3':84,'E4':87}
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=(1,2),keepdims=True),1e-30)
def spatial(x):
 z=unit(x);return np.einsum('nmk,nlk->kml',z,z.conj())/len(z)
def dist(a,b):return float(np.linalg.norm(a-b)/max((np.linalg.norm(a)+np.linalg.norm(b))/2,1e-30))
def steering(state):
 y=np.array(GEO['array_y_local_m'])[np.array(MAP[state])-1]
 return np.exp(2j*np.pi*F[:,None,None]/299792458*y[None,:,None]*np.sin(np.deg2rad(ANG))[None,None,:])
def prepare(b,state):
 eig,u=np.linalg.eigh(.9*b+.1*np.trace(b,axis1=1,axis2=2).real[:,None,None]/3*np.eye(3)[None])
 w=(u/np.sqrt(np.maximum(eig,1e-30))[:,None,:])@u.conj().transpose(0,2,1)
 a=steering(state);wa=w@a;wb=w@b@w.conj().transpose(0,2,1)
 den=np.real(np.sum(wa.conj()*(wb@wa),axis=1))
 return dict(background=b,w=w,wa=wa,den=den,a=a)
def scores(c,m):
 wc=m['w']@c@m['w'].conj().transpose(0,2,1)
 v=np.real(np.sum(m['wa'].conj()*(wc@m['wa']),axis=1))/np.maximum(m['den'],1e-30)
 return np.array([v.mean(0),v[:15].mean(0),v[15:].mean(0)])
def peaks(s,threshold):
 return np.where((s==maximum_filter1d(s,size=5,mode='nearest'))&(s>threshold))[0]
def model_from_source(x,dev,state):
 b=spatial(x);m=prepare(b,state);blocks=[spatial(v) for v in np.array_split(dev,4)]
 sd=np.array([scores(c,m) for c in blocks]);td=max(dist(c,b) for c in blocks)
 ts=np.maximum(1.,sd.max(axis=(0,2)));return m,td,ts
def assess(x,m,td,ts):
 c=spatial(x);sp=scores(c,m);dv=dist(c,m['background']);change=dv>td
 full=peaks(sp[0],ts[0]) if change else np.array([],int)
 bs=[];bd=[]
 for z in np.array_split(x,4):
  cc=spatial(z);bd.append(dist(cc,m['background']))
  bs.append(scores(cc,m)[0])
 bs=np.array(bs)
 support=[]
 for i in full:
  matches=sum(any(abs(ANG[peaks(s,ts[0])]-ANG[i])<=9) and dd>td for s,dd in zip(bs,bd))
  fs=all(any(abs(ANG[peaks(sp[j],ts[j])]-ANG[i])<=9) for j in [1,2])
  support.append(dict(angle_parameter_deg=float(ANG[i]),score=float(sp[0,i]),temporal_blocks=int(matches),frequency_halves_consistent=bool(fs),stable=bool(matches>=3 and fs)))
 return dict(distance=dv,distance_threshold=td,matrix_change=bool(change),max_score=float(sp[0].max()),score_threshold=float(ts[0]),raw_candidates=len(full),stable_candidates=sum(p['stable'] for p in support)),support,dict(scores=sp,block_scores=bs,block_distances=np.array(bd),angles=ANG)
def loadempty(i,s):return np.load(B/'cache'/f'{i:03d}_{s}.npz')['x']
def checks():
 x=loadempty(78,'G1')[:128];rng=np.random.default_rng(1104)
 factor=np.exp(1j*rng.uniform(-np.pi,np.pi,len(x))[:,None]-2j*np.pi*(F-F.mean())[None,:]*rng.uniform(-300,300,len(x))[:,None]*1e-9)
 e=float(abs(spatial(x)-spatial(x*factor[:,None,:])).max());assert e<1e-12
 gain=rng.uniform(.2,5,len(x));eg=float(abs(spatial(x)-spatial(x*gain[:,None,None])).max());assert eg<1e-12
 b=spatial(x);m=prepare(b,'G1');ei=float(abs(scores(b,m)-1).max());assert ei<1e-10
 # Isotropic synthetic reference plus known directional covariance: no measured human truth.
 iso=np.tile(np.eye(3)[None],(30,1,1)).astype(complex);mi=prepare(iso,'G1')
 target=steering('G1')[:,:,np.where(ANG==21)[0][0]]
 cs=iso+.5*target[:,:,None]*target[:,None,:].conj()
 recovered=float(ANG[np.argmax(scores(cs,mi)[0])]);assert recovered==21
 # Arbitrary consistent unitary change of coordinates leaves the score unchanged.
 diag=np.exp(1j*rng.uniform(-np.pi,np.pi,(30,3)));D=np.eye(3)[None]*diag[:,None,:]
 xp=prepare(b,'G1');v=scores(spatial(x[64:]),xp)
 dd=D@spatial(x[64:])@D.conj().transpose(0,2,1)
 xp2=dict(xp);xp2['w']=D@xp['w']@D.conj().transpose(0,2,1);xp2['wa']=D@xp['wa']
 eu=float(abs(v-scores(dd,xp2)).max());assert eu<1e-10
 return dict(common_phase_timing_invariance_max=e,packet_gain_invariance_max=eg,background_identity_max=ei,synthetic_direction_recovered=recovered,consistent_coordinate_transform_max=eu)
