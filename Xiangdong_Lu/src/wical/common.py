"""Deterministic I/O and audited physical geometry. No human coordinates."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STATES=('G1','G2','G3','G4','C1','C2')
MAP={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]}
FC=5.230e9; LIGHT=299792458.; FREQ=FC+np.arange(-58,59,4)*312500.
ANGLES=np.arange(-90.,91.); RANGES=np.array([.5,.65,.8,1.,1.3,1.7,2.2,2.8,3.5,4.5,6.,20.])
def coords(g):return (np.array(MAP[g])-6.5)*.028
def wrapdeg(a):return (np.asarray(a)+180)%360-180
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(p,x):(ROOT/p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def readcsv(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
def csvout(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(rows))
def key(r):return '_'.join(r[k] for k in ('batch','session','run'))
def circ_sd(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return float(np.degrees(np.sqrt(-2*np.log(np.clip(abs(np.mean(np.exp(1j*np.radians(a)))),1e-15,1))))) if len(a) else np.nan
def peaks(p,limit=8,separation=8):
    p=np.asarray(p);ix=np.flatnonzero((p>=np.r_[-np.inf,p[:-1]])&(p>=np.r_[p[1:],-np.inf]))
    ix=sorted(ix,key=lambda i:(-p[i],i));keep=[]
    for i in ix:
        if all(abs(i-j)>=separation for j in keep):keep.append(int(i))
        if len(keep)>=limit:break
    return keep
def norm(p):return p/np.maximum(np.mean(p,axis=-1,keepdims=True),1e-30)
def steering(g,near=False):
    y=coords(g)
    if not near:return np.exp(2j*np.pi*FC/LIGHT*(y-y.mean())[:,None]*np.sin(np.radians(ANGLES)))
    x=RANGES[:,None]*np.cos(np.radians(ANGLES))[None,:]
    yy=RANGES[:,None]*np.sin(np.radians(ANGLES))[None,:]
    d=np.sqrt(x[:,:,None]**2+(yy[:,:,None]-y)**2)
    return np.exp(-2j*np.pi*FC/LIGHT*(d-d[:,:,1:2])).transpose(2,0,1).reshape(3,-1)
