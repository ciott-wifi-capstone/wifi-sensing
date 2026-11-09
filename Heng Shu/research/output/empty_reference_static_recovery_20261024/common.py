from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'ratnam_phase_timing_validation'
CFG=json.loads((R/'config.json').read_text(encoding='utf-8'))
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
rat=module('ratnam_snapshot',R/'provenance/ratnam/core.py');dec=module('decoder_snapshot',R/'provenance/analyze.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def spatial(x):
    # Same-frequency 3x3 spatial matrices: invariant to common CPE AND timing.
    x=rat.unit(x);return np.einsum('nmk,nlk->kml',x,x.conj())/len(x)
def distance(a,b):return float(np.linalg.norm(a-b)/max((np.linalg.norm(a)+np.linalg.norm(b))/2,1e-15))
def invariants(x):
    x=rat.unit(x);sp=spatial(x);amp=np.mean(abs(x),axis=0)
    coh=[];angles=[]
    for m,n in [(0,1),(1,2),(0,2)]:
        z=x[:,m]*x[:,n].conj();mean=z.mean(0);den=np.mean(abs(z),axis=0)
        coh.append(abs(mean)/np.maximum(den,1e-15));angles.append(np.angle(mean))
    return dict(spatial=sp,amplitude=amp,pair_coherence=np.array(coh),pair_angle=np.array(angles),covariance=rat.covariance(x))
