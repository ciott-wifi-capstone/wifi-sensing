"""Overlapping-antenna graph diagnostics. No virtual simultaneous snapshots."""
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import *
EDGES=[('G1',0,'C1',0,1),('G2',2,'C1',2,6),('G2',2,'C2',2,6),('G3',1,'C1',1,8),('G3',1,'C2',1,8),('G4',0,'C2',0,10),('C1',2,'C2',2,6),('C1',1,'C2',1,8)]
def solve(edges):
 n=6;H=np.zeros((n,n),complex);deg=np.zeros(n);A=[];b=[];w=[]
 for e in edges:
  i,j=STATES.index(e['s']),STATES.index(e['t']);wt=e['weight'];z=np.exp(1j*e['phase_rad']);H[i,j]+=wt*z;H[j,i]+=wt*z.conjugate();deg[i]+=wt;deg[j]+=wt
  row=np.zeros(n);row[i]=1;row[j]=-1;A.append(row);b.append(e['delay_ns']);w.append(wt)
 A=np.array(A);w=np.array(w);b=np.array(b)
 if np.linalg.matrix_rank(A)<5:return None
 _,v=np.linalg.eigh(np.diag(deg)-H);z=v[:,0]/np.maximum(abs(v[:,0]),1e-15);z/=z[4];keep=[0,1,2,3,5];d=np.zeros(6);d[keep]=np.linalg.lstsq(A[:,keep]*np.sqrt(w[:,None]),b*np.sqrt(w),rcond=None)[0]
 return z,d
def run(stage):
 rows=[];held=[];loops=[];coverage=[]
 for meta in sorted((R/'results').glob('*_complete.json')):
  info=json.loads(meta.read_text(encoding='utf-8'));
  if info.get('stage')!=stage:continue
  key=str(info['index']);rat=engine(info['band']);rep={};method='D' if key.startswith('office') else 'C'
  for s in STATES:
   z=np.load(R/'packets'/f'{key}_{s}.npz');x=z[f'out_{method}'];win=z['window_id'];ts=z['time_s']
   for wi in np.unique(win):
    ix=np.where(win==wi)[0]
    if len(ix)<2:continue
    v=x[ix];ph=np.angle(np.einsum('nmk,mk->n',v,x[ix[0]].conj()));v=v*np.exp(-1j*ph[:,None,None]);avg=v.mean(0)
    rep[(int(wi)//6,s)]=(avg,float(np.linalg.norm(avg)),float(np.mean(ts[ix])),len(ix))
  cycles=sorted(set(c for c,s in rep));count=0
  for cy in cycles:
   if not all((cy,s) in rep for s in STATES):continue
   count+=1;edges=[]
   for ei,(s,si,t,ti,ant) in enumerate(EDGES):
    u,cu,tu,nu=rep[(cy,s)];v,cv,tv,nv=rep[(cy,t)];delay,phase,corr,edge=rat.fit(v[ti:ti+1],u[si:si+1]);pred=rat.apply(v[ti:ti+1],delay,phase);res=float(abs(rat.score(pred,u[si:si+1],np.ones_like(pred.real))[2])*180/np.pi)
    e=dict(index=key,stage=stage,cycle=cy,edge=ei,s=s,t=t,antenna=ant,method=method,phase_rad=phase,delay_ns=delay,correlation=corr,search_boundary=edge,weight=max(1e-8,min(cu,cv)**2*corr**2),representative_coherence=min(cu,cv),gap_ms=abs(tv-tu)*1000,packets_min=min(nu,nv));edges.append(e);rows.append(e)
   by={i:e for i,e in enumerate(edges)}
   for name,terms in [('C1_C2_shared6_vs8',[(6,1),(7,-1)]),('G2_C1_C2',[(1,1),(6,1),(2,-1)]),('G3_C1_C2',[(3,1),(7,1),(4,-1)])]:
    total_delay=sum(sign*by[i]['delay_ns'] for i,sign in terms);dt=float(rat.wrap_delay(total_delay));ph=float(rat.wrap_phase(sum(sign*by[i]['phase_rad'] for i,sign in terms)+2*np.pi*rat.F[0]*(total_delay-dt)*1e-9))
    loops.append(dict(index=key,stage=stage,cycle=cy,loop=name,phase_abs_deg=abs(ph)*180/np.pi,delay_abs_ns=abs(dt),gate=abs(ph)*180/np.pi<=15 and abs(dt)<=5,interpretation='redundant algebraic consistency, not fixed hardware calibration'))
   for ei,e in enumerate(edges):
    sol=solve([ee for j,ee in enumerate(edges) if j!=ei])
    if sol is None:held.append(dict(index=key,stage=stage,cycle=cy,edge=ei,s=e['s'],t=e['t'],antenna=e['antenna'],identifiable=False,reason='leaf_bridge_has_no_alternative_path'));continue
    z,d=sol;i,j=STATES.index(e['s']),STATES.index(e['t']);rawde=d[i]-d[j]-e['delay_ns'];wrapped=float(rat.wrap_delay(rawde));pe=abs(float(rat.wrap_phase(np.angle(z[i]*z[j].conj())-e['phase_rad']+2*np.pi*rat.F[0]*(rawde-wrapped)*1e-9)))*180/np.pi;de=abs(wrapped)
    held.append(dict(index=key,stage=stage,cycle=cy,edge=ei,s=e['s'],t=e['t'],antenna=e['antenna'],identifiable=True,phase_error_deg=pe,delay_error_ns=de,pass_gate=pe<=15 and de<=5 and e['correlation']>=.9,reason=''))
  coverage.append(dict(index=key,stage=stage,possible_cycles=len(cycles),complete_cycles=count,all6available=count>0,full_hardware_calibration=False,coherent_assembly_authorized=False,stop_reason='independent_chain_response_unidentified_and_leaf_constraints_not_independent'))
  print(key,'complete cycles',count,flush=True)
 for n,r in [('edges',rows),('held_edges',held),('closure',loops),('coverage',coverage)]:pd.DataFrame(r).to_csv(R/'results'/f'graph_{stage}_{n}.csv',index=False)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','evaluation']);args=ap.parse_args()
 with threadpool_limits(limits=1):run(args.stage)
