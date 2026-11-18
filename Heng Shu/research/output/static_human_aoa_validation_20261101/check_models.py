"""Bounded numerical controls, independent of human ground truth."""
import numpy as np
import pandas as pd
from common import *
from state_sync import solve,EDGES
from music_baseline import steering,spectrum,ANG
from threadpoolctl import threadpool_limits

def run():
 rng=np.random.default_rng(20261101);rows=[];weak=[]
 for band in ['HT20','HT40']:
  rat=engine(band);ref=rat.unit(rng.normal(size=(3,30))+1j*rng.normal(size=(3,30)))
  te=[];pe=[];re=[]
  for t,p in zip(rng.uniform(-100,100,30),rng.uniform(-np.pi,np.pi,30)):
   x=rat.apply(ref,-t,-p);tt,pp,co,ed=rat.fit(x,ref);te.append(abs(tt-t));pe.append(abs(rat.wrap_phase(pp-p)));re.append(np.linalg.norm(rat.apply(x,tt,pp)-ref))
  rows.append(dict(band=band,test='known_common_error_recovery',value=max(re),limit=1e-4,passed=max(re)<1e-4,detail=f'max delay error {max(te):.6g} ns; phase {max(pe):.6g} rad'))
  x=np.array([rat.apply(ref,-t,-p) for t,p in zip(rng.uniform(-30,30,128),rng.uniform(-np.pi,np.pi,128))]);b,_=rat.correct(x,'B');c,_=rat.correct(x,'C')
  rows.append(dict(band=band,test='CPE_covariance_invariant',value=distance(rat.covariance(x),rat.covariance(b)),limit=1e-10,passed=distance(rat.covariance(x),rat.covariance(b))<1e-10,detail='Uncentred outer products; not a test of human detection'))
  rows.append(dict(band=band,test='spatial_phase_preserved',value=rat.pair_error(x,c),limit=1e-8,passed=rat.pair_error(x,c)<1e-8,detail='All chains receive the same frequency-dependent rotation'))
  # Verify exact-frequency FFT search against direct evaluation, including the irregular HT20 tone set.
  z=x[17];t,p,co,ed=rat.fit(z,ref,refine=False);grid=2 if band=='HT40' else 4;ts=np.arange(-rat.PERIOD/2,rat.PERIOD/2,grid)
  vals=abs(np.exp(-2j*np.pi*ts[:,None]*rat.F[None,:]*1e-9)@(z.conj()*ref).sum(0));actual=abs(np.sum(z.conj()*ref*np.exp(-2j*np.pi*rat.F*t*1e-9)))
  rows.append(dict(band=band,test='FFT_matches_exact_frequency_grid',value=float(abs(actual-vals.max())),limit=1e-10,passed=abs(actual-vals.max())<1e-10,detail='No continuous-tone substitution'))
  merr=[]
  for state in STATES:
   ys=np.array([(6.5-m)*.028 for m in MAP[state]])
   for angle,delay in [(-33.,28.),(42.,-60.)]:
    ideal=np.exp(2j*np.pi*rat.fc*ys[:,None]*np.sin(np.deg2rad(angle))/299792458-2j*np.pi*rat.F[None,:]*delay*1e-9)
    cov=rat.covariance(np.repeat(ideal[None],8,axis=0));a,taxis,period=steering(rat,state);p=spectrum(cov,a,1,len(ANG),len(taxis));i,j=np.unravel_index(p.argmax(),p.shape);merr.append(max(abs(ANG[i]-angle),abs(taxis[j]-delay)))
  rows.append(dict(band=band,test='MUSIC_same_model_sign_index_all_states',value=max(merr),limit=1e-9,passed=max(merr)<1e-9,detail='Two on-grid synthetic paths per state, separately; no physical calibration implied'))
  # A weak static component is part of the reference, whereas a changed component can be absorbed by the estimated common gauge.
  ys=np.array([.028,0,-.028]);f=rat.fc+rat.F
  bg=np.exp(2j*np.pi*f[None,:]*ys[:,None]*np.sin(.2)/299792458)
  path=np.exp(2j*np.pi*f[None,:]*ys[:,None]*np.sin(-.65)/299792458)*np.exp(-2j*np.pi*rat.F[None,:]*17e-9)
  for amplitude in [.02,.05,.1]:
   reference=bg+amplitude*path
   for case in ['static_in_reference','changed_coefficient']:
    errors=[]
    for phase in np.linspace(-np.pi,np.pi,24,endpoint=False):
     clean=reference if case=='static_in_reference' else bg+amplitude*np.exp(1j*phase)*path
     observed=rat.apply(clean,-31.7,-1.2);dt,ph,co,ed=rat.fit(observed,reference);out=rat.apply(observed,dt,ph)
     errors.append(float(np.linalg.norm(out-clean)/np.linalg.norm(amplitude*path)))
    weak.append(dict(band=band,case=case,amplitude=amplitude,relative_weak_component_distortion_median=np.median(errors),relative_weak_component_distortion_max=max(errors),engineering_limit=.05,passed=max(errors)<=.05,interpretation='synthetic preservation control, not proof of physical human-path identity'))
 # Graph convention and leaf observability are checked using exact synthetic offsets.
 phase=rng.uniform(-1,1,6);delay=rng.uniform(-10,10,6);edges=[]
 for s,si,t,ti,ant in EDGES:
  i,j=STATES.index(s),STATES.index(t);edges.append(dict(s=s,t=t,phase_rad=phase[i]-phase[j],delay_ns=delay[i]-delay[j],weight=1.))
 z,d=solve(edges);err=max(abs(np.angle(z*np.exp(-1j*(phase-phase[4])))));dt=max(abs(d-(delay-delay[4])))
 rows.append(dict(band='graph',test='phase_and_delay_sign_convention',value=max(err,dt),limit=1e-10,passed=max(err,dt)<1e-10,detail='Reference node C1'))
 rows.append(dict(band='graph',test='leaf_bridge_unidentifiable',value=int(solve(edges[1:]) is None),limit=1,passed=solve(edges[1:]) is None,detail='Removing the G1 overlap disconnects the graph'))
 pd.DataFrame(rows).to_csv(R/'results/numerical_controls.csv',index=False);pd.DataFrame(weak).to_csv(R/'results/weak_component_controls.csv',index=False)
 assert all(r['passed'] for r in rows)
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(weak).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
