from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101'
geo=json.loads((S/'geometry.json').read_text());states=geo['state_map'];c=299792458.
keys=['office_R1','office_R2','0','3','12','15','21','24','30','33'];refs={};rows=[];pred=[];sources=[]
for key in keys:
 for state,ants in states.items():
  path=S/'packets'/f'{key}_{state}.npz';z=np.load(path);x=z['x'];pairs=x[:,1:,:]*x[:,:1,:].conj();u=pairs/np.maximum(abs(pairs),1e-30)
  n=np.arange(-58,59,4) if not key.startswith('office') else np.array([-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28])
  freqs=(5.20e9 if key.startswith('office') else 5.23e9)+n*312500.
  model=np.zeros((2,30))
  if not key.startswith('office'):
   ys=np.array([geo['array_y_local_m'][a-1] for a in ants]);d=np.sqrt(3.65**2+ys**2)
   model=-2*np.pi*(d[1:]-d[0])[:,None]*freqs[None,:]/c
  avg=u[:64].mean(0);ref=np.exp(1j*np.angle(avg));refs[(key,state)]=(ref,u,model)
  errs=abs(np.angle(u[64:]*ref.conj()))*180/np.pi
  pred.append(dict(kind='within_record_prefix64',source=key,target=key,state=state,median_deg=float(np.median(errs)),p90_deg=float(np.quantile(errs,.9)),prefix_resultant_median=float(np.median(abs(avg))),n_packets=len(x)-64))
  for a in range(2):
   for k in range(30):rows.append(dict(record=key,state=state,chain=a+2,relative_to_chain=1,physical_antenna=ants[a+1],reference_antenna=ants[0],tone=int(n[k]),frequency_Hz=freqs[k],measured_pair_phase_deg=float(np.angle(ref[a,k])*180/np.pi),conditional_LoS_removed_phase_deg=None if key.startswith('office') else float(np.angle(ref[a,k]*np.exp(-1j*model[a,k]))*180/np.pi),prefix_resultant=float(abs(avg[a,k])),hardware_calibrated=False))
  sources.append(dict(record=key,state=state,cache=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n_packets=len(x)))
for src,dst in [('office_R1','office_R2'),('0','3'),('12','15'),('21','24'),('30','33')]:
 for state in states:
  ref,_,_=refs[(src,state)];_,u,_=refs[(dst,state)];err=abs(np.angle(u[64:]*ref.conj()))*180/np.pi
  pred.append(dict(kind='frozen_pair_reference_transfer',source=src,target=dst,state=state,median_deg=float(np.median(err)),p90_deg=float(np.quantile(err,.9)),n_packets=len(u)-64))
# Transfer labels specify known archive pairs, not a new test or a hardware-session claim.
pd.DataFrame(rows).to_csv(R/'empirical_phase_candidates.csv',index=False)
pd.DataFrame(pred).to_csv(R/'pair_phase_checks.csv',index=False)
pd.DataFrame(sources).to_csv(R/'cache_sources.csv',index=False)
summary={}
for a,b in [('office_R1','office_R2'),('0','3'),('12','15'),('21','24'),('30','33')]:
 s=[r for r in pred if r['kind']=='frozen_pair_reference_transfer' and r['source']==a]
 summary[a+' -> '+b]={'state_p90_min_deg':min(r['p90_deg'] for r in s),'state_p90_max_deg':max(r['p90_deg'] for r in s),'median_of_state_medians_deg':float(np.median([r['median_deg'] for r in s]))}
within={}
for key in keys:
 s=[r for r in pred if r['kind']=='within_record_prefix64' and r['source']==key]
 within[key]={'state_p90_range_deg':[min(r['p90_deg'] for r in s),max(r['p90_deg'] for r in s)],'prefix_resultant_median_range':[min(r['prefix_resultant_median'] for r in s),max(r['prefix_resultant_median'] for r in s)]}
y=np.array(geo['array_y_local_m']);d=np.sqrt(3.65**2+y*y);curv=float((d.max()-d.min())/c*5.23e9*360)
result=dict(scope='Existing cached normalized raw CSI only: 10 records x 6 states, 768 packets/state. No new MUSIC, no hardware calibration applied, no human GT used.',frozen_prefix_packets=64,transfer=summary,within_record=within,known_TX_direct_wave_geometric_phase_span_deg=curv,caveat='Pair phases contain spatial propagation and hardware response. LoS removal is conditional on direct-path dominance and is not a validated calibration.')
(R/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
