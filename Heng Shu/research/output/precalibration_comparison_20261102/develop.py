from models import *
import hashlib
us={k:load(k)[1] for k in DEV};rows=[]
for held in DEV:
 train=[k for k in DEV if k!=held]
 for model in ['constant','affine','tone']:
  t=fit(train,us,model);rows.extend(residuals(held,us[held],t,'pi_'+model))
pd.DataFrame(rows).to_csv(R/'development_loro.csv',index=False)
f=pd.DataFrame(rows);scores=f.groupby('model').agg(score_median=('median_deg','median'),score_p90_median=('p90_deg','median')).reset_index();best=scores.score_median.min();eligible=set(scores[scores.score_median<=best+1].model);chosen=next(x for x in ['pi_constant','pi_affine','pi_tone'] if x in eligible)
protocol_sha=hashlib.sha256((R/'PROTOCOL_core.md').read_bytes()).hexdigest();decision=dict(protocol_sha256=protocol_sha,chosen_model=chosen,rule='development leave-one-record-out median of temporal-block phase medians; simpler within 1 degree',scores=scores.to_dict('records'),human_ground_truth_used=False,independent_test=False)
(R/'development_decision.json').write_text(json.dumps(decision,indent=2))
rotrows=[];tplrows=[]
for mode,model in [('pi','constant'),('pi','affine'),('pi','tone'),('continuous','affine')]:
 t=fit(DEV,us,model,mode);np.savez_compressed(R/f'template_{mode}_{model}.npz',ghat=t['ghat'],anchor=t['anchor'])
 for key,d in t['details']:
  for j in range(2):rotrows.append(dict(record=key,method=mode+'_'+model,chain=j+2,phase_change_deg=d['phase_deg'][j],chosen_change_deg=d['chosen_deg'][j],coherence=d['coherence'][j],choice_margin=d['binary_margin'][j]))
 for s,state in enumerate(ST):
  for j in range(2):
   for k in range(30):tplrows.append(dict(model=mode+'_'+model,state=state,chain=j+2,frequency_Hz=F[k],phase_deg=float(np.angle(t['ghat'][s,j,k])*180/np.pi),observed_training_slot=bool(k in KTR)))
pd.DataFrame(rotrows).to_csv(R/'development_rotations.csv',index=False);pd.DataFrame(tplrows).to_csv(R/'phase_templates.csv',index=False)
print(json.dumps(decision,indent=2));print(pd.DataFrame(rotrows).query('method=="pi_affine"').to_string(index=False))
