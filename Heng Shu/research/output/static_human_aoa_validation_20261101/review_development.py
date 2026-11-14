"""Freeze graph diagnostic templates before opening bounded evaluation records."""
import pandas as pd
from common import *
states=pd.concat([pd.read_csv(p) for p in (R/'results').glob('*_states.csv')]);states=states[states.stage=='development']
edges=pd.read_csv(R/'results/graph_development_edges.csv');templates=[]
edges['cohort']=np.where(edges['index'].astype(str).str.startswith('office'),'Office','primary')
for (cohort,edge),g in edges.groupby(['cohort','edge']):
 rat=engine('HT20' if cohort=='Office' else 'HT40');w=g.weight.to_numpy();w=w/w.sum();phase=g.phase_rad.to_numpy();d=g.delay_ns.to_numpy()
 # Canonical slope median; circular phase at DC. No position label selects a template.
 templates.append(dict(cohort=cohort,edge=int(edge),s=g.iloc[0].s,t=g.iloc[0].t,antenna=int(g.iloc[0].antenna),phase_rad=float(np.angle(np.sum(w*np.exp(1j*phase)))),delay_ns=float(np.median(d)),phase_resultant=float(abs(np.sum(w*np.exp(1j*phase)))),delay_iqr_ns=float(np.subtract(*np.quantile(d,[.75,.25]))),n_cycles=len(g)))
pd.DataFrame(templates).to_csv(R/'results/frozen_graph_templates.csv',index=False)
checks=pd.read_csv(R/'results/numerical_controls.csv');weak=pd.read_csv(R/'results/weak_component_controls.csv')
assert checks.passed.all()
dump(R/'results/development_review.json',dict(config_sha256=sha(R/'config.json'),all_numerical_checks_pass=True,common_gate_pass=int(states.common_model_pass.sum()),common_gate_total=len(states),failed_states=states.loc[~states.common_model_pass,['index','state','method','chain_phase_p90_deg']].to_dict('records'),weak_static_controls_pass=bool(weak[weak['case']=='static_in_reference'].passed.all()),weak_changed_controls_pass=bool(weak[weak['case']=='changed_coefficient'].passed.all()),evaluation_decision='Proceed only with the five predefined bounded records for diagnostics and falsification; do not expand coherent-array or human-location branches.',frozen_graph_templates_sha256=sha(R/'results/frozen_graph_templates.csv'),physical_human_path_valid=False,hardware_angle_valid=False,thresholds_unchanged=True))
print((R/'results/development_review.json').read_text(encoding='utf-8'))
