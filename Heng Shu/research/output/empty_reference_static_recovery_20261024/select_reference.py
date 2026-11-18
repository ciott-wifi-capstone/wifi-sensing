"""Select one reference on E1 development only; never inspect holdout scores."""
import pandas as pd
from common import *
d=pd.read_csv(R/'results/reference_checks.csv')
d=d[d['index'].isin(CFG['empty_development'])]
scores=[]
for name,z in d.groupby('variant'):
    scores.append(dict(variant=name,passes=int(z.actual_model_applied.sum()),valid_templates=int(z.reference_valid.sum()),median_prediction=float(z.predicted_corr_C.median())))
# Reject unsupported templates before ranking predictive correlation; deterministic tie break.
scores.sort(key=lambda z:(-z['passes'],-z['valid_templates'],-z['median_prediction'],z['variant']))
dump(R/'results/development_selection.json',dict(selected_variant=scores[0]['variant'],scores=scores,selection_rule='Descending gate passes, supported templates, then median future held-feature correlation; variant name breaks ties.',development_indices=CFG['empty_development'],config_sha256=sha(R/'config.json'),all_development_gates_failed=all(z['passes']==0 for z in scores),status='Diagnostic selection; not evidence of a valid correction'))
dump(R/'provenance/confirmed_facts.json',dict(empty_pair=['E1','E3'],empty_pair_relation='continuous same scene and hardware session; no reboot/rewire/relocation, user confirmed',occupied_relation='different hardware session, user confirmed',filename_dates_valid=False,E2_E4_relation='not confirmed',supersedes_config_field='occupied_session_relationship only; numeric frozen parameters unchanged'))
print(scores)
