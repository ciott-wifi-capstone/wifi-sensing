from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101'
f=pd.concat([pd.read_csv(R/'fused_predictions.csv'),pd.read_csv(R/'projection_fused_predictions.csv')],ignore_index=True);m=pd.read_csv(S/'input_manifest.csv')[['index','label']].rename(columns={'index':'record'});g=pd.read_csv(S/'results/evaluation_geometry.csv')[['label','theory_angle_deg']]
f=f.merge(m,on='record',validate='many_to_one').merge(g,on='label',validate='many_to_one');f['unassigned_peak_deviation_deg']=abs((f.angle_deg-f.theory_angle_deg+180)%360-180);f.to_csv(R/'post_prediction_deviations.csv',index=False)
rows=[]
for (stage,model),a in f.groupby(['stage','model']):
 e=a.unassigned_peak_deviation_deg;rows.append(dict(stage=stage,model=model,n=len(e),mean_deviation_deg=e.mean(),median_deviation_deg=e.median(),p90_deviation_deg=e.quantile(.9),max_deviation_deg=e.max(),fraction_within8=(e<=8).mean(),physical_human_MAE=None,validated_human_outputs=0))
pd.DataFrame(rows).to_csv(R/'descriptive_summary.csv',index=False);print(pd.DataFrame(rows).query('stage=="evaluation"').to_string(index=False));print(f.query('stage=="evaluation"').pivot(index='model',columns='label',values='angle_deg').to_string())
