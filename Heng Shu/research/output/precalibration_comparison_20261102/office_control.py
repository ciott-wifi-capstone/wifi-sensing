import models as m
import numpy as np,pandas as pd
m.F=5.2e9+np.array([-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28])*312500.
m.GEOM=np.ones((6,2,30),complex)
us={k:m.load(k)[1] for k in ['office_R1','office_R2']};rows=[]
for model in ['constant','affine','tone']:
 t=m.fit(['office_R1'],us,model);rows.extend(m.residuals('office_R2',us['office_R2'],t,'office_relative_'+model))
f=pd.DataFrame(rows);f['reference_type']='scene-specific relative phase, not hardware; Office geometry unknown';f.to_csv(m.R/'office_relative_checks.csv',index=False);s=f.groupby('model').agg(median_block_median_deg=('median_deg','median'),median_block_p90_deg=('p90_deg','median'));s.to_csv(m.R/'office_relative_summary.csv');print(s.to_string())
