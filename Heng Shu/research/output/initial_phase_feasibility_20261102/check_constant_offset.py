from pathlib import Path
import numpy as np,pandas as pd,json
R=Path(__file__).resolve().parent;S=R.parent/'static_human_aoa_validation_20261101';states=json.loads((S/'geometry.json').read_text())['state_map'];rows=[]
def pairs(key,s):
 x=np.load(S/'packets'/f'{key}_{s}.npz')['x'];p=x[:,1:]*x[:,:1].conj();return p/np.maximum(abs(p),1e-30)
for src,dst in [('office_R1','office_R2'),('0','3'),('12','15'),('21','24'),('30','33')]:
 for s in states:
  a=pairs(src,s);b=pairs(dst,s);ref=np.exp(1j*np.angle(a[:64].mean(0)))
  for ci in range(2):
   train=b[:64,ci,::2]*ref[ci,::2].conj();offset=float(np.angle(train.mean()))
   test=b[64:,ci,1::2]*ref[ci,1::2].conj();before=abs(np.angle(test))*180/np.pi;after=abs(np.angle(test*np.exp(-1j*offset)))*180/np.pi
   rows.append(dict(source=src,target=dst,state=s,chain=ci+2,offset_deg=offset*180/np.pi,fit_resultant=abs(train.mean()),raw_test_median_deg=float(np.median(before)),raw_test_p90_deg=float(np.quantile(before,.9)),corrected_test_median_deg=float(np.median(after)),corrected_test_p90_deg=float(np.quantile(after,.9)),status='relative change diagnostic; not physical hardware calibration; not applied to MUSIC'))
f=pd.DataFrame(rows);f.to_csv(R/'constant_offset_diagnostic.csv',index=False)
s=f.groupby(['source','target']).agg(median_channel_offset_deg=('offset_deg','median'),median_test_residual_deg=('corrected_test_median_deg','median'),min_pair_p90_deg=('corrected_test_p90_deg','min'),max_pair_p90_deg=('corrected_test_p90_deg','max'));print(s.to_string());print(f[(f.source=='office_R1')&(f.state=='G1')].to_string(index=False))
