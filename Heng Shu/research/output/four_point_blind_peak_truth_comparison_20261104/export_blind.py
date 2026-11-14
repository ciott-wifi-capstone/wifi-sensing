from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
R=Path(__file__).resolve().parent
A=R.parent/'e1_e3_mapping_phase_timing_audit_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for name,want in json.loads((A/'within_batch_predictions.json').read_text()).items():assert sha(A/name)==want
 candidates=pd.read_csv(A/'occupied_within_candidates.csv')
 validation=pd.read_csv(A/'occupied_within_validation.csv')
 rows=[];source=[]
 for v in validation.itertuples(index=False):
  p=A/'occupied_spectra'/f'{v.reference}_{v.index}_{v.state}.npz';z=np.load(p)
  theta=z['angles'];score=z['scores'][0]
  imax=int(np.argmax(score));angle=float(theta[imax])
  cc=candidates[(candidates['index']==v.index)&(candidates.reference==v.reference)&(candidates.state==v.state)]
  stable=cc[cc.stable].sort_values(['score','angle_parameter_deg'],ascending=[False,True])
  matched=cc[np.isclose(cc.angle_parameter_deg,angle)]
  isstable=bool(len(matched) and matched.stable.iloc[0])
  topstable=float(stable.angle_parameter_deg.iloc[0]) if len(stable) else np.nan
  rows.append(dict(index=v.index,batch_first=v.batch_first,reference=v.reference,state=v.state,
   spectrum_top1_deg=angle,spectrum_top1_score=float(score[imax]),spectrum_top1_is_stable=isstable,
   spectrum_top1_is_boundary=bool(abs(angle)==90),blind_stable_top1_deg=topstable,
   stable_count=len(stable),stable_angles=';'.join(str(x) for x in stable.angle_parameter_deg),
   candidate_angles=';'.join(str(x) for x in cc.angle_parameter_deg),
   matrix_change=v.matrix_change,score_threshold=v.score_threshold,
   human_path_validated=False))
  source.append(dict(path=str(p),sha256=sha(p)))
 df=pd.DataFrame(rows);df.to_csv(R/'blind_predictions.csv',index=False)
 candidates.to_csv(R/'all_frozen_candidates.csv',index=False)
 pd.DataFrame(source).to_csv(R/'spectrum_sources.csv',index=False)
 frozen={p.name:sha(p) for p in [R/'blind_predictions.csv',R/'all_frozen_candidates.csv',R/'PROTOCOL.md']}
 (R/'blind_predictions_frozen.json').write_text(json.dumps(frozen,indent=2))
 print('exported',len(df),'blind rows, no human coordinates loaded')
if __name__=='__main__':main()
