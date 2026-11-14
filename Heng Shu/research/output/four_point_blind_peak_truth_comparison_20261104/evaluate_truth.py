from export_blind import *
def main():
 for f,h in json.loads((R/'blind_predictions_frozen.json').read_text()).items():assert sha(R/f)==h
 geo=json.loads((R.parent/'static_human_aoa_validation_20261101/geometry.json').read_text())
 records=pd.read_csv(A/'occupied_within_records.csv')
 truth={0:(1.64,1.),12:(1.64,-1.),21:(1.64,2.),30:(1.64,-2.)}
 mapping=geo['state_map'];ys=np.array(geo['array_y_local_m'])
 b=pd.read_csv(R/'blind_predictions.csv').merge(records[['index','label','run','complete_window','trailing_s']],on='index',validate='many_to_one')
 g=[]
 for batch,p in truth.items():
  xl=3.65-p[0];yl=-p[1];center=np.degrees(np.arctan2(yl,xl))
  for s,ant in mapping.items():
   yc=float(ys[np.array(ant)-1].mean());theta=np.degrees(np.arctan2(yl-yc,xl))
   g.append(dict(batch_first=batch,point=f'({p[0]:.2f},{p[1]:+.0f})',state=s,target_x=p[0],target_y=p[1],rxlocal_x=xl,rxlocal_y=yl,subarray_center_y=yc,theory_array_center_deg=center,theory_subarray_deg=theta))
 gt=pd.DataFrame(g);gt.to_csv(R/'theoretical_angles.csv',index=False)
 df=b.merge(gt,on=['batch_first','state'],validate='many_to_one')
 for kind in ['spectrum_top1','blind_stable_top1']:
  err=(df[kind+'_deg']-df.theory_subarray_deg+180)%360-180
  df[kind+'_signed_difference_deg']=err;df[kind+'_absolute_difference_deg']=abs(err)
  df[kind+'_within8_eval_only']=abs(err)<=8
 df.to_csv(R/'blind_peak_truth_comparison.csv',index=False)
 allp=pd.read_csv(R/'all_frozen_candidates.csv').merge(gt,on=['batch_first','state'],validate='many_to_one')
 allp=allp.merge(records[['index','run','complete_window']],on='index',validate='many_to_one')
 allp['signed_difference_deg']=(allp.angle_parameter_deg-allp.theory_subarray_deg+180)%360-180
 allp['absolute_difference_deg']=abs(allp.signed_difference_deg)
 allp['within8_evaluation_only']=allp.absolute_difference_deg<=8
 allp['human_path_verified']=False
 allp.to_csv(R/'all_candidates_truth_annotations.csv',index=False)
 def trip(q,col,digits=0):
  q=q.sort_values('index')
  return ' / '.join('NA' if not np.isfinite(v) else f'{v:+.{digits}f}' for v in q[col])
 compact=[]
 for ref in ['E1','E2']:
  for batch in truth:
   for s in mapping:
    q=df[(df.reference==ref)&(df.batch_first==batch)&(df.state==s)].sort_values('index')
    compact.append(dict(reference=ref,point=q.point.iloc[0],state=s,theory_deg=q.theory_subarray_deg.iloc[0],
     raw_peak_R1_R2_R3=trip(q,'spectrum_top1_deg'),
     raw_absdiff_R1_R2_R3=' / '.join(f'{v:.2f}' for v in q.spectrum_top1_absolute_difference_deg),
     raw_stability_R1_R2_R3=' / '.join('yes' if v else 'no' for v in q.spectrum_top1_is_stable),
     stable_peak_R1_R2_R3=trip(q,'blind_stable_top1_deg'),
     stable_absdiff_R1_R2_R3=' / '.join('NA' if not np.isfinite(v) else f'{v:.2f}' for v in q.blind_stable_top1_absolute_difference_deg)))
 pd.DataFrame(compact).to_csv(R/'compact_all_subarrays.csv',index=False)
 near=[]
 for (ref,batch,s),q in allp[allp.stable & allp.within8_evaluation_only].groupby(['reference','batch_first','state']):
  near.append(dict(reference=ref,batch_first=batch,state=s,records=','.join(map(str,sorted(q['index'].unique()))),candidate_angles=';'.join(map(str,sorted(q.angle_parameter_deg.unique()))),round_count=q['index'].nunique(),all_records_complete=bool(q.complete_window.all()),human_path_verified=False))
 pd.DataFrame(near).to_csv(R/'near_truth_evaluation_only.csv',index=False)
 print(pd.DataFrame(compact).query('reference=="E1"').to_string(index=False))
 print('Near truth stable candidates, retrospective annotations only:')
 print(pd.DataFrame(near).to_string(index=False))
 # Completeness/coverage only; no reported human AoA MAE.
 summary=df.groupby('reference').agg(records=('index','nunique'),state_records=('index','size'),complete_state_records=('complete_window','sum'),strongest_peak_stable=('spectrum_top1_is_stable','sum'),stable_top1_available=('blind_stable_top1_deg','count'))
 summary.to_csv(R/'coverage_summary.csv')
 print(summary.to_string())
if __name__=='__main__':main()
