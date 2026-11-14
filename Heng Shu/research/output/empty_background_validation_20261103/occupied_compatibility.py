from transfer_common import *
def main():
 S=R.parent/'static_human_aoa_validation_20261101'
 rows=[];details=[]
 for cap,idx in [('E1',78),('E2',81)]:
  for key in [3,15,24,33]:
   for state in rat.MAP:
    source=np.load(R/'cache'/f'{idx:03d}_{state}.npz');ref=template(source['x'])
    path=S/'packets'/f'{key}_{state}.npz';z=np.load(path);x=rat.unit(z['x'])
    for role,xx in [('prefix_development',x[:64]),('remaining_validation',x[64:])]:
     d,g=examine(xx,ref);rows.append(dict(empty_source=cap,occupied_record=key,state=state,role=role,**g))
     d['empty_source']=cap;d['occupied_record']=key;d['state']=state;d['role']=role;details.append(d)
   print(cap,key,flush=True)
 df=pd.DataFrame(rows);df.to_csv(R/'occupied_compatibility_gates.csv',index=False);pd.concat(details).to_csv(R/'occupied_compatibility_predictions.csv',index=False)
 print(df.groupby(['empty_source','occupied_record','role']).common_model_pass.agg(['sum','count']).to_string())
 # No occupied-whitening outputs are created on failed gates.
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
