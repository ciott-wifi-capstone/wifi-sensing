"""Source-only common-aligned average reference control; thresholds unchanged."""
from validate_sign import *
def make_mean(x):
 z=x[:64];ref=template(x).copy();supports=[]
 for iteration in range(3):
  aligned=[]
  for a in z:
   t,p,c,e=rat.fit(a,ref)
   if not e:aligned.append(rat.apply(a,t,p))
  supports.append(len(aligned)/len(z))
  if aligned:ref=rat.unit(np.mean(aligned,axis=0))
 return ref,min(supports)
def main():
 frozen=json.loads((O/'frozen_signs.json').read_text())['pairs'];refs={};rows=[];supports=[]
 for cap,idx in [('E1',78),('E2',81),('E3',84),('E4',87)]:
  for s in ST:
   ref,support=make_mean(load(idx,s)['x']);refs[(cap,s)]=ref
   supports.append(dict(source=cap,state=s,prefix_nonedge_fraction=support))
   for off in [1,2]:
    _,g=examine(load(idx+off,s)['x'],ref)
    rows.append(dict(comparison='within_'+cap,run=off+1,state=s,reference='source_common_mean3',**g))
 np.savez_compressed(O/'source_mean_references.npz',**{cap+'_'+s:v for (cap,s),v in refs.items()})
 for ca,cb,ia,ib in PAIRS:
  pair=ca+'_'+cb;v=np.array(frozen[pair]['sign'])
  for s in ST:
   ref=refs[(ca,s)]
   for off in [0,1,2]:
    xx=load(ib+off,s)['x'];xx=xx[:64] if off==0 else xx
    _,g=examine(xx*v[None,:,None],ref)
    rows.append(dict(comparison=pair,run=off+1,state=s,reference='source_common_mean3',**g))
 df=pd.DataFrame(rows);df.to_csv(O/'mean_reference_controls.csv',index=False);pd.DataFrame(supports).to_csv(O/'mean_reference_support.csv',index=False)
 print(df.groupby(['comparison','run']).agg(passes=('common_model_pass','sum'),states=('state','size'),corr=('predictive_corr_min_median','median')).to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
