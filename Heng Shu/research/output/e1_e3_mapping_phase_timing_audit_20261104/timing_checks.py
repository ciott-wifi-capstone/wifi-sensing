from audit import *
def select(z,guard_ms=None,shift_ms=0):
 t=z['time_s'];starts=z['starts']+shift_ms/1000
 guard=float(z['guard']) if guard_ms is None else guard_ms/1000
 out={s:[] for s in ST}
 for st,si in zip(starts[:-1],z['states'][:-1]):
  if st+.025>z['duration'] or st<0:continue
  l,r=np.searchsorted(t,[st+guard,st+.025-guard]);out[ST[si]].extend(range(l,r))
 return {s:np.array(i,int) for s,i in out.items()}
def transitions(z):
 x=z['x'];t=z['time_s'];v=np.array([x[:,i]*x[:,j].conj() for i,j in PA]).transpose(1,0,2)
 v=v/np.maximum(abs(v),1e-30)
 jumps=np.median(abs(np.angle(v[1:]*v[:-1].conj()))*180/np.pi,axis=(1,2))
 mid=(t[1:]+t[:-1])/2;gap=t[1:]-t[:-1]
 starts=z['starts'];ei=np.searchsorted(starts,mid).clip(1,len(starts)-1)
 left=ei-1;nearest=np.where(abs(mid-starts[left])<abs(mid-starts[ei]),left,ei)
 residual=(mid-starts[nearest])*1000
 good=(gap<=.003)&(mid>=.05)&(mid<=z['duration']-.05)
 return pd.DataFrame(dict(mid_s=mid[good],boundary_residual_ms=residual[good],phase_jump_deg=jumps[good],gap_ms=gap[good]*1000))
def main():
 variants=[('baseline',None,0),('guard4',4,0),('guard6',6,0),('guard9',9,0),('guard6_shift-3',6,-3),('guard6_shift+3',6,3)]
 data={i:np.load(R/'raw_cache'/f'{i}.npz') for i in [78,79,80,84,85,86]}
 extracted={};counts=[];rows=[];transitionrows=[];tstats=[];offsetcurves=[]
 for i,z in data.items():
  for name,g,shift in variants:
   ids=select(z,g,shift)
   for s in ST:
    extracted[(i,name,s)]=z['x'][ids[s]].astype(complex)
    counts.append(dict(index=i,variant=name,state=s,packets=len(ids[s])))
  d=transitions(z);d['index']=i;transitionrows.append(d)
  for segment,mask in [('early',d.mid_s<z['duration']/2),('late',d.mid_s>=z['duration']/2)]:
   q=d[mask]
   bins=np.arange(-12.5,13.5,1);cent=(bins[1:]+bins[:-1])/2
   binned=[]
   for a,b,c in zip(bins[:-1],bins[1:],cent):
    v=q[(q.boundary_residual_ms>=a)&(q.boundary_residual_ms<b)].phase_jump_deg
    med=float(v.median()) if len(v)>=5 else np.nan;binned.append(med)
    offsetcurves.append(dict(index=i,segment=segment,residual_bin_ms=c,count=len(v),median_phase_jump_deg=med))
   peak=float(cent[np.nanargmax(binned)])
   near=q[q.boundary_residual_ms.abs()<=2.5].phase_jump_deg
   far=q[q.boundary_residual_ms.abs()>=5].phase_jump_deg
   tstats.append(dict(index=i,segment=segment,peak_jump_offset_ms=peak,near_switch_jump_median_deg=float(near.median()),deep_interior_jump_median_deg=float(far.median()),deep_interior_jump_p90_deg=float(far.quantile(.9))))
 mw=np.load(R/'frozen_phase_models.npz')
 for name,g,shift in variants:
  for i in [79,80,84,85,86]:
   pair='E1_E1' if i<84 else 'E1_E3'
   for s in ST:
    src=extracted[(78,name,s)];tar=extracted[(i,name,s)]
    for mode in ['raw','pi','phase_slope'] if i>=84 else ['raw']:
     w=mw['E1_E3_'+mode] if i>=84 else np.ones((3,30))
     rows.append(dict(pair=pair,index=i,variant=name,state=s,model=mode,**metrics(src,tar*w[None])))
 for name,d in [('timing_guard_counts',counts),('timing_sensitivity',rows),('transition_summary',tstats),('transition_curves',offsetcurves)]:pd.DataFrame(d).to_csv(R/f'{name}.csv',index=False)
 pd.concat(transitionrows).to_csv(R/'packet_transition_jumps.csv',index=False)
 print(pd.DataFrame(rows).query('index==86').groupby(['variant','model']).agg(amp=('amplitude_distance','median'),spatial=('spatial_distance','median'),phase=('phase_median_deg','median'),passed=('gate','sum')).to_string())
 print(pd.DataFrame(tstats).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
