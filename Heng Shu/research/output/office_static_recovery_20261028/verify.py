"""Source, bounded-sample, central-window and synthetic model checks."""
import analyze as a
import numpy as np
import json,hashlib
from threadpoolctl import threadpool_limits
def main():
 c,h,nic,dn,dc,perm=a.dec.decode(a.D/'session.dat');host=a.dec.host_times(a.D,'2026-09-07');rows=[]
 for run in [1,2,3]:
  issued,starts,states,guard,info=a.dec.events(a.D/f'run{run}_events.log');ids={s:[] for s in a.STATES};cent={s:[] for s in a.STATES}
  for ei in range(len(starts)-1):
   if starts[ei]+.025>issued+30:continue
   i,j=np.searchsorted(host,[starts[ei]+guard,starts[ei]+.025-guard]);u,v=np.searchsorted(host,[starts[ei]+.0075,starts[ei]+.0175])
   if j-i>=10:ids[a.STATES[states[ei]]].extend(range(i,j));cent[a.STATES[states[ei]]].extend(range(u,v))
  for s in a.STATES:
   full=a.rat.unit(c[ids[s]].astype(complex));mid=a.rat.unit(c[cent[s]].astype(complex));z=np.load(a.O/'packets'/f'R{run}_{s}.npz');x=z['x']
   rows.append(dict(run=run,state=s,valid_packets=len(full),central_packets=len(mid),sample_full_spatial_distance=a.dist(a.spatial(x),a.spatial(full)),sample_full_cov_distance=a.dist(a.cov(x),a.cov(full)),central_full_spatial_distance=a.dist(a.spatial(mid),a.spatial(full))))
 a.csv('sampling_and_guard_sensitivity.csv',rows)
 # Rank-three noiseless narrowband spatial model with the actual frequency samples.
 angles=np.array([-30.,9.,45.]);taus=np.array([-120.,80.,216.]);ys=a.Y[np.array(a.MAP['G1'])-1]
 H=np.sum(np.exp(-2j*np.pi*(-5.2e9/299792458*ys[:,None,None]*np.sin(np.deg2rad(angles))[None,None,:]+a.F[None,:,None]*taus[None,None,:]*1e-9)),axis=2)
 r=a.cov(a.rat.unit(np.repeat(H[None],8,axis=0)));s=a.music(r,a.steer('G1'));scores=[float(s[np.argmin(abs(a.ANG-an)),np.argmin(abs(a.TAU-ta))]) for an,ta in zip(angles,taus)]
 assert min(scores)>.99
 for e in json.loads((a.D/'delivery_manifest.json').read_text())['files']:assert hashlib.sha256((a.D/e['path']).read_bytes()).hexdigest()==e['sha256']
 checks={'original_manifest_still_matches':True,'synthetic_music_true_location_scores':scores,'sample_full_cov_distance_max':max(r['sample_full_cov_distance'] for r in rows),'central_full_spatial_distance_max':max(r['central_full_spatial_distance'] for r in rows),'state_count':len(rows),'no_physical_localization_claim':True}
 a.O.joinpath('results/verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
