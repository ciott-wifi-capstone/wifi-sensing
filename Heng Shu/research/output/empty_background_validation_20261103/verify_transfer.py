from transfer_common import *
import json
ref=np.load(R/'cache/078_G1.npz')['x'][0]
synthetic=np.array([ref*np.exp(-1j*(2*np.pi*rat.F*(i/10)*1e-9+.03*i))[None,:] for i in range(64)])
_,good=examine(synthetic,ref);assert good['common_model_pass'],good
bad=synthetic.copy();bad[:,0]*=-1
_,fail=examine(bad,ref);assert not fail['common_model_pass']
a=pd.read_csv(R/'transfer_background_scores.csv')
u=a[a['mode']=='unaligned'].set_index(['pair','state','target_run'])
g=a[a['mode']=='gated'].set_index(['pair','state','target_run'])
err=float(abs(u.max_ratio-g.max_ratio).max());assert err<1e-8 and g.accepted_packets.sum()==0
out=dict(synthetic_common_phase_timing_pass=good,synthetic_single_chain_pi_rejected=fail,failed_gate_fallback_max_ratio_difference=err,accepted_real_packets=0)
(R/'transfer_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
