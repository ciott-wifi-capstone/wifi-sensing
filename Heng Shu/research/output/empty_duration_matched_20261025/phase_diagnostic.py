"""Describe frequency-wise circular phase differences; never apply rotations."""
from analyze import *
def main():
    mm=pd.read_csv(R/'input_manifest.csv');inv={};out=[];traces=[]
    for row in mm.itertuples(index=False):
        if row.capture_id in ['E2','E4']:x,_,_=dec.decode(Path(row.path))
        for state in rat.MAP:
            if row.capture_id in ['E1','E3']:z=np.load(PRE/'cache'/f'{row.index:03d}_{state}.npz')['x']
            else:z=rat.unit(x[np.load(OLD/'packets'/f'{row.index:03d}_{state}.npz')['packet_id']].astype(complex))
            inv[(row.index,state)]=invariants(z)
    for ca,cb in [('E1','E2'),('E3','E4')]:
        for ia in mm[mm.capture_id==ca]['index']:
            for ib in mm[mm.capture_id==cb]['index']:
                for state in rat.MAP:
                    a,b=inv[(ia,state)],inv[(ib,state)]
                    for j,label in enumerate(['chain1-chain2','chain2-chain3','chain1-chain3']):
                        mask=(a['pair_coherence'][j]>=.5)&(b['pair_coherence'][j]>=.5)
                        delta=rat.wrap_phase(b['pair_angle'][j]-a['pair_angle'][j]);v=delta[mask];avg=np.mean(np.exp(1j*v))
                        out.append(dict(pairing=ca+'_'+cb,index_a=int(ia),index_b=int(ib),state=state,chain_pair=label,supported_tones=int(mask.sum()),circular_mean_deg=float(np.angle(avg)*180/np.pi),circular_concentration=float(abs(avg)),median_abs_difference_deg=float(np.median(abs(v))*180/np.pi),median_distance_to_180_deg=float(np.median(abs(np.pi-abs(v)))*180/np.pi),fraction_within_20deg_of_180=float(np.mean(abs(np.pi-abs(v))<np.deg2rad(20)))))
                        for k in range(30):traces.append(dict(pairing=ca+'_'+cb,index_a=int(ia),index_b=int(ib),state=state,chain_pair=label,frequency_offset_mhz=rat.F[k]/1e6,delta_deg=delta[k]*180/np.pi,supported=bool(mask[k])))
    df=pd.DataFrame(out);df.to_csv(R/'results/chain_pair_phase.csv',index=False);pd.DataFrame(traces).to_csv(R/'results/phase_traces.csv',index=False)
    s=df.groupby(['pairing','state','chain_pair'])[['median_abs_difference_deg','median_distance_to_180_deg','circular_concentration','fraction_within_20deg_of_180']].median().reset_index();s.to_csv(R/'results/phase_summary.csv',index=False);print(s.to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
