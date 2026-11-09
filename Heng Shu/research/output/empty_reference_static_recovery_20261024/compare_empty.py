"""Bounded invariants, cross-record prediction and uncalibrated empty-null spectra."""
import itertools
import numpy as np,pandas as pd
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits
from common import *
from reference_models import *

IDS=[78,79,80,84,85,86]
def metrics(a,b):
    support=(a['pair_coherence']>=.5)&(b['pair_coherence']>=.5)
    ph=abs(rat.wrap_phase(a['pair_angle']-b['pair_angle']))*180/np.pi
    return dict(spatial_distance=distance(a['spatial'],b['spatial']),amplitude_distance=distance(a['amplitude'],b['amplitude']),covariance_distance=distance(a['covariance'],b['covariance']),pair_phase_median_deg=float(np.median(ph[support])) if support.any() else None,pair_phase_support_fraction=float(support.mean()))

def main():
    blockrows=[];pairs=[];cross=[];nulls=[];thresholds=[];perms=[]
    selected=load(R/'results/development_selection.json')['selected_variant']
    for state in rat.MAP:
        vv={i:np.load(R/'cache'/f'{i:03d}_{state}.npz') for i in IDS}
        whole={i:invariants(v['x']) for i,v in vv.items()};steer=rat.steering(state)
        spectra={i:rat.music(whole[i]['covariance'],steer) for i in IDS}
        devdist=[];devnull=[]
        for i,v in vv.items():
            edges=np.linspace(0,10 if i<84 else 15,5)
            blocks=[v['x'][(v['relative_s']>=a)&(v['relative_s']<b)] for a,b in zip(edges[:-1],edges[1:])]
            for j,k in itertools.combinations(range(4),2):
                if min(len(blocks[j]),len(blocks[k]))<16:continue
                a,b=invariants(blocks[j]),invariants(blocks[k]);m=metrics(a,b)
                blockrows.append(dict(index=i,state=state,block_a=j,block_b=k,n_a=len(blocks[j]),n_b=len(blocks[k]),**m))
                if i in CFG['empty_development']:
                    devdist.append(m)
                    devnull.append(float(np.max(abs(rat.music(a['covariance'],steer)-rat.music(b['covariance'],steer)))))
            for ant in np.unique(v['ant_sel']):
                z=v['x'][v['ant_sel']==ant]
                if len(z)<16:continue
                perms.append(dict(index=i,state=state,ant_sel=int(ant),packets=len(z),spatial_distance_to_all=distance(spatial(z),whole[i]['spatial'])))
        limits={k:max(CFG['repeatability_absolute_floor'],CFG['invariant_repeatability_ratio_limit']*max(z[k] for z in devdist)) for k in ['spatial_distance','amplitude_distance','covariance_distance']}
        null_limit=max(devnull)
        thresholds.append(dict(state=state,**limits,empty_spectral_excess_limit=null_limit,development_pairs=len(devdist)))
        for i,j in itertools.combinations(IDS,2):
            typ='within_E1' if j<84 else ('within_E3' if i>=84 else 'E1_E3')
            variants={'full':(vv[i]['x'],vv[j]['x'])}
            for name,extra in [('guard_extra_1ms',.001),('guard_extra_2ms',.002)]:
                variants[name]=tuple(vv[n]['x'][vv[n]['margin_s']>=extra] for n in (i,j))
            z=[vv[n]['x'][vv[n]['relative_s']<10] for n in (i,j)]
            variants['equal_duration_10s']=tuple(z)
            cap=min(CFG['equal_packet_cap'],*(len(t) for t in z))
            variants['equal_duration_and_count']=tuple(t[np.linspace(0,len(t)-1,cap,dtype=int)] for t in z)
            for name,(a,b) in variants.items():
                if min(len(a),len(b))<16:
                    pairs.append(dict(index_a=i,index_b=j,state=state,comparison=typ,variant=name,n_a=len(a),n_b=len(b),status='insufficient_packets'));continue
                m=metrics(invariants(a),invariants(b));passes={k+'_pass':m[k]<=lim for k,lim in limits.items()}
                pairs.append(dict(index_a=i,index_b=j,state=state,comparison=typ,variant=name,n_a=len(a),n_b=len(b),status='computed',**m,**passes))
            # Empty-to-empty positive change of independently normalized MUSIC spectra.
            # This is a null diagnostic, NOT a white-background human detector.
            delta=np.maximum(spectra[j]-spectra[i],0);local=(delta==maximum_filter(delta,size=(5,5),mode='nearest'))&(delta>null_limit)
            idxs=np.argwhere(local);idxs=sorted(idxs,key=lambda z:-delta[tuple(z)])
            nulls.append(dict(index_reference=i,index_test=j,state=state,comparison=typ,threshold=null_limit,excess_max=float(delta.max()),threshold_exceeded=bool(delta.max()>null_limit),local_maxima_above_limit=len(idxs),strongest_angle_deg=float(rat.ANG[idxs[0][0]]) if idxs else None,strongest_delay_ns=float(rat.TAU[idxs[0][1]]) if idxs else None,meaning='uncalibrated empty spectral change; no person present'))
        # Fit only a target prefix; predict on later target packets and disjoint features.
        for src in CFG['empty_development']:
            ref,_=make_reference(vv[src]['x'],selected)
            for dst in CFG['empty_within_holdout']+CFG['empty_cross_holdout']:
                x=vv[dst]['x'];g=gate(x,ref);p=predict(x,ref)
                candidate=rat.apply(x[64::8],p[:,1],np.array([rat.fit(a,ref,rat.crossfit_masks()[0])[1] for a in x[64::8]]))
                err=rat.pair_error(x[64::8],candidate)
                cross.append(dict(index_reference=src,index_test=dst,state=state,variant=selected,**g,future_pred_corr_A=float(np.median(p[:,4])),future_pred_corr_C_candidate=float(np.median(p[:,6])),future_delay_residual_iqr_ns=rat.iqr(p[:,3]),future_phase_residual_sd_deg=rat.circ_sd(p[:,7])*180/np.pi,spatial_phase_preserved_deg=err,actual_cross_alignment_applied=False if not g['common_pass'] else None,meaning='candidate-only prediction until both prefix and future validation pass'))
                np.savez_compressed(R/'parameters'/f'cross_{src}_{dst}_{state}.npz',reference=ref,prediction=p)
        np.savez_compressed(R/'parameters'/f'empty_spectra_{state}.npz',indices=IDS,angles=rat.ANG,delays_ns=rat.TAU,spectra=np.stack([spectra[i] for i in IDS]))
        print('Compared',state,flush=True)
    for name,rows in [('within_run_blocks',blockrows),('empty_pair_comparisons',pairs),('cross_record_prediction',cross),('empty_null_spectral_changes',nulls),('development_thresholds',thresholds),('permutation_strata',perms)]:pd.DataFrame(rows).to_csv(R/'results'/f'{name}.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
