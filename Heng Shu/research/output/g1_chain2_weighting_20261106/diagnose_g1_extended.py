"""Additional cause diagnostics; no estimator weights or predictions are changed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import analyze_g1 as a

R=a.ROOT;c=a.c

def main():
    frozen=json.loads((R/'prediction_freeze.json').read_text())
    assert all(c.base.sha(R/k)==v for k,v in frozen.items())
    rows=[];block=[];invariant=[];settling=[]
    for site in a.SITES:
        p=np.load(a.OLD/f'{site}_selected.npz')['switched'].astype(complex)
        mask=np.zeros((3,30));mask[[0,2],:]=c.TRAIN
        held=np.where(~c.TRAIN)[0]
        for r in range(3):
            for cy in range(24):
                raw=p[r,cy,0];x=c.unit(raw);ref=x[0]
                even=[]
                for pi in a.TRAIN_PACKETS:
                    t,ph,_,_=c.rat.fit(x[pi],ref,mask)
                    even.append(c.rat.apply(x[pi],t,ph))
                ref=np.mean(even,axis=0)
                for pi in a.HELD_PACKETS:
                    t,ph,_,_=c.rat.fit(x[pi],ref,mask)
                    z=c.rat.apply(x[pi],t,ph)
                    for m in range(3):
                        for k in held:
                            valid=abs(z[m,k])>0 and abs(ref[m,k])>0
                            rows.append(dict(point=site,run=r+1,cycle=cy,packet=int(pi),chain=m+1,tone=int(k),
                                raw_abs=float(abs(raw[pi,m,k])),valid=valid,
                                held_phase_error_deg=float(abs(np.angle(z[m,k]*ref[m,k].conj()))*180/np.pi) if valid else np.nan))
            x=p[r,:,0]
            checks=a.pair_checks(x)
            # Both pairs share chain 1 and cancel packet phase/slope exactly at each tone.
            for _,m,err,valid in checks[:2]:
                for cy in range(24):
                    tr=np.median(abs(x[cy,a.TRAIN_PACKETS,m,:]),axis=0)
                    tr1=np.median(abs(x[cy,a.TRAIN_PACKETS,0,:]),axis=0)
                    for pi,op in enumerate(a.HELD_PACKETS):
                        for k in range(30):
                            if valid[cy,pi,k] and abs(x[cy,op,0,k])>=12 and tr1[k]>=12:
                                invariant.append(dict(point=site,run=r+1,cycle=cy,target_chain=m+1,
                                    raw_abs=float(abs(x[cy,op,m,k])),training_abs=float(tr[k]),
                                    error_deg=float(abs(err[cy,pi,k])*180/np.pi)))
            for pair in ((0,1),(0,2)):
                z=x[:,:,pair[0],:]*x[:,:,pair[1],:].conj()
                for name,ix in [('early3',np.arange(3)),('late3',np.arange(3,6))]:
                    v=z[:,ix];ok=abs(v)>0
                    unit=np.divide(v,abs(v),out=np.zeros_like(v),where=ok)
                    count=ok.sum(1)
                    coherent=np.divide(abs(unit.sum(1)),count,out=np.zeros(count.shape,float),where=count>0)
                    sd=np.sqrt(-2*np.log(np.clip(coherent,1e-15,1)))*180/np.pi
                    sd=sd[count==3]
                    rel=abs(x[:,ix,pair[1],:])**2/np.maximum(abs(x[:,ix,0,:])**2,1e-15)
                    settling.append(dict(point=site,run=r+1,pair=f'{pair[0]+1}-{pair[1]+1}',part=name,
                        valid_cycle_tones=len(sd),circular_sd_median_deg=np.median(sd),
                        circular_sd_p90_deg=np.quantile(sd,.9),target_to_chain1_median_db=np.median(10*np.log10(np.maximum(rel,1e-15)))))
            for name,ix in [('R1_fit',np.arange(8)),('R1_hold',np.arange(8,24))] if r==0 else [(f'R{r+1}_all',np.arange(24))]:
                val=[np.nanmean(1-np.cos(err[ix])) for _,_,err,_ in checks]
                block.append(dict(point=site,part=name,V12=val[0],V13=val[1],V23=val[2],
                    reliability_ratio=val[1]/max((val[0]+val[2])/2,1e-15)))
    d=a.save('g1_matched_tone_prediction.csv',rows)
    a.save('g1_reliability_holdout.csv',block)
    bins=[0,3,6,12,24,48,np.inf]
    d['amplitude_bin']=pd.cut(d.raw_abs,bins,right=False)
    groups=[]
    for (chain,band),g in d.groupby(['chain','amplitude_bin'],observed=True):
        v=g.held_phase_error_deg.dropna()
        groups.append(dict(chain=chain,amplitude_bin=str(band),possible=len(g),valid=len(v),
            phase_median_deg=v.median(),phase_p90_deg=v.quantile(.9),
            visits=g[['point','run','cycle']].drop_duplicates().shape[0]))
    a.save('g1_amplitude_matched_phase.csv',groups)
    corr=[]
    for (site,run,chain),g in d.groupby(['point','run','chain']):
        g=g.dropna(subset=['held_phase_error_deg'])
        rho=spearmanr(g.raw_abs,g.held_phase_error_deg).statistic
        corr.append(dict(point=site,run=run,chain=chain,rho_amplitude_phase_error=rho,samples=len(g),
                         note='descriptive correlated samples; not a significance test'))
    a.save('g1_amplitude_error_association.csv',corr)
    a.save('g1_switch_settling.csv',settling)
    di=pd.DataFrame(invariant)
    di['bin']=pd.cut(di.raw_abs,bins,right=False)
    di['training_bin']=pd.cut(di.training_abs,bins,right=False)
    matched=[]
    for (chain,binname),g in di[di['bin']==di['training_bin']].groupby(['target_chain','bin'],observed=True):
        matched.append(dict(target_chain=chain,amplitude_bin=str(binname),samples=len(g),
            visits=g[['point','run','cycle']].drop_duplicates().shape[0],
            phase_median_deg=g.error_deg.median(),phase_p90_deg=g.error_deg.quantile(.9)))
    a.save('g1_common_phase_invariant_amplitude_match.csv',matched)
    t=pd.read_csv(R/'g1_tone_profiles.csv');prof=[]
    for site in a.SITES:
        z=t[t.point==site].pivot(index='tone',columns='run',values='relative_chain2_db')
        prof.append(dict(point=site,R1_R3_profile_spearman=spearmanr(z[1],z[3]).statistic,
            median_abs_R1_R3_db=(z[1]-z[3]).abs().median(),max_abs_R1_R3_db=(z[1]-z[3]).abs().max(),
            median_abs_R2_R1_db=(z[2]-z[1]).abs().median(),max_abs_R2_R1_db=(z[2]-z[1]).abs().max()))
    a.save('g1_frequency_profile_repeat.csv',prof)
    print(pd.DataFrame(groups).to_string(index=False))
    print(pd.DataFrame(prof).to_string(index=False))
    print('All chains amplitude/phase correlation, median across point-runs:')
    print(pd.DataFrame(corr).groupby('chain').rho_amplitude_phase_error.median().to_string())
    print('Common phase invariant, matched held and reference amplitudes, chain1>=12:')
    print(pd.DataFrame(matched).to_string(index=False))
    print('Early/late three-packet phase dispersion, median over the 15 point-runs:')
    print(pd.DataFrame(settling).groupby(['pair','part'])[['circular_sd_median_deg','circular_sd_p90_deg']].median().to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
