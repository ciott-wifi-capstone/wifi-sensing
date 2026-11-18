"""Additional hardware-shape hypothesis: R1 affine phase on training tones only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import validate_coherent as c
ROOT=Path(__file__).resolve().parent

def relative(e,y):
    _,v=np.linalg.eigh(c.herm(e));h=v[:,:,-1]
    lo=np.exp(-2j*np.pi/c.base.C*c.F[:,None]*(np.sqrt(3.65**2+y*y)-3.65)[None,:])
    z=(h/h[:,:1])/(lo/lo[:,:1])
    return np.exp(1j*np.angle(z)).T

def main():
    protocol=dict(status='Additional reference-model diagnostic before truth scoring; no replacement of primary.',
        calibration='R1 first8 only. Known exact TX propagation divided out of relative antenna response. Fit a constant phase and baseband-frequency slope using 20 tones; untouched 10 tones and R1/R3 held blocks audit the assumption.',
        caveat='A good affine fit does not distinguish fixed hardware from an approximately affine multipath mixture. No portable or absolute calibration is asserted.',
        no_more_search='Raw, per-tone empirical and affine reference are all retained, with no point-specific selection.')
    (ROOT/'affine_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[];checks=[];params=[];saved={}
    for site in 'ABCDE':
        z=np.load(ROOT/f'{site}_complex_covariances.npz')
        for arm in ('G2','tree6','tree8'):
            raw=z[arm];y=c.base.Y[3:6] if arm=='G2' else c.base.Y
            ratio=relative(raw[0,:8].mean(0),y)
            rot=[]
            for j in range(len(y)):
                t,p,co,edge=c.rat.fit(ratio[j:j+1],np.ones((1,30)),c.TRAIN[None,:])
                r=np.exp(1j*(2*np.pi*c.FB*t*1e-9+p));rot.append(r)
                params.append(dict(point=site,arm=arm,element=j+1,delay_ns=t,phase_deg=p*180/np.pi,fit_corr=co,edge=edge))
            rot=np.array(rot).T
            for block,data in [('fit_tones_hold',raw[0,:8].mean(0)),('R1_held',raw[0,8:].mean(0)),('R3_held',raw[2].mean(0))]:
                zz=relative(data,y)*rot.T
                ph=abs(np.angle(zz[1:,~c.TRAIN]))*180/np.pi
                checks.append(dict(point=site,arm=arm,block=block,held_phase_p90_deg=float(np.quantile(ph,.9)),
                                   pass_phase=bool(np.quantile(ph,.9)<=15)))
            d=np.array([c.transform(x,rot) for x in raw]);ref=d[0,:8].mean(0)
            train=np.array([d[0,:4].mean(0),d[0,4:8].mean(0)])
            views={};power={}
            for block,data in [('predict',d[1,:12].mean(0)),('held',d[1,12:].mean(0)),('full',d[1].mean(0)),('empty',d[2].mean(0))]:
                v,p,n=c.music_profile(ref,data,train,y)
                views[block]=v;power[block]=p;saved[f'{site}_{arm}_{block}']=v
            a=c.peak(views['predict'][0]);h=[c.peak(v) for v in views['held']];blank=[c.peak(v) for v in views['empty']]
            consistent=np.isfinite(h+[a]).all() and max(abs(np.array(h)-a))<=9
            bg=np.isfinite(blank).all() and max(abs(np.array(blank)-a))<=9
            pr=power['predict']/max(power['empty'],1e-15)
            rows.append(dict(point=site,arm=arm,calibration='R1_affine',prediction=a,all_R2_peak=c.peak(views['full'][0]),
                held_R2_peak=h[0],held_low=h[1],held_high=h[2],empty_peak=blank[0],foreground_empty_ratio=pr,
                spectral_gate=bool(np.isfinite(a) and abs(a)<87 and consistent and not bg and pr>1),physical_release=False))
    pd.DataFrame(rows).to_csv(ROOT/'blind_affine.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(ROOT/'affine_reference_checks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(params).to_csv(ROOT/'affine_parameters.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(ROOT/'affine_profiles.npz',angle=c.ANG,**saved)
    names=['affine_protocol.json','validate_affine_reference.py','blind_affine.csv','affine_reference_checks.csv','affine_parameters.csv','affine_profiles.npz']
    (ROOT/'affine_freeze.json').write_text(json.dumps({n:c.base.sha(ROOT/n) for n in names},indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
