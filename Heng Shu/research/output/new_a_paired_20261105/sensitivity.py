"""Exploratory after the frozen result: empirical R1 calibration and joint-peak stability."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
# The legacy 2D code also owns a module called 'analyze'. Keep this pilot separate.
spec=importlib.util.spec_from_file_location("paired_pilot",ROOT/"analyze.py")
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
b=p.base


def matched_raw():
    result={}
    for r in p.LABEL:
        folder=p.RAW/"capture"/f"run{r}"
        meta=b.read(folder/"run.json")
        file=folder/meta["raw_path"]
        x,host,_=b.decoder.decode(file)
        ta,_=b.frame_details(file)
        times=(host-meta["start_ns"])/1e9
        starts,order,g=b.decoder.align(folder/"events.log",meta["start_ns"]/1e9)
        bins,_=b.select_visits(times,starts,order,10.,g,ta==b.APPARENT_TA)
        for state,v in bins.items():
            ids=np.concatenate([idx for _,idx in v])
            result[r,state]=x[ids].astype(complex), times[ids]
    return result


def empirical_model(samples):
    """Unconstrained effective per-tone response estimated using R1 first half only."""
    rows, changes=[], []
    from_data={}; corrected={}
    for s in b.STATES:
        x,t=samples[1,s]
        q=b.relative(b.spatial(x[t<5]))
        los=b.los_response(s)
        h=np.array([np.ones(30,complex),np.exp(1j*np.angle(q[0]*los[1].conj())),
                    np.exp(1j*np.angle(q[1]*los[2].conj()))])
        for r in p.LABEL:
            x,t=samples[r,s]
            corrected[r,s]=x*h.conj()[None],t
            xx,tt=corrected[r,s]
            vectors=b.steering(s)
            spectra=b.angular_spectra(b.spatial(xx),vectors)
            parts=[b.angular_spectra(b.spatial(z),vectors)[0] for z in p.time_blocks(xx,tt)]
            peak=b.peak_summary(spectra,parts)[0]
            from_data[r,s]=spectra[0]
            rows.append(dict(state=s,run=r,dominant_deg=peak["peak_deg"],both_bands=peak["both_bands"],
                             quarter_support=peak["time_blocks"],
                             low_band_dominant_deg=b.ANGLE[spectra[1].argmax()],
                             high_band_dominant_deg=b.ANGLE[spectra[2].argmax()],
                             quarter_dominant_deg="/".join(f"{b.ANGLE[z.argmax()]:g}" for z in parts)))
        # Apply the existing *same* whitening+steering rule after calibration, never raw CSI subtraction.
        for ref,empty_control in [(1,3),(3,1)]:
            x,t=corrected[ref,s]
            cov=b.spatial(x[t<5])
            model=p.background.prepare(cov,s)
            dev=[b.spatial(z) for z in p.time_blocks(x,t,5,10)]
            td=max(b.distance(z,cov) for z in dev)
            ts=np.maximum(1,np.stack([p.background.scores(z,model) for z in dev]).max(axis=(0,2)))
            for test in [2,empty_control]:
                y,t=corrected[test,s]
                cy=b.spatial(y)
                sc=p.background.scores(cy,model)
                blocks=[b.spatial(v) for v in p.time_blocks(y,t)]
                full=p.background.peaks(sc[0],ts[0]) if b.distance(cy,cov)>td else []
                for i in full:
                    def match(curve,limit):
                        return np.any(abs(b.ANGLE[p.background.peaks(curve,limit)]-b.ANGLE[i])<=9)
                    temporal=sum(b.distance(z,cov)>td and match(p.background.scores(z,model)[0],ts[0]) for z in blocks)
                    bands=all(match(sc[k],ts[k]) for k in [1,2])
                    changes.append(dict(state=s,reference=ref,test=test,angle=b.ANGLE[i],ratio=sc[0,i],
                                        quarter_support=temporal,both_bands=bands,stable=bool(bands and temporal>=3)))
    for r in p.LABEL:
        combined=np.mean([from_data[r,s] for s in b.GROUPS],axis=0)
        rows.append(dict(state="G fusion",run=r,dominant_deg=b.ANGLE[np.argmax(combined)]))
    p.write("exploratory_empirical_angular_peaks.csv",rows)
    np.savez_compressed(ROOT/"exploratory_empirical_spectra.npz", angles=b.ANGLE,
                        **{f"{s}_R{r}":curve for (r,s),curve in from_data.items()})
    p.write("exploratory_empirical_bg_peaks.csv",changes,
            columns=["state","reference","test","angle","ratio","quarter_support","both_bands","stable"])


def joint_check(samples):
    rows=[]
    ja=b.joint.steering(b.Y[np.array(b.MAP["C1"])-1])
    for r in p.LABEL:
        x,t=samples[r,"C1"]
        for block,xx in enumerate([x,*p.time_blocks(x,t)]):
            rcov=b.joint.covariance(xx)
            for q in [2,3,4]:
                z=b.joint.music(rcov,ja,q=q)[:,:-1]
                ai,ti=np.unravel_index(z.argmax(),z.shape)
                rows.append(dict(run=r,quarter=block,model_order=q,peak_deg=b.ANGLE[ai],delay_parameter_ns=b.joint.TAU[ti]))
    p.write("exploratory_C1_joint_time_order.csv",rows)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        assert all(b.sha(ROOT/k)==v for k,v in b.read(ROOT/"frozen_outputs.json").items())
        samples=matched_raw()
        empirical_model(samples)
        joint_check(samples)
        print("Supplemental checks completed.")
