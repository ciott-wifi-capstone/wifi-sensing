"""Exploratory controls, reported separately from the frozen primary analysis."""
import itertools
import json

import numpy as np
from threadpoolctl import threadpool_limits

import analyze_captures as a


def read_covariances():
    covariances = {}
    for batch in a.CAPTURES:
        for run in (1, 2, 3):
            folder = a.DESKTOP / batch / "capture" / f"run{run}"
            meta = a.read(folder / "run.json")
            path = folder / meta["raw_path"]
            x, host, _ = a.decoder.decode(path)
            start, order, guard = a.decoder.align(folder/"events.log", meta["start_ns"]/1e9)
            ta, _ = a.frame_details(path)
            visits, _ = a.select_visits((host-meta["start_ns"])/1e9, start, order,
                                        meta["requested_seconds"], guard, ta==a.APPARENT_TA)
            for state in a.STATES:
                ids = np.concatenate([idx for _, idx in visits[state]])
                covariances[batch, run, state] = a.spatial(x[ids].astype(np.complex128))
    return covariances


def empirical_repeatability(covariances):
    """Measure repeatability if every tone has its own unconstrained phase reference."""
    checks = []
    for state in a.STATES:
        los = a.los_response(state)
        r1 = a.relative(covariances["Kong1",1,state])
        for run in (2,3):
            other = a.relative(covariances["Kong1",run,state])
            for j in (1,2):
                # Relative effective offset calibrated only from empty R1 at this tone.
                offset = np.exp(1j*np.angle(r1[j-1]*los[j].conj()))
                desired = los[j]*offset
                good = (abs(r1[j-1])>=.5) & (abs(other[j-1])>=.5)
                err = np.degrees(abs(np.angle(other[j-1][good]*desired[good].conj())))
                p90 = np.quantile(err,.9) if len(err) else np.nan
                checks.append(dict(state=state, chain=j+1, empty_target_run=run, support=good.mean(),
                                   median_deg=np.median(err) if len(err) else np.nan,p90_deg=p90,
                                   pass_repeatability_only=good.mean()>=.8 and p90<=15))
    a.save("sensitivity_empty_per_tone.csv", checks)


def sign_control(covariances):
    train = np.arange(len(a.F)) % 3 != 2
    candidates = []
    for signs in itertools.product((1,-1),repeat=2):
        full = np.array([1,*signs])
        factors = np.array([full[j]*full[i] for i,j in a.PAIRS])
        terms = []
        for s in ["G1","G2"]:
            k0 = a.relative(covariances["Kong1",1,s])[:,train]
            occ = a.relative(covariances["Kong1A",1,s])[:,train]
            terms.extend(np.real((occ*k0.conj())*factors[:,None]).ravel())
        candidates.append((np.mean(terms),full))
    best = max(candidates,key=lambda item:item[0])[1]
    a.save("sensitivity_sign_development.csv", [dict(signs="/".join(map(str,z)),score=sc,
                                                    chosen=bool(np.array_equal(z,best))) for sc,z in candidates])
    results = []
    for state in a.STATES:
        k0 = covariances["Kong1",3,state][~train]
        occ = covariances["Kong1A",3,state][~train]
        for name, signs in [("raw",np.ones(3)),("fitted_sign_only",best)]:
            transformed = occ*signs[None,:,None]*signs[None,None,:]
            metrics = a.compare(k0,transformed)
            results.append(dict(state=state,both_states_unseen=state not in ["G1","G2"],
                                model=name,**metrics))
    a.save("sensitivity_sign_holdout.csv",results)


def numerical_controls():
    state = "G2"
    angles = a.ANGLE
    vectors = a.steering(state)
    target = vectors[:,:,angles==21][:,:,0]
    cov = target[:,:,None]*target[:,None,:].conj()+.02*np.eye(3)[None,:,:]
    peak = angles[a.angular_spectra(cov,vectors)[0].argmax()]
    assert peak==21
    rng=np.random.default_rng(20261105)
    x=rng.normal(size=(60,3,30))+1j*rng.normal(size=(60,3,30))
    common=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(60,1,30))) * rng.uniform(.2,3,size=(60,1,1))
    invariant_error=float(np.max(abs(a.spatial(x)-a.spatial(x*common))))
    assert invariant_error<1e-12
    synthetic = (target.T*np.exp(-2j*np.pi*(a.F-a.F.mean())*32e-9))[None,:,:]
    joint_spec = a.joint.music(a.joint.covariance(synthetic),
                               a.joint.steering(a.Y[np.array(a.MAP[state])-1]), q=1)[:,:-1]
    ai,ti = np.unravel_index(joint_spec.argmax(),joint_spec.shape)
    assert angles[ai]==21 and a.joint.TAU[ti]==32
    return dict(synthetic_spatial_angle_deg=peak,synthetic_joint_peak_deg=float(angles[ai]),
                synthetic_joint_delay_ns=float(a.joint.TAU[ti]),covariance_invariance_error=invariant_error)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        frozen=a.read(a.HERE/"blind_output_hashes.json")
        assert all(a.sha(a.HERE/name)==v for name,v in frozen.items())
        covariances=read_covariances()
        empirical_repeatability(covariances)
        sign_control(covariances)
        checks=numerical_controls()
        checks["primary_hashes_verified"]=len(frozen)
        (a.HERE/"numerical_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf8")
        print(checks)
