"""Projected covariance-difference MUSIC, a bounded experimental adaptation.

No coordinates of human standing points or measured errors are imported here.
The old frozen runner supplies only the state map, frequencies and steering.
This module neither loads real A--E caches nor runs physical error scoring.

Model: a stationary rank-one background b and a newly added coherent ray a.
If B b=0, B[(b+alpha*a)(b+alpha*a)^H-b*b^H]B is rank-one PSD.
Residual background, obstruction, hardware drift or noise changes can violate
that model. A positive change mode is a candidate, not an identified human.
"""
from pathlib import Path
import sys
import json
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'noncoherent_uniform_20261106'))
import predict_uniform as prior

STEER = prior.STEER
ANG = prior.ANG
ST = prior.ST
EYE = prior.EYE
herm = prior.herm
outer = prior.outer


def training_change_floor(bg):
    """Largest projected spectral norm of either R1 half minus fitted R1.

    These halves contribute to their shared reference and are not independent
    empty records. This is a deterministic development floor, not a calibrated
    false-alarm threshold. All operations stay in the unwhitened B subspace.
    """
    B = bg['B']
    delta = herm(B[None] @ (bg['train'] - bg['ref'][None]) @ B[None])
    return np.max(np.abs(np.linalg.eigvalsh(delta)), axis=(0, 2))


def state_profile(bg, target, state, kernel='PCD'):
    """Return band profiles, old raw residual-power control and rank metadata.

    Compatible with predict_uniform.state_profile. One positive change mode
    above the R1-only floor is required. Rank two is rejected because the
    projected three-chain space has only two dimensions; fitting both would
    leave no direction-sensitive MUSIC noise subspace. Negative modes are
    recorded and never silently reclassified as positive signal modes.
    """
    if kernel != 'PCD':
        raise ValueError('This module implements only the frozen PCD kernel.')
    a = STEER[state]
    ref, B = bg['ref'], bg['B']
    delta = herm(B @ (target - ref) @ B)
    vals, vec = np.linalg.eigh(delta)
    floor = training_change_floor(bg)
    # Numerical tolerance is not an adaptive statistical gate.
    tol = 1e-10 * np.maximum(1., np.max(np.abs(vals), axis=1))
    threshold = floor + tol
    positive = vals > threshold[:, None]
    negative = vals < -threshold[:, None]
    rank = positive.sum(axis=1)
    neg_rank = negative.sum(axis=1)
    dominance = bg['e'][:, -1] / np.maximum(bg['e'].sum(axis=1), 1e-12)
    use = (dominance >= .7) & (rank == 1)
    # A single positive eigenvalue has a nonzero gap from other positive modes.
    # Record all failed tones; do not turn rank-two changes into rank one.
    N = B - outer(vec[:, :, -1])
    num = np.maximum(np.einsum('kma,kmn,kna->ka', a.conj(), N, a).real, 0.)
    den = np.einsum('kma,kmn,kna->ka', a.conj(), B, a).real
    energy = np.sum(np.abs(a)**2, axis=1)
    profiles = []
    for ids in (np.arange(30), np.arange(15), np.arange(15, 30)):
        good = ids[use[ids]]
        valid = 2 * len(good) >= len(ids)
        if valid:
            n, d, e = num[good].sum(0), den[good].sum(0), energy[good].sum(0)
            cost = np.divide(n, d, out=np.full(len(ANG), np.nan), where=d >= .1*e)
        else:
            n = np.zeros(len(ANG)); d = np.zeros(len(ANG)); e = np.zeros(len(ANG))
            cost = np.full(len(ANG), np.nan)
        profiles.append(dict(num=n, den=d, energy=e, cost=cost, valid=valid, tones=len(good)))
    # This control is intentionally identical to the previous runner and is
    # not the PCD signal eigenvalue. The caller must label the two separately.
    raw_power = float(np.trace(B @ target @ B, axis1=1, axis2=2).real.mean()/2)
    meta = dict(valid_tones=int(use.sum()),
                q0=int((rank==0).sum()), q1=int((rank==1).sum()),
                q2=int((rank==2).sum()), q3=int((rank==3).sum()),
                negative_change_tones=int((neg_rank>0).sum()),
                second_positive_above_floor=int((vals[:,-2]>threshold).sum()),
                median_change_floor=float(np.median(floor)),
                median_positive_change=float(np.median(np.maximum(vals[:,-1],0.))),
                negative_dominant_tones=int((np.abs(vals[:,0])>np.maximum(vals[:,-1],0.)+tol).sum()))
    return profiles, raw_power, meta


def mathematical_checks():
    """Small algebra checks; no claim of field human-path validation."""
    nulls = []
    floor_checks = []
    for s in ST:
        h = STEER[s][:,:,np.flatnonzero(ANG==4)[0]]
        a = STEER[s][:,:,np.flatnonzero(ANG==-28)[0]]
        noise = .02*EYE
        ref = outer(h) + noise
        m = np.broadcast_to(ref,(3,24,30,3,3)).copy()
        bg = prior.background(m)
        target = outer(1.3*h + (.68+.45j)*a) + noise
        p, _, info = state_profile(bg,target,s)
        value = p[0]['cost'][np.flatnonzero(ANG==-28)[0]]
        assert info['q1']==30 and np.isfinite(value) and abs(value)<1e-8, (s,info,value)
        nulls.append(float(value))
        for empty in (ref, outer(1.6*h)+noise, outer(.4*h)+noise):
            p0, _, i0 = state_profile(bg,empty,s)
            assert i0['q0']==30 and i0['valid_tones']==0
            assert not np.isfinite(p0[0]['cost']).any()
        p2,_,i2 = state_profile(bg, ref+.5*bg['B'],s)
        assert i2['q2']==30 and i2['second_positive_above_floor']==30
        assert not np.isfinite(p2[0]['cost']).any()
        # Train halves with known +/- epsilon changes have floor epsilon.
        bg2=dict(bg)
        bg2['train']=np.stack([ref+.007*bg['B'],ref-.007*bg['B']])
        f=training_change_floor(bg2)
        assert np.allclose(f,.007,atol=1e-12)
        psmall,_,ismall=state_profile(bg2,ref+.003*bg['B'],s)
        assert ismall['q0']==30 and not np.isfinite(psmall[0]['cost']).any()
        floor_checks.append(float(np.max(np.abs(f-.007))))
    return dict(coherent_background_plus_ray_with_nonzero_white_noise=True,
                background_unchanged_rejected=True,
                background_amplitude_only_change_rejected=True,
                two_positive_projected_modes_rejected=True,
                R1_projected_spectral_norm_floor_verified=True,
                subfloor_change_rejected=True,
                true_steering_max_abs_cost=max(abs(x) for x in nulls),
                floor_max_absolute_error=max(floor_checks),
                field_data_run=False, unique_angle_recovery_tested=False,
                physical_human_path_validation=False)


if __name__ == '__main__':
    print(json.dumps(mathematical_checks(), indent=2))
