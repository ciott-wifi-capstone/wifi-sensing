"""Analyze the two named captures. Only the reporting step reads A's surveyed position."""
import os
import hashlib
import itertools
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
VALIDATION = OUTPUT / "static_human_aoa_validation_20261101"
sys.path.insert(0, str(VALIDATION / "provenance"))
import decoder_ht40 as decoder
sys.path.insert(0, str(OUTPUT / "static_analysis_20261018"))
import geometry_update as joint

GEOMETRY = json.loads((VALIDATION / "geometry.json").read_text(encoding="utf8"))
MAP = GEOMETRY["state_map"]
STATES = list(MAP)
GROUPS = STATES[:4]
PAIRS = [(0, 1), (0, 2), (1, 2)]
F = 5.23e9 + np.arange(-58, 59, 4) * 312500.
ANGLE = np.arange(-90., 91., 3.)
C = 299792458.
Y = np.array(GEOMETRY["array_y_local_m"])
DESKTOP = Path(os.environ.get('WIFI_DATA_ROOT', str(Path.home()/'Desktop')))
CAPTURES = ("Kong1", "Kong1A")
EVENT = re.compile(r"([\d.]+)\s+READY (G[1-4]|C[12]) chains=([\d,]+) t=(\d+) seq_idx=(\d)/6")
APPARENT_TA = os.environ.get("WIFI_TARGET_MAC", "")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / name, index=False, encoding="utf-8-sig")
    return frame


def identity_audit(folder):
    claims = read(folder / "manifest.json")["files"]
    manifest_rows = []
    for item in claims:
        path = folder / item["path"]
        exists = path.is_file()
        observed = sha(path) if exists else ""
        manifest_rows.append(dict(batch=folder.name, file=item["path"], expected_sha256=item["sha256"],
                                  observed_sha256=observed, ok=exists and observed == item["sha256"] and
                                  path.stat().st_size == item["bytes"]))
    ops = (folder / "operations.log").read_text(encoding="utf8")
    log_rows = []
    for line_number, line in enumerate(ops.splitlines(), 1):
        if not line.startswith('{"command":'):
            continue
        op = json.loads(line)
        cmd = op["command"]
        if cmd[0] in ("rmmod", "modprobe", "array_prepare_for_picoscenes") or cmd[-2:] == ["mon1", "del"]:
            log_rows.append(dict(batch=folder.name, source="operations.log", line=line_number,
                                 event=" ".join(cmd), returncode=op["returncode"]))
    for line_number, line in enumerate((folder / "rx_kernel.log").read_text(encoding="utf8").splitlines(), 1):
        if "loaded firmware version" in line or re.search(r"phy\d+: Selected rate control", line):
            log_rows.append(dict(batch=folder.name, source="rx_kernel.log", line=line_number, event=line, returncode=None))
    return manifest_rows, log_rows


def frame_details(path):
    data = path.read_bytes()
    offset = 0
    ta = []
    perm = Counter()
    while offset < len(data):
        size, _, _, segments = struct.unpack_from("<IIHB", data, offset)
        end = offset + size + 4
        cursor = offset + 11
        for _ in range(segments):
            length, length_name = struct.unpack_from("<IB", data, cursor)
            name = data[cursor+5:cursor+5+length_name].rstrip(b"\0")
            if name == b"CSI":
                selection = data[cursor+7+length_name+32]
                perm[str(tuple((selection >> (2*j)) & 3 for j in range(3)))] += 1
            cursor += length + 4
        # These captures include an 802.11 data frame after the PicoScenes segments.
        mpdu = data[cursor:end]
        ta.append(mpdu[10:16].hex(":") if len(mpdu) >= 24 and (mpdu[0] & 12) == 8 else "")
        offset = end
    return np.asarray(ta), perm


def select_visits(times, starts, state_numbers, duration, guard, target):
    selected = {s: [] for s in STATES}
    eligible = Counter()
    for event_index, (t, si) in enumerate(zip(starts[:-1], state_numbers[:-1])):
        if t < 0 or t+.025 > duration:
            continue
        state = STATES[si]
        eligible[state] += 1
        left, right = np.searchsorted(times, (t+guard, t+.025-guard))
        packet_ids = np.arange(left, right)
        packet_ids = packet_ids[target[packet_ids]]
        if len(packet_ids):
            selected[state].append((event_index, packet_ids))
    return selected, eligible


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=(1, 2), keepdims=True), 1e-30)


def spatial(x):
    z = normalized(x)
    return np.einsum("nmk,nlk->kml", z, z.conj(), optimize=True) / len(z)


def relative(cov):
    power = np.diagonal(cov, axis1=1, axis2=2).real
    return np.stack([cov[:, j, i] / np.sqrt(np.maximum(power[:, i]*power[:, j], 1e-30)) for i, j in PAIRS])


def distance(a, b):
    return float(2*np.linalg.norm(a-b) / max(np.linalg.norm(a)+np.linalg.norm(b), 1e-30))


def compare(a, b):
    qa, qb = relative(a), relative(b)
    good = (abs(qa) >= .5) & (abs(qb) >= .5)
    degrees = np.rad2deg(abs(np.angle(qb * qa.conj())))[good]
    amp_a = np.sqrt(np.maximum(np.diagonal(a, axis1=1, axis2=2).real, 0))
    amp_b = np.sqrt(np.maximum(np.diagonal(b, axis1=1, axis2=2).real, 0))
    return dict(covariance_distance=distance(a, b), amplitude_shape_distance=distance(amp_a, amp_b),
                coherence_magnitude_distance=distance(abs(qa), abs(qb)), reliable_fraction=good.mean(),
                phase_median_deg=float(np.median(degrees)) if len(degrees) else np.nan,
                phase_p90_deg=float(np.quantile(degrees, .9)) if len(degrees) else np.nan)


def load_records():
    samples, covariance = {}, {}
    inventories, operations, runs, states, closure, guard_sensitivity = [], [], [], [], [], []
    old = set(pd.read_csv(VALIDATION / "input_manifest.csv").sha256)
    for batch in CAPTURES:
        folder = DESKTOP / batch
        files, events = identity_audit(folder)
        inventories.extend(files)
        operations.extend(events)
        assert all(r["ok"] for r in files if r["file"].endswith(".csi"))
        for run in (1, 2, 3):
            directory = folder / "capture" / f"run{run}"
            meta = read(directory / "run.json")
            qc = read(directory / "metadata_qc.json")
            source = directory / meta["raw_path"]
            x, host, nic = decoder.decode(source)
            x = x.astype(np.complex128)
            assert len(x) == qc["complete_frames"] == meta["frames"] and np.isfinite(x).all()
            assert np.all(np.diff(host)>0) and np.all(np.diff(nic)>0)
            addresses, orders = frame_details(source)
            assert len(addresses) == len(x) and sum(orders.values()) == len(x)
            target = addresses == APPARENT_TA
            times = (host-meta["start_ns"])/1e9
            duration = (meta["end_ns"]-meta["start_ns"])/1e9
            window = (times>=0) & (times<duration)
            assert sum(window) == meta["window"]["records"]
            raw_events = EVENT.findall((directory / "events.log").read_text(encoding="utf8"))
            errors = sum(list(map(int, line[2].split(","))) != MAP[line[1]] or
                         int(line[4]) != STATES.index(line[1]) for line in raw_events)
            assert not errors and len(raw_events)==meta["events"]["ready_events"]
            aligned = decoder.align(directory / "events.log", meta["start_ns"]/1e9)
            assert aligned is not None
            starts, indices, guard = aligned
            selected, eligible = select_visits(times, starts, indices, duration, guard, target)
            deeper, _ = select_visits(times, starts, indices, duration, .005, target)
            runs.append(dict(batch=batch, run=run, source=str(source), raw_sha256=sha(source),
                             match_in_historical_manifest=sha(source) in old, boot_id=meta["before"]["boot_id"],
                             phy=meta["before"]["phy"].split("/")[-1], ifindex=meta["before"]["ifindex"],
                             before_after_same=meta["before"]==meta["after"], continuity_gate=meta["continuity_gate"],
                             frames=len(x), formal_frames=int(sum(window)), target_window_frames=int(sum(window & target)),
                             formal_pps=sum(window & target)/duration, guard_ms=guard*1000,
                             start_ns=meta["start_ns"], end_ns=meta["end_ns"], max_window_gap_ms=np.max(np.diff(host[window]))/1e6,
                             first_gap_ms=1000*times[window][0], last_gap_ms=1000*(duration-times[window][-1]),
                             ready_count=len(raw_events), mapping_errors=errors, tx_addresses=json.dumps(dict(Counter(addresses))),
                             permutation_hist=json.dumps(dict(orders)), metadata_mismatches=sum(qc["mismatches"].values())))
            for state, visits in selected.items():
                ids = np.concatenate([v for _, v in visits])
                assert np.unique(ids).size == len(ids)
                xs = x[ids]
                cov = spatial(xs)
                key = (batch, run, state)
                samples[key] = xs, times[ids]
                covariance[key] = cov
                deep_ids = np.concatenate([v for _, v in deeper[state]])
                guard_sensitivity.append(dict(batch=batch, run=run, state=state, guard_raw_packets=len(ids),
                                              guard_5ms_packets=len(deep_ids), **compare(cov, spatial(x[deep_ids]))))
                eig = np.linalg.eigvalsh(cov)
                states.append(dict(batch=batch, run=run, state=state, guarded_packets=len(ids),
                                   complete_eligible_visits=eligible[state], visits_with_packets=len(visits),
                                   coverage=len(visits)/eligible[state], first_half_packets=sum(times[ids]<5),
                                   second_half_packets=sum(times[ids]>=5), median_pair_coherence=np.median(abs(relative(cov))),
                                   dominant_eigen_fraction=np.median(eig[:, -1]/np.sum(eig, axis=1))))
            # Each cycle has two adjacent cross states, both observing antenna 8 (chain 2) and 6 (chain 3).
            shared = {}
            for s in ("C1", "C2"):
                shared[s] = {event//6: relative(spatial(x[ids]))[2] for event, ids in selected[s]}
            cycles = sorted(set(shared["C1"]) & set(shared["C2"]))
            q1 = np.array([shared["C1"][i] for i in cycles])
            q2 = np.array([shared["C2"][i] for i in cycles])
            mask = (abs(q1)>=.5) & (abs(q2)>=.5)
            residual = np.degrees(abs(np.angle(q1*q2.conj())))[mask]
            closure.append(dict(batch=batch, run=run, matched_cycles=len(cycles), support=mask.mean(),
                                phase_median_deg=np.median(residual) if len(residual) else np.nan,
                                phase_p90_deg=np.quantile(residual,.9) if len(residual) else np.nan))
    assert len(set(r["raw_sha256"] for r in runs)) == 6
    save("input_integrity.csv", inventories)
    save("receiver_events.csv", operations)
    save("runs.csv", runs)
    save("state_quality.csv", states)
    save("guard_sensitivity.csv", guard_sensitivity)
    save("shared_6_8.csv", closure)
    return samples, covariance


def between_records(covariance):
    rows = []
    records = [(b, r) for b in CAPTURES for r in (1, 2, 3)]
    for (b1, r1), (b2, r2) in itertools.combinations(records, 2):
        comparison = b1 if b1==b2 else "between"
        for state in STATES:
            rows.append(dict(comparison=comparison, first=b1, first_run=r1, second=b2, second_run=r2, state=state,
                             **compare(covariance[b1,r1,state], covariance[b2,r2,state])))
    save("record_comparisons.csv", rows)


def los_response(state):
    y = Y[np.array(MAP[state])-1]
    lengths = np.hypot(3.65, y)
    return np.exp(-2j*np.pi*(lengths-lengths[0])[:, None]*F[None, :]/C)


def fit_offset(q, geometry, allow_delay):
    offsets = F-F.mean()
    support = (np.arange(len(F)) % 3 != 2) & (abs(q)>=.5)
    if support.sum()<5:
        return None
    # Fit phase plus an optional differential slope in radians around the band centre.
    observations = (q * geometry.conj())[support]
    freqs = offsets[support]
    def objective(ns):
        return -abs(np.sum(observations*np.exp(-2j*np.pi*freqs*ns*1e-9)))
    delay = 0.
    if allow_delay:
        grid = np.arange(-400., 400., 1.)
        best = grid[np.argmin([objective(d) for d in grid])]
        delay = minimize_scalar(objective, bounds=(best-1,best+1), method="bounded").x
    phase = np.angle(np.sum(observations*np.exp(-2j*np.pi*freqs*delay*1e-9)))
    return np.exp(1j*(phase+2*np.pi*offsets*delay*1e-9)), float(np.degrees(phase)), float(delay)


def residual(q, prediction, frequency_mask):
    reliable = frequency_mask & (abs(q)>=.5)
    phase = np.degrees(abs(np.angle(q[reliable]*prediction[reliable].conj())))
    support = reliable.sum()/frequency_mask.sum()
    return dict(support=support, median_phase_deg=np.median(phase) if len(phase) else np.nan,
                p90_phase_deg=np.quantile(phase,.9) if len(phase) else np.nan)


def calibration(covariance):
    templates, fits, checks = {}, [], []
    held = np.arange(len(F)) % 3 == 2
    for state in STATES:
        los = los_response(state)
        empirical = [relative(covariance["Kong1", r, state]) for r in (1,2,3)]
        corrections = [np.ones(len(F), complex)]
        for j in (1,2):
            idx = j-1  # (chain 1, chain 2) or (chain 1, chain 3)
            options = {}
            for label, linear in [("constant", False), ("linear", True)]:
                result = fit_offset(empirical[0][idx], los[j], linear)
                if result is None:
                    continue
                offset, phase, slope = result
                check = residual(empirical[1][idx], offset*los[j], held)
                fits.append(dict(state=state, chain=j+1, model=label, intercept_deg=phase,
                                 effective_slope_ns=slope, **check))
                if np.isfinite(check["p90_phase_deg"]):
                    options[label] = (offset, check["p90_phase_deg"])
            if not options:
                raise ValueError(f"Insufficient 0-degree calibration support: {state}/{j+1}")
            if "constant" in options and options["constant"][1] <= 15:
                chosen = "constant"
            else:
                chosen = min(options, key=lambda m: options[m][1])
            offset = options[chosen][0]
            corrections.append(offset)
            for test_mask, label in [(np.ones(len(F),bool),"all"),(held,"unfitted_tones")]:
                check = residual(empirical[2][idx], offset*los[j], test_mask)
                checks.append(dict(state=state, chain=j+1, model=chosen, frequency_set=label,
                                   check_batch="Kong1", check_run=3, **check,
                                   pass_same_direction=check["support"]>=.8 and check["p90_phase_deg"]<=15))
        templates[state] = np.array(corrections)
    save("zero_degree_model_development.csv", fits)
    save("zero_degree_holdout.csv", checks)
    np.savez_compressed(HERE / "empty_only_zero_degree_templates.npz", **templates)
    return templates


def steering(state):
    y = Y[np.array(MAP[state])-1]
    return np.exp(2j*np.pi/C * F[:,None,None]*y[None,:,None]*np.sin(np.deg2rad(ANGLE))[None,None,:])


def angular_spectra(covariance, steering_vectors):
    _, eigvec = np.linalg.eigh(covariance)
    noise = eigvec[:,:,:2].conj().transpose(0,2,1)
    energy = (abs(noise @ steering_vectors)**2).sum(axis=1)/3
    indices = (slice(None), slice(0,15), slice(15,None))
    spectra = np.array([1/np.maximum(energy[idx].mean(axis=0),1e-12) for idx in indices])
    return spectra/spectra.max(axis=1,keepdims=True)


def local_indices(spectrum):
    return np.flatnonzero(spectrum == maximum_filter1d(spectrum, size=3, mode="nearest"))


def peak_summary(spec, quarters):
    output = []
    for i in sorted(local_indices(spec[0]), key=lambda k: -spec[0,k]):
        def matches(curve):
            peaks = local_indices(curve)
            return bool(np.any((abs(ANGLE[peaks]-ANGLE[i])<=9) & (curve[peaks]>=10**(-.6))))
        time_support = sum(matches(p) for p in quarters)
        bands = [matches(p) for p in spec[1:]]
        power = 10*np.log10(spec[0,i])
        output.append(dict(peak_deg=ANGLE[i], peak_db=power, time_blocks=time_support,
                           both_bands=all(bands), boundary=abs(ANGLE[i])>=87,
                           repeat_within_window=power>=-6 and time_support>=3 and all(bands)))
    return output


def raw_diagnostics(samples, covariance, templates):
    all_curves, primary, candidates, joint_peaks = {}, [], [], []
    joint_curves = {}
    for state in STATES:
        vectors = steering(state)
        joint_vectors = joint.steering(Y[np.array(MAP[state])-1])
        for batch in CAPTURES:
            for run in (1,2,3):
                x, times = samples[batch,run,state]
                for mode in (["raw","empty_reference"] if batch=="Kong1" else ["raw"]):
                    evaluated = x if mode=="raw" else x*templates[state].conj()[None,:,:]
                    cov = covariance[batch,run,state] if mode=="raw" else spatial(evaluated)
                    spec = angular_spectra(cov,vectors)
                    quarters = [angular_spectra(spatial(evaluated[(times>=k*2.5)&(times<(k+1)*2.5)]),vectors)[0]
                                for k in range(4)]
                    key = f"{batch}_{run}_{state}_{mode}"
                    all_curves[key] = spec
                    peaks = peak_summary(spec,quarters)
                    for rank,peak in enumerate(peaks,1):
                        candidates.append(dict(batch=batch,run=run,state=state,mode=mode,rank=rank,**peak))
                    primary.append(dict(batch=batch,run=run,state=state,mode=mode,**peaks[0]))
                two_d = joint.music(joint.covariance(x),joint_vectors,q=3)[:, :-1]
                ai,ti=np.unravel_index(np.argmax(two_d),two_d.shape)
                joint_curves[f"{batch}_{run}_{state}"]=two_d
                joint_peaks.append(dict(batch=batch,run=run,state=state,raw_peak_deg=ANGLE[ai],
                                        delay_parameter_ns=joint.TAU[ti],physical_delay_valid=False))
    for batch in CAPTURES:
        for run in (1,2,3):
            avg = np.mean([all_curves[f"{batch}_{run}_{s}_raw"][0] for s in GROUPS],axis=0)
            all_curves[f"{batch}_{run}_fusion_raw"] = avg/avg.max()
            primary.append(dict(batch=batch,run=run,state="G1-G4 noncoherent",mode="raw",peak_deg=ANGLE[np.argmax(avg)]))
    save("blind_spatial_maxima.csv", primary)
    save("blind_local_peaks.csv", candidates)
    save("blind_joint_maxima.csv", joint_peaks)
    np.savez_compressed(HERE/"angular_spectra.npz", angles=ANGLE,**all_curves)
    np.savez_compressed(HERE/"joint_spectra.npz", angles=ANGLE,tau_ns=joint.TAU[:-1],**joint_curves)
    names = ["analysis_rules.json","analyze_captures.py","blind_spatial_maxima.csv","blind_local_peaks.csv",
             "blind_joint_maxima.csv","angular_spectra.npz","joint_spectra.npz","zero_degree_holdout.csv"]
    (HERE/"blind_output_hashes.json").write_text(json.dumps({f:sha(HERE/f) for f in names},indent=2),encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(limits=1):
        samples, covariances = load_records()
        between_records(covariances)
        templates = calibration(covariances)
        raw_diagnostics(samples,covariances,templates)
    print("Completed all six raw inputs and froze peak tables.")
