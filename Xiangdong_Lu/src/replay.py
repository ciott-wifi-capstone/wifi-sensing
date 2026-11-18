"""WiCAL A/B/C replay. No target coordinates are read by this program."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json
import re
import sys
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
from wical.common import ANGLES, FC, MAP, STATES, csvout, norm, peaks, sha
from wical.estimate import PLANE, coarse, inputs, phase, profiles
from wical.original.extended_analysis import read, sha as bytes_sha
from wical.original.sage_adapter import load_core

BRANCHES = ('raw', 'background', 'kong_phase_hypothesis')
PAIRS = ((0, 1), (2, 1), (2, 0))
ROLES = {'run1': 'pre_empty', 'run2': 'static_person', 'run3': 'post_empty'}
ROLE_EVIDENCE = 'user confirmed A, B and C acquisition order; capture logs identify runs only'
READY = re.compile(r'(\d+\.\d+)\s+READY\s+(G\d|C\d)\s+chains=([\d,]+)\s+t=(\d+)(?:us)?\s+seq_idx=(\d+)/6')
POLICY = {
    'reference': 'Run1 within the same session and radio initialization only',
    'role_evidence': ROLE_EVIDENCE, 'target': 'Run2', 'held_out_empty': 'Run3',
    'points': 'A replay plus B/C; D deferred by user',
    'window': 'recorded 10 s, 3 through 21 ms after READY, complete dwells only; offset is median host_ns - ESP32_us*1000',
    'covariance': 'unit mean power per packet over all 30 tones and 3 Rx, then average outer product separately per tone',
    'reference_column': 1, 'grid': [-90, 90, 1], 'frequency_hz': FC,
    'spacing_m': 0.028, 'physical_state_map_from_READY': MAP,
    'steering': 'exp(+j*k*y*sin(theta)); angle positive toward higher nominal antenna numbers',
    'mapping': 'cov = retained Intel mask3; alt = public-stable double-rank sensitivity; physical wiring unconfirmed',
    'branches': BRANCHES, 'phase': 'Run1 principal eigenvector divided by nominal TX spherical steering, 3.65 m, diagnostic',
    'G3': 'one-path AoA-only SAGE per tone; native argmax cross-check; average profile power; strongest angle',
    'J6_product': 'geometric mean of six noncoherent mean-normalized powers floored at 0.05; inherited coarse support from G1-G4 and their average; strongest angle',
    'tie_rule': 'first index of ascending angle grid', 'GT_used_in_estimation': False,
    'drift': 'robust phase center per dwell, equal votes across dwells, wrapped Run3 minus Run1, P95 over 3 pairs x 30 tones',
    'phase_gate_p95_deg': 10, 'G3_control_score': 'max G3 unnormalized power',
    'J6_control_score': 'max of mean unnormalized six-state power; also report mean power at selected product angle',
}


class CaptureFiles:
    def __init__(self, a_directory, zip_file):
        self.a = a_directory
        self.z = zip_file
        self.names = [f'raw/New A/{p.relative_to(self.a).as_posix()}' for p in self.a.rglob('*') if p.is_file()]
        self.names += [f'raw/{n}' for n in self.z.namelist() if not n.endswith('/')]
        assert len(self.names) == len(set(self.names)), 'Duplicate input path'

    def namelist(self):
        return self.names

    def read(self, name):
        if name.startswith('raw/New A/'):
            return (self.a / name.removeprefix('raw/New A/')).read_bytes()
        return self.z.read(name.removeprefix('raw/'))

    def load(self, name):
        return json.loads(self.read(name))


def optional_core(path):
    """The production AoA profile needs NumPy only; native SAGE is optional."""
    if path is None:
        return None, {}, None
    expected = '1d39f976b819f6b07b77d5f72772707c3a001c4ffb0bddde10d4594d003d5b4c'
    if sha(path) != expected:
        raise ValueError('External Widar2 file differs from the audited revision; see NOTICE.md')
    core, calls, _ = load_core(path)
    con = core['sage_const_build'](1,1,3,1,3,0,0,.028*FC/299792458.,np.array([[0.]]),
                                  np.radians(ANGLES)[None],np.array([[0.]]),1)
    con.update(tof_candidates=np.ones((1,1),complex),aoa_candidates=PLANE[2],doppler_candidates=np.ones((1,1),complex))
    return core, calls, con


def circular_summary(samples):
    """Identical to the earlier A replay. One circular vote per nonzero observation."""
    unit = np.divide(samples, abs(samples), out=np.zeros_like(samples), where=abs(samples) > 0)
    valid = abs(unit) > 0
    count = valid.sum(axis=0)
    mean = unit.sum(axis=0) / np.maximum(count, 1)
    center = np.angle(mean)
    residual = np.where(valid, np.angle(unit * np.exp(-1j * center)), np.nan)
    median = np.full(center.shape, np.nan)
    observed = count > 0
    median[observed] = np.nanmedian(residual[:, observed], axis=0)
    return np.exp(1j * (center + median)), abs(mean), count


def utc_ns(ns):
    seconds, remainder = divmod(int(ns), 10**9)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{remainder:09d}Z'


def identity(radio):
    return tuple(radio[k] for k in ('boot_id', 'phy', 'ifindex', 'radio'))


def audit_session(source, sess, tables):
    prefix = sess + '/'
    manifest = source.load(prefix + 'manifest.json')['files']
    expected = {prefix + row['path'] for row in manifest}
    present = {name for name in source.names if name.startswith(prefix)}
    assert present == expected | {prefix + 'manifest.json'}, (sess, expected ^ present)
    for entry in manifest:
        name = prefix + entry['path']
        payload = source.read(name)
        digest = bytes_sha(payload)
        passed = len(payload) == entry['bytes'] and digest == entry['sha256']
        tables['manifest'].append(dict(session=sess.split('/')[-1], path=entry['path'], bytes=len(payload),
                                       sha256=digest, expected_sha256=entry['sha256'], match=passed))
        assert passed, name
    config = source.load(prefix + 'config.json')
    plan = source.load(prefix + 'capture/plan.json')
    summary = source.load(prefix + 'capture/summary.json')
    acquisition = source.load(prefix + 'acquisition_summary.json')
    kernel = source.load(prefix + 'kernel_assessment.json')
    progress = source.load(prefix + 'capture/progress.json')
    assert config['center_mhz'] * 1e6 == FC and config['width'] == 'HT40-' and config['channel'] == 48
    assert config['control_mhz'] == 5240 and config['runs'] == 3 and config['duration'] == 10
    assert tuple(plan['sequence']) == tuple(config['states']) == STATES
    assert plan['schedule'] == [[r, config['duration']] for r in ROLES]
    assert progress == {'phase': 'finished', 'completed_runs': 3}
    assert summary['finished'] and summary['completed_schedule'] and not summary['campaign_failure']
    assert acquisition['status'] == 'PASS' and kernel['status'] == 'PASS'
    assert kernel['complete'] and not kernel['has_loss']
    ops = source.read(prefix + 'operations.log').decode()
    logs = source.read(prefix + 'rx_kernel.log').decode(errors='replace')
    get_phy = lambda name: re.search(r'phy#(\d+)', source.load(prefix + name)['radio'])[1]
    tables['sessions'].append(dict(
        session=sess.split('/')[-1], point=sess[-1], label=config['label'], planned_runs=config['runs'],
        raw_runs=3, status='analyzed', channel=config['channel'], width=config['width'],
        control_mhz=config['control_mhz'], center_mhz=config['center_mhz'], nominal_pps=config['pps'],
        plan_gap_s=plan['pause_s'], dwell_settling_us=plan['settling_us'], dwell_valid_us=plan['valid_us'],
        boot_id=plan['reference']['boot_id'], phy_before_session='phy'+get_phy('runtime_before.json'),
        phy_capture=Path(plan['reference']['phy']).name, phy_after_session='phy'+get_phy('runtime_after.json'),
        ifindex=plan['reference']['ifindex'], driver_reload_at_setup='["rmmod", "iwlwifi"]' in ops,
        kernel_status=kernel['status'], kernel_records=kernel['record_count'],
        cfg80211_warning_lines=len(re.findall(r'WARNING:.*nl80211', logs)),
        firmware_successful_loads=len(re.findall(r';iwlwifi .*: loaded firmware version', logs)),
        stream_starts=logs.count('starts streaming'), stream_stops=logs.count('stops streaming'),
        acquisition_status=acquisition['status'], manifest_entries=len(manifest), roles=ROLE_EVIDENCE,
    ))
    return config, plan, {row['label']: row for row in summary['runs']}


def audit_run(source, sess, run, config, plan, summary_row, tables):
    point = sess[-1]
    prefix = f'{sess}/capture/{run}/'
    d = read(source, sess, run)
    cfg, q, t, starts, states = (d[k] for k in ('cfg', 'q', 't', 'starts', 'states'))
    raw = np.frombuffer(source.read(q['csi_file']), np.uint8).reshape(-1, 385)
    meta = source.load(prefix + 'metadata_qc.json')
    assert cfg == summary_row and cfg['raw_path'] == q['csi_file'].split('/')[-1]
    assert identity(cfg['before']) == identity(cfg['after']) == identity(plan['reference'])
    assert cfg['continuity_gate'] and cfg['identity_unchanged'] and not cfg['forced_stop'] and not cfg['failure']
    assert cfg['radio_initialization'] == 'none; logger restarts only'
    assert not cfg['esp32_forced_stop'] and cfg['esp32_exit'] == 0 and not cfg['logger_forced_stop']
    t0, t1 = (cfg['window'][k] for k in ('start_ns', 'end_ns'))
    assert t1 - t0 == config['duration'] * 10**9 and cfg['window'] == meta['window']
    in_window = (t >= t0) & (t < t1)
    ant = raw[:, 161]
    assert Counter(map(int, ant)) == {int(k): v for k, v in meta['csi']['antenna_selections'].items()}
    assert len(t) == cfg['frames'] == meta['complete_frames'] == meta['timed_csi_segment_frames']
    assert meta['complete_bytes'] == raw.size and in_window.sum() == cfg['window']['records']
    assert np.all(raw[:, 45:51] == np.frombuffer(b'\x6e\x14\x78\x14\x28\x00', np.uint8))
    nic = raw[:, 29:37].copy().view('<u8').ravel().astype(np.int64)
    assert np.all(np.diff(nic) > 0) and tuple(nic[[0, -1]]) == (meta['nic']['first'], meta['nic']['last'])
    assert tuple(t[[0, -1]]) == (meta['host']['first'], meta['host']['last'])
    event_text = source.read(prefix + 'events.log').decode()
    assert int(Decimal(event_text.split()[0]) * 10**9) == cfg['events']['run_issued_ns'] == t0
    events = [READY.fullmatch(line) for line in event_text.splitlines() if ' READY ' in line]
    assert all(events) and len(events) == len(starts) == cfg['events']['ready_events']
    assert Counter(states) == cfg['events']['ready_counts']
    host = np.array([int(Decimal(m[1]) * 10**9) for m in events], np.int64)
    dev = np.array([int(m[4]) for m in events], np.int64)
    assert (host[0], dev[0], states[0]) == tuple(cfg['events']['first'][k] for k in ('host_ns', 'device_us', 'state'))
    assert (host[-1], dev[-1], states[-1]) == tuple(cfg['events']['last'][k] for k in ('host_ns', 'device_us', 'state'))
    assert np.all(np.diff(host) > 0) and np.all(np.diff(dev) > 0)
    assert np.diff(dev).min() == cfg['events']['device_period_min_us']
    assert np.diff(dev).max() == cfg['events']['device_period_max_us']
    serial = source.read(prefix + 'serial_transcript.log').decode()
    transcript = [line.replace('us seq_idx', ' seq_idx').replace('\tRX\t', '  ')
                  for line in serial.splitlines() if '\tRX\tREADY ' in line]
    assert [m.groups() for m in events] == [READY.fullmatch(line).groups() for line in transcript]
    for i, m in enumerate(events):
        assert m[2] == STATES[i % 6] and int(m[5]) == i % 6 and states[i] == m[2]
        tables['switch_events'].append(dict(point=point, run=run, event_index=i, state=m[2],
                                           chains=m[3], host_ns=host[i], device_us=dev[i], aligned_ns=starts[i],
                                           residual_ms=(host[i]-starts[i])/1e6))
    assert q['invalid_permutation'] == 0
    assert np.isfinite(d['h']).all() and np.all(np.sum(abs(d['h'])**2, axis=1) > 0)
    dwell = np.searchsorted(starts, t, side='right') - 1
    safe = np.clip(dwell, 0, len(starts)-1)
    age = (t-starts[safe])/1e6
    valid = (dwell >= 0) & (dwell < len(starts)-1) & (age >= 3) & (age <= 21) & in_window & d['ok']
    state = states[safe]
    tables['runs'].append(dict(
        point=point, session=sess.split('/')[-1], run=run, scene_role=ROLES[run],
        role_evidence=ROLE_EVIDENCE, status='analyzed', raw_path=q['csi_file'].removeprefix('raw/'),
        sha256=q['csi_sha256'], raw_frames=len(raw), raw_bytes=raw.size, frames_10s=in_window.sum(),
        selected_frames=valid.sum(), window_start_ns=t0, window_end_ns=t1, start_utc=utc_ns(t0), end_utc=utc_ns(t1),
        first_frame_ns=t[0], last_frame_ns=t[-1], first_nic_us=nic[0], last_nic_us=nic[-1],
        host_max_gap_ms=np.diff(t).max()/1e6, nic_max_gap_ms=np.diff(nic).max()/1000,
        observed_window_pps=in_window.sum()/config['duration'],
        boot_before=cfg['before']['boot_id'], boot_after=cfg['after']['boot_id'],
        phy_before=Path(cfg['before']['phy']).name, phy_after=Path(cfg['after']['phy']).name,
        ifindex_before=cfg['before']['ifindex'], ifindex_after=cfg['after']['ifindex'],
        continuity=cfg['continuity_gate'], forced_stop=cfg['forced_stop'], radio_initialization=cfg['radio_initialization'],
        ready_events=len(events), invalid_antsel=q['invalid_permutation'],
        ready_period_min_us=np.diff(dev).min(), ready_period_max_us=np.diff(dev).max(),
        switch_resid_std_ms=q['host_event_residual_std_ms'], switch_resid_span_ms=q['host_event_residual_span_ms'],
        switch_resid_slope_ms_per_s=q['host_event_slope_ms_per_s'], channel=config['channel'], width=config['width'],
    ))
    cov, alt_cov, centers, concentration = [], [], [], []
    for group in STATES:
        keep = valid & (state == group)
        h = d['h'][keep].astype(complex)
        tags = ant[keep]
        old = np.stack([tags & 3, (tags >> 2) & 3, (tags >> 4) & 3], axis=1)
        variant = old.copy()
        variant[:, 0] = tags & 1
        public = np.argsort(np.argsort(variant, axis=1, kind='stable'), axis=1, kind='stable')
        alt = np.empty_like(h)
        for slot in range(3):
            alt[np.arange(len(h)), :, public[:, slot]] = h[np.arange(len(h)), :, old[:, slot]]
        for values, output in ((h, cov), (alt, alt_cov)):
            power = np.mean(abs(values)**2, axis=(1, 2))
            values = values/np.sqrt(power[:, None, None])
            output.append(np.einsum('nka,nkb->kab', values, values.conj())/len(values))
        products = np.stack([h[:, :, a]*h[:, :, b].conj() for a, b in PAIRS], axis=1)
        _, packet_r, n = circular_summary(products)
        dwell_ids, dwell_counts = np.unique(dwell[keep], return_counts=True)
        dwell_statistics = [circular_summary(products[dwell[keep] == i]) for i in dwell_ids]
        per_dwell = [row[0] for row in dwell_statistics]
        center, dwell_r, _ = circular_summary(np.asarray(per_dwell))
        centers.append(center)
        concentration.append(dwell_r)
        tables['state_quality'].append(dict(
            point=point, run=run, state=group, nominal_chains=','.join(map(str, MAP[group])),
            ready_events=(states == group).sum(), dwells_used=len(dwell_ids), selected_packets=len(h),
            min_packets_per_dwell=dwell_counts.min(), median_packets_per_dwell=np.median(dwell_counts),
            max_packets_per_dwell=dwell_counts.max(), median_packet_R=np.median(packet_r),
            median_dwell_R=np.median(dwell_r), p05_dwell_R=np.percentile(dwell_r, 5),
            mapping_changed_packets=np.any(old != public, axis=1).sum(),
            zero_pair_tone_observations=(abs(products) == 0).sum(),
            missing_dwell_pair_tones=sum((row[2] == 0).sum() for row in dwell_statistics),
            **{f'antsel_{v}': (tags == v).sum() for v in (6, 9, 18, 24, 33, 36)},
        ))
        for j, (a, b) in enumerate(PAIRS):
            for k in range(30):
                tables['phase_tones'].append(dict(point=point, run=run, state=group, tone=k,
                                                  ant_a=MAP[group][a], ant_b=MAP[group][b],
                                                  phase_deg=np.degrees(np.angle(center[j, k])),
                                                  packet_R=packet_r[j, k], dwell_R=dwell_r[j, k], packets=n[j, k]))
    return dict(cov=np.asarray(cov), alt=np.asarray(alt_cov), phase=np.asarray(centers), concentration=np.asarray(concentration))


def compare_empty(point, before, after, tables):
    for si, state in enumerate(STATES):
        delta = np.degrees(np.angle(after['phase'][si]*before['phase'][si].conj()))
        p95 = np.percentile(abs(delta), 95)
        tables['empty_phase_drift'].append(dict(
            point=point, state=state, pair_tones=delta.size, mean_abs_deg=np.mean(abs(delta)),
            median_abs_deg=np.median(abs(delta)), p95_abs_deg=p95, rms_deg=np.sqrt(np.mean(delta**2)),
            pair_tones_both_dwell_R_ge_0p8=((before['concentration'][si] >= .8) & (after['concentration'][si] >= .8)).sum(),
            p95_le_10deg=p95 <= 10,
        ))
        for j, (a, b) in enumerate(PAIRS):
            for k in range(30):
                tables['empty_phase_drift_tones'].append(dict(point=point, state=state, tone=k,
                    ant_a=MAP[state][a], ant_b=MAP[state][b], delta_deg=delta[j, k],
                    run1_dwell_R=before['concentration'][si, j, k], run3_dwell_R=after['concentration'][si, j, k]))


def estimate_point(point, datasets, core, con, tables):
    estimates = {}
    for mapping in ('cov', 'alt'):
        bg = datasets['run1'][mapping]
        for branch in BRANCHES:
            q = phase(bg, 'kong_phase' if branch == 'kong_phase_hypothesis' else 'identity')
            if branch == 'kong_phase_hypothesis':
                for si, state in enumerate(STATES):
                    for k in range(30):
                        for rx, ant in enumerate(MAP[state]):
                            tables['effective_phase'].append(dict(point=point, mapping=mapping, state=state, tone=k,
                                                                  nominal_antenna=ant, phase_deg=np.degrees(np.angle(q[si,k,rx]))))
            for run, data in datasets.items():
                if run == 'run1' and branch != 'raw':
                    continue
                x = inputs(data[mapping], q, np.zeros_like(bg) if branch == 'raw' else bg)
                p = profiles(x)
                support, _ = coarse(p)
                unrestricted = np.exp(np.log(np.maximum(norm(p), .05)).mean(0))
                j6 = np.where(support, unrestricted, 0)
                ident = dict(point=point, run=run, role=ROLES[run], mapping=mapping, branch=branch)
                for method, power, score_power in (('G3', p[2], p[2]), ('J6_product', j6, p.mean(0))):
                    selected = np.argmax(power)
                    candidates = peaks(power, limit=len(ANGLES))
                    second = next((k for k in candidates if k != selected), None)
                    result = dict(**ident, method=method, angle_deg=ANGLES[selected], spectral_peak=power[selected],
                                  score=score_power.max(), score_at_selected_angle=score_power[selected],
                                  runner_up_angle_deg=ANGLES[second] if second is not None else None,
                                  runner_up_relative=power[second]/power[selected] if second is not None else None)
                    estimates[(mapping, branch, method, run)] = result
                    tables['blind_predictions'].append(result)
                    for k, a in enumerate(ANGLES):
                        tables['spectra'].append(dict(**ident, method=method, angle_deg=a, power=power[k],
                            candidate=(k in candidates), selected=(k == selected),
                            coarse_support=bool(support[k]) if method == 'J6_product' else True,
                            pre_mask_power=unrestricted[k] if method == 'J6_product' else power[k],
                            unnormalized_score_profile=score_power[k]))
                for si, state in enumerate(STATES):
                    for k, angle in enumerate(ANGLES):
                        tables['state_spectra'].append(dict(**ident, state=state, angle_deg=angle, power=p[si,k]))
                if mapping == 'cov' and core is not None:
                    for k, v in enumerate(x[2]):
                        param, res = core['sage_sfg'](v[None,None,:], np.zeros((4,1),complex), np.array([[0],[90],[0]]),
                                                    con, np.array([0.,1e-12,0.,1e-12]), 4, 4)
                        angle = np.degrees(param[1,0].real)
                        direct = abs(v@PLANE[2].conj())**2/9
                        assert np.isclose(direct[int(round(angle+90))], direct.max(), rtol=1e-8, atol=1e-12)
                        tables['native_sage'].append(dict(**ident, tone=k, angle_deg=angle, residual=res))
    for (mapping, branch, method, run), row in estimates.items():
        if run != 'run2':
            continue
        control = estimates[(mapping, branch, method, 'run3')]
        tables['negative_control'].append(dict(point=point, mapping=mapping, branch=branch, method=method,
            target_angle_deg=row['angle_deg'], empty_angle_deg=control['angle_deg'],
            target_score=row['score'], empty_score=control['score'], target_empty_ratio=row['score']/control['score'],
            target_score_at_selected=row['score_at_selected_angle'], empty_score_at_selected=control['score_at_selected_angle'],
            target_empty_selected_angle_power_ratio=row['score_at_selected_angle']/control['score_at_selected_angle']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sage-core', type=Path, help='Optional permission-cleared external Widar2 Python file')
    ap.add_argument('--a', required=True, type=Path, help='Original New A session directory')
    ap.add_argument('--bcd', required=True, type=Path, help='Original New BCD.zip')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use an empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'policy_before_fit.json').write_text(json.dumps(POLICY, indent=2), encoding='utf-8')
    tables = {name: [] for name in ('manifest','sessions','runs','state_quality','phase_tones','switch_events',
        'empty_phase_drift','empty_phase_drift_tones','blind_predictions','spectra','state_spectra','negative_control',
        'native_sage','effective_phase','zip_inventory','d_directory_gap')}
    sage_file = args.sage_core
    core, calls, con = optional_core(sage_file)
    with ZipFile(args.bcd) as z:
        assert z.testzip() is None, 'Archive member CRC failure'
        for f in z.infolist():
            tables['zip_inventory'].append(dict(path=f.filename, is_dir=f.is_dir(), bytes=f.file_size,
                                                compressed_bytes=f.compress_size, crc32=f'{f.CRC:08x}'))
        source = CaptureFiles(args.a, z)
        for point in 'ABC':
            sess = f'raw/New {point}'
            config, plan, summary = audit_session(source, sess, tables)
            datasets = {run: audit_run(source, sess, run, config, plan, summary[run], tables) for run in ROLES}
            compare_empty(point, datasets['run1'], datasets['run3'], tables)
            estimate_point(point, datasets, core, con, tables)
            print(f'{point} raw frames, timing, READY, spectra and spectral profiles verified (see provenance for optional native audit)', flush=True)
        # D is inventory only. Names are compared to the shared B/C layout, never used to fill D.
        def signature(point):
            prefix = f'raw/New {point}/'
            return {re.sub(r'rx_\d+_\d+_\d+\.csi$', '<raw>.csi', name[len(prefix):])
                    for name in source.names if name.startswith(prefix)}
        found = signature('D')
        common_paths = signature('B') & signature('C')
        for entry in sorted(common_paths | found):
            tables['d_directory_gap'].append(dict(path_pattern=entry, expected_from_B_C=entry in common_paths,
                                                   present_in_D_zip=entry in found, status='inventory_only'))
    for name, rows in tables.items():
        csvout(args.out/f'{name}.csv', rows)
    frozen = ('policy_before_fit.json', 'blind_predictions.csv', 'spectra.csv', 'negative_control.csv') + (('native_sage.csv',) if core is not None else ())
    provenance = dict(input_zip_sha256=sha(args.bcd), zip_crc_ok=True, native_core_calls=calls,
                      native_core_sha256=sha(sage_file) if sage_file else None, native_crosscheck=bool(core), replay_script_sha256=sha(__file__), numpy=np.__version__,
                      frozen_before_geometry={name: sha(args.out/name) for name in frozen})
    (args.out/'fit_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
