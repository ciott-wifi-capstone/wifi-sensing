# Reproduction guide

| Workflow | Required input | What has been checked |
| --- | --- | --- |
| Synthetic static example | Bundled JSON | Real estimator and SQI on generated arrays; no measured accuracy |
| Static selection metrics | Four small bundled CSVs | All eight recording-level comparisons recalculated |
| Static estimator and SQI replay | Private waveform NPZ files | All 144 saved candidates, signals, ACFs, selections and quality factors compared |
| Exploratory motion replay | Private complex-ratio NPZ files and inventory | Eight recordings, event features, run fingerprints and similarities |
| Raw static/motion loader | Authorized `.csi` and metadata required | Code and synthetic mapping checks only; final raw captures absent |
| Historical research | Original research archive, not bundled | Source inspection/syntax checks, not supported execution |

## Saved static checkpoints

Directory entries use `10_bpm_formal__run1_waveforms.npz`, with analogous names
for 10 bpm runs 2/3, 15 bpm runs 1/3, and 20 bpm runs 1/2/3. Each contains 18
observations. For observation `G1`, `Rx0/Rx1`, the prefix is `G1__Rx0_Rx1`:

| Suffix after `__` | Shape | Meaning |
| --- | --- | --- |
| dwell_ratio | (N,30), complex | Same-packet ratio aggregated over state visits |
| state_time | (N,) | Recorded dwell-relative time, seconds |
| processed_fs | (1,) | Effective sample rate, Hz |
| processed_signal | (N,) | Frozen weighted processed ratio waveform |
| acf | (N,) | Frozen combined normalized autocorrelation |
| bpm | (1,) | Frozen candidate rate, breaths/minute |
| selected | (K,) | Zero-based selected subcarriers |
| selected_bnr | (K,) | Selected BNR weights |
| selected_sign | (K,) | Alignment signs, -1 or +1 |
| angles | (30,) | Projection angles, radians |

Existing files also contain `processed_time_uniform`; it describes the
estimator's uniform-index time axis, not independently measured packet arrival.
The replay verifies each required input and computes from `dwell_ratio` again.
It does not feed saved candidate rates into the estimator. It compares them only
after computation. The SQI factors are computed afresh, without belt values.

The static adapter places the ratio in Rx1 and sets Rx2 to one before calling
the estimator. It uses no downsampling and a 2 s Savitzky–Golay window.
The estimator also adds its original epsilon to the denominator.

## Raw capture format and timing

`events.log` READY lines have epoch seconds, state name and `chains=...`.
Each READY event is checked against the state mapping. Expected
logged chains are G1=1,2,3; G2=4,5,6; G3=7,8,9; G4=10,11,12; C1=1,8,6;
C2=10,8,6. This verifies log consistency, not physical wiring.

`metadata_qc.json` requires `host.first` and `host.last` in epoch nanoseconds.
`host.maximum_gap_s`, if available, is reported. The recovered decoder uses
csiread 1.4.1 with expected layout 30 tones x 3 Rx x 1 Tx, transposed to
packets x Tx x Rx x tone. Unexpected shape, nonfinite CSI, missing states,
unordered events or insufficient samples cause explicit failure.

Packet time is evenly assigned between the first and last host timestamps;
the latest preceding READY event supplies the state and visit index. This does
not reconstruct packet loss or irregular packet timing. Packets before the first
event remain unlabelled and are counted. Ratios use the same packet and tone;
real and imaginary medians are computed separately within each visit. Effective
sampling rate is the reciprocal median interval between visits to that state.
Motion additionally crops to `run.json` `start_ns`/`end_ns` before aggregation.

As discussed in thesis Appendix A, full capture-to-reference reproduction
requires the original captures, packet timing and mapping records, and accepted
belt windows and landmarks. These inputs are not included. The saved 15 bpm
run1 reference is 15.133531 bpm; its exact correspondence to the CSI interval
remains unresolved.

## Motion checkpoints

`--archive` expects `run_inventory.csv` with `dataset,motion,run,duration_s`, and
`assets/{motion_lowercase}__{run}_complex_assets.npz`. For observations in the
documented state-major/pair-minor order, `obs_00_time_s` and `obs_00_ratio`
through `obs_17_*` provide run-relative seconds and complex `(N,30)` arrays.
The detector rebuilds magnitude and differential-energy features. It compares
new event times, grid and scores against the saved `candidate_event_time_s`,
`detection_grid_s` and `detection_consensus_score` only after detection.

Detection uses a 0.5 s grid and at least 12 s between candidates. Feature
windows are [-8,-3) s for baseline and [-1,4) s for response, relative to each
event. Thresholds are defined in `src/wical/motion.py`.
Interpolation affects detection bins only and includes long gaps/edge extension;
it is not interpolation of the original packet or observation arrays. Zero-event
inputs retain an explicit empty candidate CSV and warning. No action ground
truth, classification accuracy or formal movement breathing error is implied.

## Motion fingerprint outputs

`run_spatial_fingerprints.csv` has one row per dataset/motion/run/state/pair.
Each of the five response fields is the median over candidate events in that run:
signed amplitude shift in dB, signed robust-z amplitude shift, absolute robust-z
amplitude response, differential-energy change in dB, and normalized complex
displacement (dimensionless). `candidate_event_count` is the number aggregated.

For similarity, the 18 absolute robust-z amplitude responses and the 18
energy changes are separately robust-normalized and concatenated into a
36-component vector. Spearman correlation compares these run vectors.
Condition vectors are the component-wise medians of normalized run vectors.
`run_pair_similarity.csv` contains each unordered run pair once;
`action_similarity.csv` contains each unordered condition pair once.
The legacy field names `same_action_condition`, `left_action` and `right_action`
refer to planned condition labels, not independent observed-motion ground truth.

`run_spatial_fingerprints.png` shows signed amplitude and energy response medians;
`fingerprint_similarity.png` shows the run and condition correlation matrices.
The signed amplitude plot is distinct from the absolute-response feature used
in the correlation vector. `summary.json` reports within-condition correlation
counts, medians and ranges.

A run without events has no fingerprint: empty tables and warnings make this
explicit. Fingerprint plots are omitted if no run has events. Constant vectors
retain undefined correlations as NaN in CSV and masked cells in the plot;
summary statistics become null with a warning. Missing or duplicated
observations, nonfinite features and inconsistent event counts are rejected.

## Reference-reader helper

`src/wical/reference.py` preserves the existing NeuLog time parser and landmark
formula with strict paired input validation. It skips the seven export header
rows; later rows must supply an `HH:MM:SS` timestamp and finite sensor value.
Blank rows and recognized NeuLog export-footer lines are warned about and
excluded from measurements; malformed/nonfinite values, empty inputs and
non-increasing time are rejected. The previous reader could append a timestamp
before failing to parse its measurement, leaving unequal arrays. This is fixed
while preserving the values of valid samples and the detector parameters.

This is a library helper, not an automatic reference-acceptance pipeline. The
static and motion selection entry points do not import it. It does not resolve
clock alignment, crop decisions or manual belt landmark acceptance; the saved
eight-run evaluation uses the already accepted reference table.

## Result-table provenance

The four static CSVs are byte-identical copies of the selected local frozen
numeric exports, with no raw CSI, belt waveform, timestamps or device identities.
`frozen_method_summary.csv` retains the legacy field `n_independent_runs` for
schema continuity; here it counts eight recordings, not independent participants.
Identifiers and failed estimates are retained from the original exports.
Source references and table hashes are in `SOURCES.md`.
