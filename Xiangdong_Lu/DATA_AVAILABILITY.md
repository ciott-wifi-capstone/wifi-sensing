# Data availability

The public submission includes numerical results and small, explicitly synthetic
fixtures. It contains no real raw CSI, private capture logs, private links,
student ID, email, credential, thesis, or large historical archive.

Real data can be requested from the student/supervisor subject to applicable
permissions. No public download URL or unrestricted dataset license is asserted.
Do not place private data in a public fork or PR. If the supervisor requires full
data, transfer a separate private package using their approved channel.

## Private replay input

Use an authorized local copy of the ACDE teacher package. `--data` must point to
its `raw` directory, containing these session folders unchanged:

```text
raw/
  New A/
  Real New C/
  Real New D/
  New E/
```

Each session needs the original manifest and all files enumerated in it,
including `config.json`, `runtime_before.json`, `runtime_after.json`,
`acquisition_summary.json`, `kernel_assessment.json`, `operations.log`,
`rx_kernel.log`, plus `capture/plan.json`, `capture/summary.json`,
`capture/progress.json`, and for run1, run2, run3: the 385-byte-record `.csi`
file, `run.json`, `metadata_qc.json`, `events.log`, `serial_transcript.log`
and remaining manifest files. Do not rename or sanitize private inputs before
hash verification. Exactly one CSI file per run is expected.

Acquisition configuration: Intel 5300, 1Tx × 3Rx, 30 CSI tone groups,
HT40-, control 5240 MHz, centre 5230 MHz, channel 48, 10 s per run.
The parser validates the fixed layout, monotonic timestamps and metadata.
READY logs identify G1/G2/G3/G4/C1/C2 and antenna routing. Safe samples are
3–21 ms after aligned READY, within the recorded run window, excluding the
last incomplete dwell. Host/device clock offset uses the median logged offset;
unknown serial latency remains a limitation.

Run1: pre-empty background; Run2: static person; Run3: held-out post-empty.
The archived role evidence confirms A and assumes the same order for C/D/E;
the current submission brief supplies the same order for all four. Logs alone
label run numbers, not scene contents. E is reported as B_new, not the discarded
old B. Original receiver timestamps are 2026-10-18, not the packaging date.

Private replay produces path/host and radio identifiers from private metadata;
keep those outputs in ignored `local_results/` and review them before sharing.

## Public fixture formats

`examples/synthetic_covariance.csv`: rows `state,tone,row,col,real,imag`,
covering six states × 30 tones × 3 × 3 complex covariance entries. Generated
from an analytic single plane wave with positive diagonal noise; not measured
CSI, no raw decoder/hardware validation, no human performance evidence.

`results/acde/`: measured numerical outputs of the archived experiment, with
session identifiers removed where present. E retains its archival label;
the geometry config maps E to B_new. See `docs/RESULTS_SCHEMA.md`.
Hashes and provenance allow the supervisor to relate these to private inputs.
