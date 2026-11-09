# Validation — 2026-11-09

**PASS for the scopes below. No GitHub submission action was performed.**

The earlier local submission was reused and checked again against the actual
teacher archive. Documentation now follows the user's formal contract exactly:
create Xiangdong_Lu/README.md before uploads; a separate fork branch is optional.

## Source and dependencies

- The historical /mnt/data path is absent on this Windows host. All three named
  archive leads were found in prior local Codex outputs, opened and CRC-checked.
- The selected WiCAL_ACDE_For_Teacher.zip contains 11 Python files. Its source,
  raw inputs and result tables were checked against the working tree: **180
  files match byte for byte**. Source hashes are recorded in SOURCE_PROVENANCE
  and SOURCE_ARCHIVE_VERIFICATION; archive hashes in docs/SOURCE_DISCOVERY.json.
- The local native Widar2 file matches the November 1 archived core by SHA-256.
  No verified redistribution license accompanies it; it remains external and
  was not copied into this public folder. Matching hashes are not license proof.
- All 16 public Python files were parsed and syntax-compiled; imports of all
  packaged modules and eight CLI help paths passed. Only NumPy and Matplotlib
  are external dependencies. Detailed inventory: docs/MODULE_INVENTORY.md/json.
- Executed with Windows CPython 3.11.9, NumPy 2.2.6, Matplotlib 3.10.3. These
  packages were already installed. A fresh network dependency installation was
  not tested; no claim of testing every OS/Python version is made.

## Actually executed public checks

`python -B -m unittest discover -s tests -v`: **10 tests passed**.

The suite covers all six AntSEL permutations and signed int8 edge values,
independent bit decoding, invalid AntSEL rejection, exact nominal RF maps,
complete/malformed synthetic covariance, background and per-state gain behavior,
robust fusion fallback, three measured branch metrics, frozen-file tampering and
external native-file hash enforcement. Mapping tests verify software routing,
not independent physical wiring.

The production synthetic example, measured-table evaluator and measured plots
were run successfully. The raw/background synthetic G3 and six-state estimates
are +23 degrees. This is a software fixture, not measured human accuracy.

All cells of all **14 public measured CSV tables** were compared to source:
published numerical values are unchanged; only identifying columns/encoding were
sanitized. Measured evaluator outputs:

| Branch | G3 conditional MAE | Six-state conditional MAE |
|---|---:|---:|
| Raw | 29.7833845695° | 24.9502116585° |
| Run1 subtraction | 19.7002116585° | 7.2833845695° |
| Subtraction + phase hypothesis | 52.3951744570° | 28.6451744570° |

Selected four-point conditional reduction: 63.028901944%. Physical antenna
orientation, AntSEL and human-path identity remain unverified; this is not
established absolute human localization precision. The selected four points,
one occupied run each, old B replacement and C/D retests limit interpretation.

## Fresh private CSI replay

Both the public default profile path and optional external native path were
executed against the actual four private sessions, **12 runs / 126,025 raw
frames**, with **82,247 safe-window frames**. All **144 manifest records match**.
The optional path made **840 actual sage_sfg calls**. Default mode truthfully
records native_crosscheck=false and does not fabricate native-call records.

In both modes, blind_predictions.csv, spectra.csv, state_spectra.csv and
negative_control.csv match the archived experiment **byte for byte**. In native
mode native_sage.csv also matches. Both fresh replays were evaluated successfully.
Hashes and exit codes are in docs/REPLAY_VERIFICATION.json, CURRENT_CHECKS.json
and VALIDATION_COUNTS.json; private logs and captures remain outside the package.

## Distribution and limits

Credential/private-key/email/private-link/personal-path scans and prohibited-file
checks passed; docs/PUBLIC_PACKAGE_AUDIT.json records the scan scope. Only the
intended named author is retained; no student ID is included. A small explicitly
illustrative Chinese submission diagram is provided, not a claimed screenshot.

The ZIP has exactly one top-level Xiangdong_Lu/ directory, including dotfiles and
real source files. CRC, safe paths, exact member-to-folder bytes and manifest
coverage were checked. After extraction the same 10 tests, synthetic example,
measured evaluator and manifest verification passed. MANIFEST_SHA256.txt covers
all distributed files except itself; the adjacent .zip.sha256 covers the ZIP.

Public readers can run examples/tests and reproduce measured metrics. Full real
CSI replay needs authorized private captures; optional original-core checks need
the separately permission-cleared native file. ESP32 device firmware and radio
acquisition software were not in the selected teacher source. Historical
broadside/90-run utilities were retained, imported and help-checked, but their
separate full campaigns were not rerun or certified here. Fresh install and
physical hardware validation remain outside this verification scope.
