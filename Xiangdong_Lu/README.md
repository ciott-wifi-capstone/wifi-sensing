# WiCAL static human AoA — Xiangdong Lu

USYD ELEC5021 · **sensing** · 12 physical antennas, 3 Rx chains, six switched
states. Compare the G3 three-antenna AoA-only SAGE baseline with noncoherent
six-state angle-spectrum fusion. This project studies static sensing, not
continuous tracking or simultaneous 12-channel coherent estimation.

## Objective

Investigate whether six switched, noncoherent angular spectra help the G3
three-chain baseline in selected static-person WiFi sensing captures. Preserve
the original analysis and its failed phase-reference ablation for review.

## Hardware and data

Intel 5300/PicoScenes fixed-layout CSI, 1Tx × 3Rx, 30 tone groups, HT40-,
5230 MHz centre. Nominal element spacing 0.028 m. Switch states:
G1=[1,2,3], G2=[4,5,6], G3=[7,8,9], G4=[10,11,12],
C1=[1,8,6], C2=[10,8,6]. TX=(0,0) m, RX=(3.65,0) m.
Each point has Run1 empty → Run2 static person → Run3 empty.
E in archived files is reported as B_new, replacing old B near the wall.

Raw CSI is excluded from this public folder. See DATA_AVAILABILITY.md for exact
private input structure and examples/README.md for synthetic CSV input.

## Requirements and installation

Tested on Python 3.11.9, NumPy 2.2.6, Matplotlib 3.10.3. Python 3.11 is the
recommended version; hardware is not needed for tests or public examples.
Install Python 3.11 and ensure `python --version` selects it. On Windows,
`py -3.11` is an alternative if the Python launcher is configured.
Open a terminal in this Xiangdong_Lu folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.venv\Scripts\python.exe -B src\run_example.py --input examples\synthetic_covariance.csv --out local_results\synthetic
.venv\Scripts\python.exe -B src\evaluate_results.py --tables results\acde --out local_results\measured
.venv\Scripts\python.exe -B src\verify_manifest.py
```

On macOS/Linux replace the first two lines by `python3.11 -m venv .venv` and
`.venv/bin/python -m pip install -r requirements.txt`, use `.venv/bin/python`
and forward slashes for subsequent commands. No API key or service is required.

The synthetic example writes `estimates.csv` and `spectra.csv`; it is an
analytic software check, **not measured performance**. The measured evaluator
reads the existing experiment tables and writes per-point comparisons and
summary.json; it does not recreate raw CSI from CSV results.

## Reproduce the real capture analysis (authorized private inputs)

```text
python -B src/retest.py --data /path/to/authorized/raw --out local_results/replay
python -B src/evaluate_results.py --tables local_results/replay --out local_results/recomputed
python -B src/plot_results.py --tables local_results/replay --out local_results/plots
```

The first command validates manifests, decodes raw frames and AntSEL, checks
READY RF mapping and timing, computes all three processing branches for both
software mapping conventions, preserves Run3 controls, and freezes prediction
hashes. The second verifies those hashes and only then joins conditional
geometry. Do not run with Python optimization (`-O`), which disables assertions.
Use a fresh output directory for each run; never overwrite archived results.

The default pipeline is fully executable with NumPy/Matplotlib plus the private
capture inputs. The original native Widar2 Python file is intentionally absent
because its redistribution license was not verified. Default results use the
same direct one-path AoA profile already present in the source. For an optional
native cross-check add `--sage-core /path/to/permission-cleared/widar2_original.py`
to the first command. It checks the audited SHA-256 before loading only five
functions and records actual native call counts. See NOTICE.md and VALIDATION.md.

## Input/output formats

- Public input: `examples/synthetic_covariance.csv` has columns
  `state,tone,row,col,real,imag`; six states, tone index 0–29, row/col 0–2,
  1,620 entries representing Hermitian 3×3 complex covariance matrices.
- Example output: `estimates.csv` (branch, method, angle and synthetic flag)
  and `spectra.csv` (angle grid and powers), with both labelled synthetic.
- Private input: the unchanged ACDE `raw/` directory with four session trees,
  385-byte CSI records, manifests, READY events and metadata. Exact layout and
  timing fields are in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
- Replay output: per-run/phase/quality audits, `blind_predictions.csv`,
  `spectra.csv`, `state_spectra.csv`, Run3 `negative_control.csv` and frozen
  provenance hashes. These local outputs may contain private identifiers;
  keep them in ignored `local_results/`.
- Measured evaluator output: `conditional_results.csv` and `summary.json`
  with branch/method MAEs and an explicit unverified-accuracy flag.
  Numeric table definitions are in [docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).

The retained historical utilities in `src/wical/original/` have separate CLI
inputs; [docs/MODULE_INVENTORY.md](docs/MODULE_INVENTORY.md) distinguishes them
from the current ACDE entry point. ESP32 READY timing analysis is included;
device acquisition firmware was not present in the selected teacher source.

## Measured results and known limitations

Retained Intel mask3 mapping, provisional orientation/sign; angles in degrees:

| Point | Position m | Conditional reference | G3, Run1 subtraction | Six-state, Run1 subtraction |
|---|---|---:|---:|---:|
| A | (0.8, 0.6) | +11.89 | +14 | +31 |
| B_new (archival E) | (1.64, -2.0) | -44.86 | -21 | -38 |
| C | (0.8, -1.2) | -22.83 | +15 | -24 |
| D | (2.4, -1.6) | -52.00 | -67 | -54 |

| Fixed processing branch | G3 conditional MAE | Six-state conditional MAE |
|---|---:|---:|
| Raw CSI | 29.78° | 24.95° |
| Run1 background subtraction | 19.70° | 7.28° |
| Subtraction + unverified dominant-phase diagnostic | 52.40° | 28.65° |

Background subtraction gives a **63.0% reduction in the selected four-point
conditional MAE**. It is not independently verified human-specific path accuracy.
Physical wiring, AntSEL mapping and angle sign remain unconfirmed; alternate
mapping and sign tables are retained. A is worse with six states. There is only
one static-person run per point, C/D were retested, and old B was replaced after
inspection. This subset cannot establish general accuracy, a human detection
rate, confidence intervals, person-path isolation or tracking performance.
Empty-room drift and multipath remain material limitations. Ground-truth
coordinates are used for evaluation, not angle-peak selection.

## Folder structure

```text
Xiangdong_Lu/
  README.md, requirements.txt, LICENSE, NOTICE.md
  src/                    raw replay, estimators, parser, audit and evaluation
  config/                 fixed acquisition/algorithm policy and geometry
  examples/               clearly labelled small synthetic covariance fixture
  tests/                  numerical, parser, mapping and fixture tests
  results/acde/           real archived numeric results, sanitized identifiers
  docs/                   algorithm, source provenance, schemas, repository rules
  DATA_AVAILABILITY.md, FILES_EXCLUDED.md
  VALIDATION.md, CHECKLIST.md, SUBMISSION_GUIDE_CN.md
  MANIFEST_SHA256.txt
```

Only upload this personal folder to the **wifi-sensing** repository via your own
fork and PR. See SUBMISSION_GUIDE_CN.md. Source attribution/licensing status is
in LICENSE and NOTICE.md; no commercial relicense of Widar2 is asserted.
