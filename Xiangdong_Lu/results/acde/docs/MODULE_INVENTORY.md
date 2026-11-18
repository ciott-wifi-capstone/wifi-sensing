# Source inventory and execution scope

All 16 shipped Python files were read by AST, syntax-compiled, and their imports
enumerated. All packaged module imports succeeded. Full per-file import,
function and SHA-256 inventories are in MODULE_INVENTORY.json. NumPy and
Matplotlib are the only non-standard-library runtime dependencies.

| Path under src/ | Purpose | Current validation |
|---|---|---|
| retest.py | Current A/C/D/E raw directory entry point; wraps inherited audit and estimator | Full 12-run raw replay, default and external-native modes |
| replay.py | Fixed-layout capture, manifest, AntSEL, READY clock/state audit; same-session subtraction, three branches, G3 and six-state product, held-out Run3 | Production functions exercised by retest; legacy A/B/C ZIP CLI help/import checked |
| wical/common.py | RF map, 28 mm geometry, steering, CSV, hashing, peak selection | Current production use |
| wical/estimate.py | Per-state covariance-reference profile, coarse support, fusion; historical robust/near/equal-budget functions | Current profile use; robust fallback synthetic test; historical full campaign not rerun |
| wical/original/extended_analysis.py | Fixed 385-byte parser, independent scalar decoder, READY reading; historical 17-session diagnostics | Current parser use, six AntSEL permutations tested; historical full CLI help only |
| wical/original/background_phase_qc.py | Vectorized Intel 5300 decode and historical empty-room drift checks | Decoder unit tests and real raw replay; historical full CLI help only |
| wical/original/per_subcarrier_calibration.py | Historical broadside graph phase solver | Imported by production decoder stack; separate broadside CLI help only |
| wical/original/sage_adapter.py | Inherited adapter for an external original SAGE implementation | Optional core SHA-256 enforced; 840 native sage_sfg calls in current ACDE replay |
| run_example.py | Validated synthetic covariance CSV input through production estimator | CLI and numerical test passed |
| evaluate_results.py | Verify frozen prediction bytes, then join provisional geometry | Public tables and both fresh real replays passed; tampering rejected |
| compare.py | Measured spectra and independent-empty plots; inherited geometry helpers | Called by plot_results; two PNG outputs generated |
| plot_results.py | Hash-verified plotting entry point | CLI and actual plots passed |
| verify_manifest.py | Verify distributed file SHA-256 hashes | Folder and extracted ZIP checks passed |

Two package __init__.py files complete the 15 Python files under src/. The
sixteenth file is tests/test_pipeline.py with ten actual unittest cases.

config/acde_policy.json records the frozen analysis policy; it is documentation
of the fixed constants, not an arbitrary runtime override file. The estimator
uses inherited code constants. config/geometry.json is read only by the measured
evaluator after predictions are frozen. Prediction fitting does not import target
coordinates.

The original summarize.py mixed fixed teacher table formatting with geometry;
the public evaluate_results.py provides an explicit frozen-prediction check and
metric reproduction in its place. No original third-party Widar2 source is
vendored. Its local audited file matches the same file in the November 1 archive
(see SOURCE_DISCOVERY.json); that match establishes provenance, not authorship
or redistribution permission.

ESP32 READY log/state/timing analysis is retained. Device firmware and radio
acquisition software were not present as source in the selected teacher archive;
this is an offline analysis/replay package, not a complete firmware deployment.
Historical broadside/90-run/multipath workflows need their separate private
inputs and are not certified by the current selected ACDE tests.
