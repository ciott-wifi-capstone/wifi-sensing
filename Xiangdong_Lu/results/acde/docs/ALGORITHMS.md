# Preserved algorithm and audit paths

1. `src/wical/original/background_phase_qc.py::decode`: vectorized Intel
   int8 complex decoding and 2-bit AntSEL mapping. Invalid permutations now
   return a false validity flag and zero samples instead of indexing outside
   the receiver array. `extended_analysis.py::scalar_decode` is an independent
   bit-by-bit check for valid frames.
2. `src/replay.py::audit_session/audit_run`: full manifest checks, exact frame
   metadata checks, run/clock/radio continuity, READY RF mapping, safe-window
   selection, `cov` and alternative `alt` decoding conventions, packet-normalized
   per-tone covariance and phase diagnostics.
3. `src/wical/estimate.py::inputs/profiles/phase/coarse`: covariance subtraction,
   diagnostic phase, AoA-only one-path profile and fixed coarse search.
4. `src/replay.py::estimate_point`: all raw/background/phase branches, G3 and
   noncoherent six-state product fusion, Run3 negative control and optional
   external native SAGE cross-check. Public default records that native audit
   was not run; it never labels a direct profile as a new native execution.
5. `src/wical/estimate.py::fuse/long_gates/quality/calculate`: inherited
   historical robust fusion, phase-transfer gates, near-field nuisance-range
   scans and equal packet-budget comparisons. These reusable functions are
   preserved; they do not establish extra ACDE measured improvements. The
   unrelated historical 90-run CLI was removed because its statistics are not
   part of this selected submission. Default ACDE path is src/retest.py.
6. `src/wical/original/per_subcarrier_calibration.py`: earlier broadside
   graph-phase audit; standalone CLI requires authorized processed state NPZ,
   occurrences/calibration JSON and config supplied explicitly. It is an
   advanced diagnostic, not required for ACDE replay or public quick start.
7. `src/wical/original/sage_adapter.py`: preserved optional native adapter;
   no third-party Widar2 source is bundled. `fit` also retains its inherited
   multi-path interface for authorized local experiments; this submission's
   measured headline is restricted to static AoA-only one-path profiles.

For a three-chain covariance-reference vector x and unit-magnitude steering
vector a(theta), the least-squares complex gain is a^H x / 3; maximizing its
profile power gives |a^H x|² / 9. Tone powers are averaged before angle selection.
This direct computation already existed in the actual project; no original
Widar2 implementation was rewritten for packaging. See NOTICE.md.

No TX/RX/person coordinates enter `estimate_point` peak selection. Geometry is
joined by src/evaluate_results.py only after checking the freeze hashes.
The legacy plotting utility src/compare.py keeps the ACDE archival labels;
the new evaluator uses B_new in its concise report.
