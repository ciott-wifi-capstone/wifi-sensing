# Results and interpretation

All published numerical tables originate from the 2026-11-05 ACDE teacher
package. No angle, spectrum or measured error is synthesized or improved.
Sanitization removes identifying columns and normalizes CSV encoding only.
Archival E = reported B_new at (1.64,-2) m. Discarded old B is absent.

- `blind_predictions.csv`: point, run, role, mapping (`cov`/`alt`), branch,
  method (`G3`/`J6_product`), chosen angle, scores and runner-up evidence.
- `spectra.csv`: fused/G3 grid, selected flag, candidate support,
  pre-mask and unnormalized score profiles; grid is -90:1:+90 degrees.
- `state_spectra.csv`: independent six-state powers at each grid angle.
- `negative_control.csv`: Run2/Run3 angles and strength ratios. Six-state
  control strength uses the average unnormalized state spectrum, not product
  peak power. A high ratio alone does not verify a human-specific path.
- `all_conditional_errors.csv`: both AntSEL conventions and both geometry signs.
- `primary_results.csv`: retained Intel mask3, provisional positive sign.
- `geometry.csv`: coordinates and conditional bearings, used after fitting.
- `phase_tones.csv`, `empty_phase_drift*.csv`, `effective_phase.csv`,
  `state_quality.csv`: per-tone phase, inter-empty drift and quality diagnostics.
- `native_sage.csv`: original recorded per-tone native SAGE cross-checks;
  these remain historical measured results even when public default replay
  runs without the unbundled native implementation.
- `summary_for_teacher.csv`: Chinese side-by-side numerical summary.

Branches: `raw` leaves CSI covariance unchanged; `background` subtracts Run1
covariance; `kong_phase_hypothesis` adds the unverified empty-room dominant-phase
diagnostic after subtraction. That diagnostic is not verified hardware phase
calibration and is not the recommended empirical branch.

G3 selects the largest averaged per-tone AoA-only one-path profile. Six-state
fusion averages log of mean-normalized state powers (floor 0.05), exponentiates,
and restricts the result to the fixed G1–G4 coarse candidate support. It combines
powers noncoherently. Physical antennas 1–12 are switched through 3 receiver
chains; there are no simultaneous 12-channel observations.

The provisional bearing is atan2(y,3.65-x), in degrees. AntSEL mask3 mapping,
physical wiring and angle sign have not been independently verified. Alternate
AntSEL and mirrored-sign results are retained instead of silently choosing
whichever agrees with geometry.

Four points with one static-person run each do not estimate population accuracy,
confidence bounds, detection rate, tracking performance, human-path identity,
or robustness across rooms/subjects. Old B was replaced after inspection and
C/D were retested; the 63.0% reduction is descriptive for the selected four-point
conditional comparison. A itself is worse under six-state fusion (31 vs 14 deg).
Ground truth is excluded from peak selection but the reanalysis is not a
prospective blinded experiment.
