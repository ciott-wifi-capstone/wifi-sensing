> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# B/C follow-up of uniform noncoherent MUSIC — 6 November 2026

Read `统一改进验证报告.md` for the full Chinese report and `diagnostics/BC原因审计.md` for B/C causes. `PCD_failure_review.md` explains why no PCD first-half estimates are available. All hypotheses and six new arms are defined in `protocol.json`; the ten previous arms are retained unchanged.

Run from the group-meeting workspace with the existing Python environment:

```powershell
& 'python' -B -X utf8 'output/noncoherent_bc_followup_20261106/predict_followup.py'
& 'python' -B -X utf8 'output/noncoherent_bc_followup_20261106/score_followup.py'
& 'python' -B -X utf8 'output/noncoherent_bc_followup_20261106/finalize_followup.py'
```

The predictor depends on the preserved neighboring `noncoherent_uniform_20261106` and `coherent_array_validation_20261106` sources/caches and their imported local dependencies. It verifies their hashes, reuses exactly the old packet selection and writes only into this follow-up directory. No raw recapture or large search is performed. The scoring script is the sole new step joining standing-point geometry. Finalization only adds interpretation, missing-output attribution and figures to the generated report. Run these steps in order; finalization expects the freshly generated report from scoring.

Data budget: 864 distinct packets per arm/run; R1 first 288 for reference and reliability; R2 first 432 for prediction and later 432 for checks. R3 remains outside fitting. These are repeatedly inspected development data, not a new independent holdout.

The primary PCD-R1 method has no first-half numerical outputs at any of the five sites under its frozen rank/tone-support gates. This is not a measured human miss rate. SIGNED-R1 has the same five first-half peaks as the previously frozen SIGNED method, with 19.13-degree mean candidate-to-geometry separation and zero accepted candidates; no new weighting gain or stable overall twelve-antenna benefit is established.

Principal tables: `all_method_summary.csv`, `all_scored_predictions.csv`, `five_point_comparison.csv`, `gain_checks.csv`, `missing_output_attribution.csv`, `state_diagnostics.csv`, `tone_support.csv`.

Figures are supplied as both PNG and vector PDF. Predictions, protocol and kernel are hash-frozen before truth scoring. Raw data, old analytical outputs and the thesis are preserved. No Widar method is used.
