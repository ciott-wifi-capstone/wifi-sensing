# Workflow inventory

This index covers located research Python source only. The acquisition implementation is still missing. Generated results and private manifests are not included.

- `capture_protocol_20261105/`: 1 Python source files.
- `coherent_array_validation_20261106/`: 5 Python source files.
- `direct_path_geometry_ablation_20261102/`: 3 Python source files.
- `e1_e3_mapping_phase_timing_audit_20261104/`: 8 Python source files.
- `empty_background_validation_20261103/`: 6 Python source files.
- `empty_duration_matched_20261025/`: 4 Python source files.
- `empty_pi_sign_validation_20261104/`: 8 Python source files.
- `empty_reference_static_recovery_20261024/`: 17 Python source files.
- `empty_scale_validation_20261106/`: 6 Python source files.
- `experiment_analysis_20261010/`: 1 Python source files.
- `four_point_blind_peak_truth_comparison_20261104/`: 4 Python source files.
- `g1_chain2_weighting_20261106/`: 3 Python source files.
- `g_subarray_precalibration_20261105/`: 4 Python source files.
- `human_test_sites_20261105/`: 1 Python source files.
- `independent_phase_calibration_20261106/`: 2 Python source files.
- `initial_phase_calibration_protocol_20261105/`: 1 Python source files.
- `initial_phase_feasibility_20261102/`: 2 Python source files.
- `inverse_array_calibration_20261102/`: 7 Python source files.
- `kong1_kong1a_20261105/`: 3 Python source files.
- `multi_peak_branch_validation_20261105/`: 4 Python source files.
- `music_subarray_gain_20261105/`: 2 Python source files.
- `new_a_paired_20261105/`: 3 Python source files.
- `noncoherent_bc_followup_20261106/`: 6 Python source files.
- `noncoherent_uniform_20261106/`: 4 Python source files.
- `office_static_recovery_20261028/`: 6 Python source files.
- `paired_bc_20261105/`: 2 Python source files.
- `paired_be_abcde_20261105/`: 4 Python source files.
- `precalibration_comparison_20261102/`: 9 Python source files.
- `ratnam_phase_timing_validation/`: 14 Python source files.
- `real_cd_paired_20261105/`: 2 Python source files.
- `same_frequency_spatial_background_20261104/`: 7 Python source files.
- `short_long_branch_20261106/`: 3 Python source files.
- `static_analysis_20261018/`: 7 Python source files.
- `static_human_aoa_validation_20261101/`: 16 Python source files.
- `static_reanalysis_20261020/`: 1 Python source files.
- `teacher_music_transfer_20261105/`: 2 Python source files.

## Current main chain

1. HT20/HT40 decoding and dataset integrity checks: static_analysis, experiment_analysis, office_static_recovery and static_human_aoa_validation provenance.
2. Relative phase/timing, empty-reference checks, sign/mapping audits and within-state covariances: ratnam_phase_timing_validation, empty_* and same_frequency_spatial_background.
3. Latest paired A-E processing and geometry-independent prediction: new_a_paired, paired_bc, real_cd_paired and paired_be_abcde.
4. Subarray selection and branch diagnostics: teacher_music_transfer, music_subarray_gain, multi_peak_branch_validation and short_long_branch.
5. Twelve-element conditional coherent assembly and sensitivity checks: coherent_array_validation, g1_chain2_weighting.
6. Uniform noncoherent ten-method comparison: noncoherent_uniform; the portable supported entry point is uniform_comparison/.
7. PCD follow-up and empty-only noise-scale checks: noncoherent_bc_followup and empty_scale_validation.
8. Known-source calibration intake and future capture protocols: independent_phase_calibration and initial_phase_calibration_protocol. The input template is intentionally incomplete and should fail preflight until real measurements and metadata are supplied.

The full exact file list and digests are in FILE_MANIFEST.csv. Historical method documentation is retained with an archival notice. Personal thesis-editing/submission helpers, raw data and third-party tools are excluded. Superseded revisions are retained only in the explicitly named archival folders listed in SUPERSEDED_CODE.md.
