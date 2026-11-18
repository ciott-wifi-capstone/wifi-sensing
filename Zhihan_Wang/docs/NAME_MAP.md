# Source and name map

Original research scripts and their maintained components:

| Original source | Current component |
| --- | --- |
| `analyze_new_motion_spatial_fingerprints.py` | `src/wical/motion.py`, `scripts/analyze_motion.py` |
| `analyze_static_quality_gate.py` | `src/wical/selection.py` |
| `recalculate_static_spatial_audit.py` | `src/wical/csi.py`, `src/wical/static.py`, `scripts/analyze_static.py` |
| `recover_static_waveforms.py` | `scripts/analyze_static.py` |
| `run_m3f_full_sqi.py` | `scripts/analyze_static.py` |
| `run_m3g_sqi_evaluation.py` | `src/wical/selection.py`, `scripts/evaluate_selection.py` |
| `test_wical_sqi.py` | `tests/test_sqi.py` |
| `wical_sqi.py` | `src/wical/sqi.py` |
| `WiCAL_FarSense_STATIC_clean.ipynb` | `src/wical/reference.py` |
| `external/farsense/farsense_single_end.py` | `src/wical/farsense.py` |

Earlier timing diagnostics, motion pilots and duplicate versions remain in
the research archive. Original filenames and hashes are recorded in
`source_map.json`.

`fill_short_nan_for_detection` was renamed `fill_detection_gaps` because it
interpolates gaps of any length. The calculation is unchanged.
