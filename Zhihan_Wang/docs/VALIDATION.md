# Technical validation

Tested on macOS with Python 3.12.1, NumPy 2.2.6, SciPy 1.15.3 and
Matplotlib 3.11.2. The optional raw decoder uses csiread 1.4.1.
Dependencies were installed in an independent virtual environment; `pip check`
passes. The software checks are run with `python -B scripts/check_project.py`.

| Check | Result |
| --- | --- |
| 32 software tests | Estimator/SQI math, identity and timestamp validation, GT isolation, decoder metadata, output protection, CLI errors, paired reference parsing and motion fingerprint edge cases pass |
| Synthetic example | All 18 generated observations return 15 bpm; not a measured accuracy result |
| Static table evaluation | All eight-run MAE, median and maximum errors match the frozen full-precision exports |
| Static saved-array replay | 144 observations and 864 rate/signal/ACF/subcarrier/weight/angle comparisons match, with zero maximum absolute difference; all SQI factors also match |
| Motion saved-array replay | Eight recordings, 22 candidate events and 396 event-observation feature rows match the original outputs |
| Motion aggregation | 144 run-observation fingerprints, 28 run-pair and three condition-pair correlations match the original exports |
| Reference parsing | Malformed time/value pairs are rejected; 5,958 valid exported samples equal the original parser's arrays |

The static replay result applies to the unchanged estimator/SQI computation.
Motion aggregation uses the original event medians, robust normalization and
Spearman correlation; thresholds and windows are unchanged. Tests include
missing/duplicate observation coverage, row-order independence, no-event runs
and undefined correlations for constant fingerprints.

Full final-capture CSI replay, physical antenna mapping, manual belt landmark
acceptance and hardware collection have not been independently validated.
Saved-array agreement establishes computational reproducibility, not physiological
accuracy or synchronized action labels. Input/time/quality warnings remain in
each run's `warnings.csv`; a successful run is not a claim that these limitations
have been resolved.

The documentation and code-formatting update was also checked on Linux with
Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8. All 32 tests
passed. The synthetic example and static table evaluation produced identical
numeric outputs before and after the update. Bundled result tables and motion
figures were unchanged. Private saved-array and hardware checks were not repeated
for this update.
