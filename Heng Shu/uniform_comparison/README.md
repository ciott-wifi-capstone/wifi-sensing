# Switched Wi-Fi Array: Noncoherent MUSIC Comparison

## Objective and scope

This research code compares ten fixed MUSIC-based methods for direction candidates from a switched twelve-antenna Wi-Fi array. It includes ordinary background projection (BG), positive/signed covariance changes, whitened background projection (WBG), and R1 empty-scene reliability weighting. The primary experimental method is six-state WBG-R1, compared with G2 BG and a same-kernel G2 WBG baseline.

This is a **minimal public candidate for the final uniform comparison**, not the complete thesis working directory or a validated human-localization system. Raw acquisition, PicoScenes decoding, historical explorations, full twelve-element coherent reconstruction, independent calibration intake, and thesis/presentation generation are outside this package. Their omission does not mean they were never implemented in the wider project.

## Requirements and setup

Tested software environment: Python 3.12.7 on Windows, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, Matplotlib 3.9.2 and threadpoolctl 3.5.0. A GPU, live radio, MATLAB and network access are not required for the software tests.

From this folder:

```sh
python -m venv .venv
# Activate .venv using the normal command for your shell.
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

Dependencies were checked in the existing environment; a fresh installation is a separate check. Do not run with Python `-O`: the archived integrity checks use assertions. See [review verification](docs/VALIDATION.md).

## Public example: synthetic software checks

```sh
python -B src/predict_uniform.py --self-test --output-dir outputs/synthetic
```

Use a new or empty output directory. The tests generate their channels mathematically; no recorded human data is bundled. Expected checks cover a known-source MUSIC null, identity whitening versus BG, common phase/slope invariance, missing-state rejection and endpoint-minimum selection. `physical_validation` must remain `false`. These software checks do not establish measured accuracy.

## Reproduce the private archived comparison

Obtain the authorized original private input archive and its raw CSI/event files. See [the exact input contract](docs/INPUTS.md). Paths in these examples are placeholders:

```sh
python -B src/predict_uniform.py --input-dir /path/to/private/coherent_array_validation_20261106 --output-dir outputs/reproduction
python -B src/score_uniform.py --input-dir /path/to/private/coherent_array_validation_20261106 --output-dir outputs/reproduction
```

Prediction verifies the original cache/source digests, packet counts and chronology, runs all ten methods, and freezes predictions before the separate scorer reads the surveyed human coordinates. It will stop when required private inputs are missing. The package alone cannot reproduce measured results.

## Input, method and output

The A-E dataset has **five human-present records and ten paired empty records**, with empty → human → empty at each site. It is not three independent human repeats per site. Each arm/run uses 864 distinct packets. Twelve-antenna processing uses 24 cycles × 6 states × 6 packets; G2 uses 24 batches × 36 packets. The states are processed as simultaneous three-chain covariances and fused noncoherently. No cross-state simultaneous observation is invented.

R1's first eight cycles/batches fit the background and weights. R2's first half predicts, its second half checks temporal consistency, and R3 is an empty control. Full/half bands, competing peaks, boundary cases and foreground/background contrast are retained. [Method details](docs/METHOD.md) and [the frozen protocol](configs/protocol.json) document the fixed thresholds.

Outputs include all-method predictions, candidate alternatives, state/tone diagnostics, frozen reliability weights, spectra and integrity receipts. The scorer adds geometric deviations and figures. Local paths can appear in generated provenance reports: keep outputs private by default.

## Archived results and interpretation

The 2026-11-06 archived mean candidate-to-geometry deviations were:

- G2 BG: 22.2515°.
- Same-kernel G2 WBG: 28.7935°.
- Primary twelve-antenna WBG-R1: 25.5265°.

All ten methods had **0/5 cases passing the complete candidate checks**. These are development-data diagnostics, not validated human AoA errors, population detection rates, or independent test-set performance. The package preserves rejection and `physical_release=False`; it does not output a validated 2D position or physical ToF. See [case-by-case interpretation](docs/INTERPRETATION.md).

## Known limitations

- The data were repeatedly examined during development; splitting their time/frequency blocks does not create a new independent experiment.
- Fixed inter-chain response, physical phase calibration, background drift and human-path attribution remain unresolved. Common phase cancellation in a same-frequency covariance does not remove fixed inter-chain response.
- Six-state noncoherent fusion does not automatically obtain the resolution of a calibrated 0.308 m coherent aperture.
- Twelve-antenna overall benefit, human AoA ≤8°, and 2D localization RMSE were not validated. A promising point-specific branch is not an overall performance claim.
- The private archive and its source paths are required for measured reproduction. The public package does not include a raw-data decoder or capture controller.

## Files and reuse

`src/` contains the prediction/scoring scripts and portable array constants. `tests/` contains data-free checks. `configs/` and `docs/` contain the protocol, method, private-input schema and source notes. `requirements.txt` pins the observed environment.

This candidate excludes captured CSI, raw logs and cache arrays, participant photographs, thesis/signature/clearance documents, local account details, imported papers and third-party slide tools. Check [source and redistribution notes](docs/SOURCE_NOTES.md) before publishing. No new open-source license is granted by this package. The owner's public-submission authorization and remaining provenance uncertainties are recorded in the enclosing [rights notes](../docs/RIGHTS_AND_SOURCES.md).
