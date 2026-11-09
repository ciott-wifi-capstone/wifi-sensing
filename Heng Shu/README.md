# Heng Shu: Switched Wi-Fi Array Research Code

Submission destination: `ciott-wifi-capstone/wifi-sensing`, personal folder `Heng Shu`.

**Existing-source submission.** The located Windows research code is included. The Linux acquisition/controller project and ESP32 firmware source are intentionally deferred for a later addition, so this is not a complete capture-to-analysis release. See [missing acquisition sources](acquisition/MISSING_SOURCE.md) and [rights/provenance review](docs/RIGHTS_AND_SOURCES.md). On 2026-11-09, the project owner confirmed permission to submit this selected source publicly; this statement records that authorization and does not independently verify upstream licensing.

## Project objective and evidence limits

This project studies MUSIC-based direction estimation with an Intel 5300 switched twelve-antenna Wi-Fi array. The code covers decoding, relative phase/timing stabilization, background checks, subarray/branch comparisons, conditional coherent assembly, noncoherent fusion, PCD and noise-model follow-ups, and independent phase-calibration intake.

The software produces diagnostic candidates and validation failures. It does not establish a validated static-human localization system. The final uniform comparison used five human-present and ten empty records, not three independent human repetitions per site. Mean candidate-to-geometry deviations were G2 BG 22.2515°, G2 WBG 28.7935° and primary twelve-antenna WBG-R1 25.5265°. All ten methods had 0/5 cases passing the complete candidate checks. These are development-data results, not verified human AoA errors or general detection rates. Human-path identity, physical calibration, overall twelve-antenna benefit, an 8° target, physical ToF and 2D RMSE remain unverified.

## Package layout

- `research/output/`: 178 located current/historical research Python sources, 29 configuration files, and method documents. Layout preserves the original local imports. Earlier stages are historical evidence, not alternative validated final scores. [Superseded snapshots](docs/SUPERSEDED_CODE.md) are retained but are not supported entry points.
- `uniform_comparison/`: a separately documented portable entry point for the final ten-method comparison, previously checked against the original private results.
- `tools/decode_capture.py`: a read-only-input HT20/HT40 decoder command that never invokes the historical dataset-specific analysis main.
- `tests/`: synthetic format/bit-order/counter and invalid-input tests.
- `tools/check_full_project.py`: data-free research checks in a disposable copy.
- `acquisition/`: missing-source inventory and known code digests, not capture software.
- `docs/`: [workflow index](docs/WORKFLOWS.md), [reproduction contract](docs/REPRODUCTION.md), [validation](docs/VALIDATION.md) and [rights notes](docs/RIGHTS_AND_SOURCES.md).

## Setup

Reviewed numerical environment: Windows, Python 3.12.7. Required packages and observed versions are pinned in requirements.txt. No GPU, MATLAB or connected radio is needed for the data-free checks.

```sh
python -m venv .venv
# Activate the environment with the normal command for your shell.
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B tools/check_full_project.py --report outputs/checks.json
python -B -m unittest discover -s uniform_comparison/tests -v
```

A clean installation and other operating systems were not tested in this review. Do not use `python -O`, which disables assertions used by the archived algorithms and parsers. An optional historical PDF-QA helper needs pypdf from requirements-optional-reporting.txt; it is not part of the tested numerical path.

## Decode one authorized private file

```sh
python -B tools/decode_capture.py --format ht40 --input /path/to/capture.csi --output outputs/decoded_ht40.npz
python -B tools/decode_capture.py --format ht20 --input /path/to/session.dat --output outputs/decoded_ht20.npz
```

The narrow formats are described in [reproduction notes](docs/REPRODUCTION.md). The output must be a new .npz file and remains private. Decoding does not perform calibration or infer a human position. Raw datasets are not bundled.

## Run the final comparison

Use [uniform_comparison/README.md](uniform_comparison/README.md) for synthetic examples, the input contract and prediction/scoring commands. Its private-data runner was verified against the archived five-point results. The other historical workflows require their own omitted raw files, manifests and intermediate results. Run measured analysis only in a private working copy: many historical scripts write beside their source. They are not an automatic pipeline or device-control installer.

## Acquisition and safety

The latest measurements used Intel 5300, PicoScenes, an ESP32 and SP4T switch chains. Historical HT20 acquisition used a different Linux/Intel CSI-tool stack. Six acquisition scripts and the firmware source still need to be recovered from the capture computer. Logged setup commands change network interfaces, drivers and power state. No acquisition or system-setting action runs during installation or tests, and no such action was tested here.

## Publication and reuse

Do not upload raw captures, complete logs, device identifiers, personal paths, private-network addresses, photographs, signed clearance/thesis material or reference-paper PDFs. Generated measured outputs can contain private input paths and are excluded by default. This candidate includes no upstream driver, PicoScenes binary, firmware binary or third-party slide tool.

No new open-source license is granted by this package. The project owner reports that supplied code came from a senior labmate and confirmed on 2026-11-09 that permission from the labmate and relevant rights holders had been obtained for public submission of this selected existing-source package. That authorization is recorded as the owner's statement, not an independent legal or upstream-license verification. Exact contribution boundaries, preferred attribution and local decoder derivation remain undocumented, and no project-wide or upstream reuse license is inferred. The missing acquisition project will be added later and must be reviewed separately. Academic-integrity and AI-assistance requirements still apply. See [rights and source notes](docs/RIGHTS_AND_SOURCES.md).
