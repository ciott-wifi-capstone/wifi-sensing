# Existing-source review: 2026-11-09

Passed checks on the final local candidate:

- 186 Python files parsed successfully, including 178 archived research sources, the portable comparison and the new test/entry-point utilities.
- 11 synthetic HT20/HT40 decoder unittest cases passed: signed bit extraction, chain permutation, NIC/counter wrapping, malformed/truncated frames, incompatible format and invalid permutations.
- Four portable uniform-comparison unittest cases passed.
- Eleven isolated research check groups passed: uniform MUSIC, conditional coherent twelve-element assembly, PCD, empty-only noise-scale models, G1 weighting, independent phase-calibration software, Ratnam core/extended/deep-fade cases, empty-reference preservation and baseline/graph synchronization controls.
- The decoder command rejects optimized Python mode, existing output files and unsupported input options. No device, radio, network configuration or real capture operation is involved in these tests.
- The portable comparison had already been rerun against authorized private archived measurements: twelve parsed CSVs and all 1,311 spectrum arrays matched the original values exactly, including NaN positions. Five-point ten-method rejection results were preserved.
- Source and sanitized-file digests were verified. Original project files were not changed.
- A bounded text-pattern scan of the final candidate found no credential-token/private-key patterns, user-specific home paths, captured MAC literals, private IPs, student ID or private Overleaf-project links. A pattern scan is not proof of every possible privacy or licensing issue.

The environment was the existing Windows Python 3.12.7 installation with requirements.txt versions. Clean installation, Linux/macOS runs, most historical end-to-end measured workflows, new hardware measurements, physical calibration and human-localization performance were not tested. Optional historical PDF QA requires pypdf, absent from the reviewed numerical environment. Generated figures were not freshly reviewed visually in this code audit.

Superseded revisions are included only as explicitly labeled source history and were not run by the checks. Historical dataset-specific decoder mains are not supported as current generic entry points. The safe decoder CLI invokes the byte decoder only.

Passing these software checks does not establish public redistribution rights. On 2026-11-09, the owner confirmed permission from the senior labmate and relevant rights holders for this public submission. This records the owner's authorization rather than independent verification of contributor boundaries, upstream licensing or legal rights. Acquisition/controller and firmware source are deferred by the current submission scope. These checks were performed locally before submission; publication is separate from scientific validation.
