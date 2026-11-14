# Review verification: 2026-11-09

- All four Python files parsed successfully.
- Four data-free unittest cases passed, including the archived multi-check synthetic routine.
- Missing private inputs and a nonempty prediction output directory were rejected with exit code 2.
- All 17 prediction function bodies are AST-identical to the original 2026-11-06 script. Packaging altered imports, path selection and the command wrapper only.
- The final candidate ran prediction and scoring on the original authorized private archive. Its 15 CSI/event source pairs, frozen cache entries, packet budgets and chronology passed their existing checks.
- Twelve output CSVs matched their original parsed contents exactly, including the 100-row prediction/scoring records and all ten method summaries.
- All 1,311 arrays in spectra.npz matched the archived arrays exactly, including NaN locations. Maximum finite absolute difference was 0.
- New prediction receipt hashes passed, including the separately copied array-geometry module. All six source-file digests in SOURCE_MANIFEST.json remained unchanged.
- The final candidate was checked for credential patterns, private-key blocks, user-specific absolute paths and MAC-address literals. None were found by that bounded scan; this is not a proof of every possible disclosure or licensing issue.

The run used the existing Windows Python 3.12.7 environment and the versions pinned in requirements.txt. A clean-environment installation, Linux/macOS runs, new acquisition, independent physical calibration, and validated human localization were not tested or established. Generated figures were created successfully but were not re-reviewed visually in this code audit.

This standalone uniform-comparison subdirectory contains 14 files. The enclosing existing-source submission adds separately documented research sources, decoders, configurations and checks. Private rerun outputs, private source paths, captured measurements and the private audit report remain excluded. These checks were performed locally before submission; see the enclosing docs/RIGHTS_AND_SOURCES.md for publication authorization and unresolved provenance details.
