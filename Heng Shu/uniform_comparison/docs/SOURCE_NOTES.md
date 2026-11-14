# Source and redistribution notes

This candidate package is derived from the local project scripts `output/noncoherent_uniform_20261106/predict_uniform.py` and `score_uniform.py`, with the array constants from `output/static_human_aoa_validation_20261101/geometry.json`.

Packaging changes are limited to replacing the multi-stage historical import chain with those same geometry constants, adding configurable private input/output paths, preserving generated-output hashes, refusing a nonempty prediction output directory, and making fresh-plot QA claims accurate. Numerical algorithm functions and thresholds are unchanged. Source/derivative digests are recorded in SOURCE_MANIFEST.json.

Dependencies are NumPy, SciPy, pandas, Matplotlib and threadpoolctl, installed from their normal package distributions. Their code is not vendored here and their own licenses apply.

The wider research project contains a PicoScenes/Intel 5300 decoder that cites https://github.com/wifisensing/RXS-Parsing-Core as a format reference, and a Ratnam-inspired relative phase/timing module citing DOI 10.1109/TWC.2024.3376332. Those modules and external reference PDFs are not redistributed in this minimal candidate. The historical `teacher_music_transfer_20261105` stage is also omitted; its name alone does not establish third-party ownership or redistribution permission.

No project-wide open-source license was found in the reviewed source directory. No MIT or other license has been granted by this packaging work. For the enclosing existing-source submission, the owner confirmed on 2026-11-09 that permission from the senior labmate and relevant rights holders had been obtained for public submission. This is the owner's authorization, not independent verification of authorship, institutional obligations or upstream licenses. Select a reuse license only when the relevant rights holder approves it. Publication permission and an open-source reuse license are separate questions. See ../../docs/RIGHTS_AND_SOURCES.md for the enclosing submission's provenance and remaining uncertainties.

AI assistance was used in preparing this review package. Verify the applicable course/repository attribution requirements before submission; this note is not a claim that every historical line has the same authorship.
