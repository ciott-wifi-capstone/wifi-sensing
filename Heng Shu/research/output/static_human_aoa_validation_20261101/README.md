> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# Reproduce the bounded static-human validation

The existing thesis is in `../thesis_algorithm_draft_20261019/`. Raw files are read from the absolute paths in `input_manifest.csv`; they are never rewritten. Backups contain the local and downloaded Overleaf projects before this revision.

Use Python with NumPy, SciPy, pandas, Matplotlib, Pillow and threadpoolctl. The completed run used `python`. From this directory:

```text
python check_models.py
python preprocess.py development
python state_sync.py development
python review_development.py
python preprocess.py evaluation
python state_sync.py evaluation
python music_baseline.py
python summarize.py
python review_figures.py
python evidence_audit.py
```

`audit_inputs.py` was the initial freeze and backup operation; its snapshots and `results/setup_audit.json` record that step. Completed record caches are reused by `preprocess.py`; to change an estimator, create a new version/output directory rather than editing a frozen protocol and silently mixing results. `update_thesis.py` and `refine_layout.py` are one-time manuscript migration scripts, not steps in analysis reproduction.

Treatment A: normalized raw CSI. B: common-phase-only. C: gated common timing/phase, independent state/record reference. D: frozen Office R1 reference for each state. D is unavailable for the primary records under current evidence. Failure preserves raw observations. Treatment letters are separate from thesis implementation-status A/B/C.

Each raw observation contains 3 x 30 complex entries. The six rotating masks each fit 60 entries and predict 30. The prefix common-model gate retains the inherited mixed-chain/frequency mask; it is separate from those six predictions. References and flags are attached to every saved prediction. `actual_full_packet_applied` distinguishes the causal/full-entry correction from the mask correction used solely to test predictability.

`common.py` evaluates the objective at exact sparse frequencies. Raw-fit periods are 800 ns (HT40) and 3200 ns (HT20). Valid frequency smoothing changes the HT20 MUSIC period to1600 ns. `music_baseline.py` uses actual state positions and never reads human coordinates. Human coordinates are used only by `summarize.py` after spectral outputs exist, for evaluation geometry and deterministic curvature calculations. It does not use them to select peaks or adjust phase.

The four primary development records, four primary evaluation records and Office R1/R2 are the entire new bounded estimate set. The remaining primary records are retained in inventory and previous diagnostics. The physical inference branch was not expanded because response calibration and human-path identity are unresolved. See `PROTOCOL.md`, `REPORT.md`, `results/development_review.json` and `results/physical_evaluation.json` for the exact stopping boundary.

No independent test accuracy is claimed. A successful numerical invariant, small closure, or stable spectral peak is not human-path attribution.
