# Public synthetic example

`synthetic_covariance.csv` is an analytic, single plane-wave covariance at +23°.
It has six states × 30 tones × 9 entries, with 0.05 diagonal noise. Per-tone
complex gains vary deterministically. No measured CSI or personal data appear
in this fixture. It tests the software pipeline without a radio or private data.

From the personal project folder:

```text
python -B src/run_example.py --input examples/synthetic_covariance.csv --out local_results/synthetic
```

The program calls the actual `replay.estimate_point` path for all three branches.
`estimates.csv` gives branch/method estimates and `spectra.csv` gives angle-grid
powers. Each output row is marked synthetic. G3 and six-state product select
+23° on raw and background branches. The phase diagnostic is diagnostic only;
do not interpret this synthetic experiment as human localization accuracy.
The two mapping variants are intentionally identical in covariance input; real
AntSEL behavior is tested separately with in-memory bit-packed records.
