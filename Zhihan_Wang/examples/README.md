# Synthetic example

`synthetic.json` specifies a 90 s, 10 Hz, imposed 15 breaths/minute signal with
seed 20261118. `scripts/analyze_static.py` generates 30 complex subcarriers for
each of six states and three receive-chain pairs. Projection angles vary across
subcarriers; small deterministic noise prevents duplicate series.

The example contains generated signals only. It checks the estimator, SQI
and output files using a known input rate. It does not model RF propagation
or measure sensing accuracy. Measured results are stored separately in
`results/static/`.
