Adds `Xiangdong_Lu/` for the ELEC5021 WiCAL static sensing project: G3 three-Rx
AoA-only SAGE baseline versus noncoherent fusion of six switched antenna states.
The 12 physical antennas are switched through three simultaneous Rx chains.

Includes executable parsing/AntSEL/RF-state audit and estimation code, fixed
configuration, measured numerical results, pinned dependencies, a small labelled
synthetic example, tests and reproduction/data-availability documentation.
Only the personal project folder is changed.

From `Xiangdong_Lu/`, after installing requirements:

```text
python -B -m unittest discover -s tests -v
python -B src/run_example.py --input examples/synthetic_covariance.csv --out local_results/synthetic
python -B src/evaluate_results.py --tables results/acde --out local_results/measured
python -B src/verify_manifest.py
```

Validation: 10 tests passed; all four accessible private capture sessions were
replayed, and key numerical outputs matched the archived results byte for byte.
See VALIDATION.md for scope and evidence.

Run1 subtraction gives provisional four-point MAE 19.70° (G3) and 7.28°
(six-state), a 63.0% reduction in this selected conditional comparison. Physical
AntSEL/sign convention and human-specific path identity remain unverified.
One static-person run per point, point replacement/retests, multipath and phase
drift limit generalization; alternate mappings/branches/empty controls remain.

Real raw CSI and private materials are excluded. The third-party original
Widar2 Python file is also excluded because its redistribution license was not
verified; an optional native audit can use a separately authorized local copy.
The public default uses the existing direct one-path AoA profile. No new
commercial or open-source relicense is asserted. See NOTICE.md and LICENSE.

Please review and merge after approval.
