# Private input contract

The public package can run software checks without measured data. Reproducing the five-point results requires the original private `coherent_array_validation_20261106` archive and its source recordings. This is an archived-cache workflow, not a raw-data decoder or acquisition program.

For each site A, B, C, D and E, the input directory must contain:

- `<site>_selected.npz`: `switched` complex array of shape `(3,24,6,6,3,30)` and `g2` complex array of shape `(3,24,36,3,30)`.
- `<site>_packets.csv`: packet ledger with run, arm, sample, state, packet, time_s and visit_dt_ms columns. The original archive has 864 unique packets per run/arm and disjoint chronological first/last halves.
- `<site>_sources.csv`: original raw CSI paths, SHA-256 digests and the corresponding events.log digests. These private paths must resolve locally; `--input-dir` relocates the cache archive only, not the source paths embedded in these private ledgers.
- `<site>_complex_covariances.npz`: the earlier `G2` covariances used by the consistency assertion.
- `<site>_music_profiles.npz`: the earlier G2 background-projected profiles used for numerical regression.
- `prediction_freeze.json`: the original per-file hashes, including the selected caches and source/packet ledgers.

The first array axis is R1 empty before, R2 human present, R3 empty after. Six states are ordered G1, G2, G3, G4, C1, C2. The last axes are three simultaneous receiver chains and 30 frequency groups. See `array_geometry.py` for the exact geometry and mapping.

Do not fabricate private source ledgers, remove integrity checks, or silently update old hash receipts to make different inputs pass. Relocating an original raw recording requires a separately documented provenance migration and renewed verification. Supply only trusted local NPZ/CSV/JSON inputs. NumPy object arrays are not required.

Prediction writes CSV diagnostics, spectra.npz, weight and prediction freeze receipts, and software checks into a new empty output directory. Scoring then writes all-method and five-point CSVs, diagnostic figures and a Chinese report into that same output directory. Generated files may contain local raw-data paths and must remain private unless separately sanitized and approved.

The five measured sites are hard-coded only in the scoring script for this archived study. For a different dataset, define a new protocol and scorer rather than reporting its outputs as a reproduction of the archived five-point study.
