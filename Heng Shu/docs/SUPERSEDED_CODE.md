# Superseded source retained for history

The requested existing-source archive includes these old snapshots, but they are not supported entry points and are not run by the check harness:

- `research/output/empty_background_validation_20261103/superseded_frequency_alias/transfer_common.py`: superseded frequency-alias version. Use the current sibling transfer_common.py for that stage and preserve the recorded failed validation boundary.
- `research/output/noncoherent_uniform_20261106/before_endpoint_fix/`: pre-fix candidate selection snapshot. The old rule could admit non-minimum endpoints as alternatives. Use the current parent predict_uniform.py or portable uniform_comparison/ entry point. The recorded endpoint correction did not change the main spectra, primary peaks or the 0/5 complete-candidate result.

These files preserve development history, not alternative validated scores. Do not promote their outputs as final results or silently mix them with the corrected current implementation.
