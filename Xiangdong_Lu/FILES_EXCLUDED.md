# Files excluded from the public upload

- All real raw CSI and session logs (~49.45 MB across the four selected session
  trees before compression); access is permission-controlled, not missing locally.
- Unlicensed/unverified-redistribution `widar2_original.py`; optional native
  audit requires a separately authorized local file. Full Widar2 tracking demo,
  its picos/scipy/MATLAB inputs and third-party data are not included.
- Thesis/report Word/PDF files, photos, presentation materials, historical ZIPs,
  temporary output, caches, local environments, credentials and personal paths.
- Identifying acquisition fields: session, boot_id, raw_path, csi_file in public
  result tables; full manifest/runtime/kernel/session logs remain private.
- Duplicate historical packages; the current ACDE pipeline and inherited
  robust/equal-budget/near-field functions are retained once in src.
- Old-B data are not inserted into this selected four-point result set.

No public package can independently reproduce real CSI decoding without the
private capture inputs. It can reproduce metrics from measured tables and run
the end-to-end estimator on synthetic covariance without them. The preparation
audit additionally replayed the actual accessible private captures locally.

Historical `/mnt/data/...` paths were not present in this Windows workspace.
Matching archives were located under the existing Codex project directories.
See docs/SOURCE_PROVENANCE.json and VALIDATION.md for verified evidence.

- Original teacher summarize.py is replaced by the public hash-verified metric
  evaluator, not silently omitted functionality. Its source hash remains in
  docs/SOURCE_ARCHIVE_VERIFICATION.json.
- ESP32 device firmware and radio acquisition program source were not in the
  selected teacher archive. READY timing analysis is included; acquisition is
  not represented as a tested firmware deployment.
