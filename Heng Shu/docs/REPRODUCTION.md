# Reproduction and private-input boundaries

## Supported public checks

From the Heng Shu folder, run the decoder unittest suite and tools/check_full_project.py. They synthesize their own inputs and run mathematical checks in a disposable copy. They do not capture packets, open devices, configure interfaces or use private data.

The safe decoder command imports only decode() from a historical decoder; it never invokes that file's dataset-specific main. HT40 accepts the observed PicoScenes frame v1 / RxSBasic v4 / CSI v3, Intel 5300, one TX, three RX and thirty frequency groups with 40 MHz / 5230 MHz center / 5240 MHz control. HT20 expects fixed 215-byte Intel records with 192-byte CSI payloads. Neither is a general decoder for arbitrary captures. Do not use Python -O, because archived validation uses assertions. The supplied decoder/check entry points reject optimized mode.

## Historical source tree

The original output/<workflow> layout is retained under research/output so imports and provenance snapshots remain recognizable. It contains source/configuration, not the original result archive. Scripts often write beside themselves and some compute as soon as invoked. Use a separate private working copy for any measured-data run. Do not run every script indiscriminately or write into the original thesis directory.

WIFI_DATA_ROOT can replace direct Desktop lookups in the sanitized scripts. WIFI_TARGET_MAC supplies the transmitter address for the latest raw-frame filter; no captured address is bundled. Original CSV/JSON provenance may still hold paths relative to a historical workspace or absolute paths on the owner's private computer. Moving these inputs requires a documented provenance migration, not deleting integrity assertions or quietly rehashing different data.

Historical decoders contain dataset-specific analysis mains. In particular the earliest HT20 main used an Office 'empty' label that was later corrected to static human-present. Use tools/decode_capture.py for bytes-only decoding and the later office_static_recovery workflow for the corrected study; do not cite the earliest label as current ground truth.

Required private material varies by workflow: raw CSI/event/session files, dataset manifests, selected packet arrays, intermediate covariance/spectrum archives, frozen receipts and earlier stage outputs. These are excluded, as are reference PDFs and personal thesis files used by some historical report/audit helpers. The full historical experiment is therefore not a one-command public reproduction.

The runnable portable final comparison is in uniform_comparison/. Its input contract identifies the exact five-point cache files and source/hash requirements. With the authorized private archive, this entry point was rerun successfully during this review. It matched twelve original CSVs and 1,311 spectrum arrays exactly. Its provenance hashes include the copied geometry configuration.

## Configurations and checksums

SOURCE_MANIFEST.json records original and sanitized source digests. Newly generated config.sha256 files describe the sanitized candidate configurations only; they are not substitutes for historical experimental freeze receipts. A path/runtime-text edit changes a file hash without changing a numerical method. Old real-data integrity receipts must remain separately traceable.

The archive retains earlier approaches and failed hypotheses for research transparency. Passing a synthetic check does not validate a method on measurements. PCD and the empty-only noise-scale follow-up must retain their original stop/reject decisions. Do not use lower residuals or a stable peak as proof of calibration or human-path identity.

## Not tested or not supplied

No new capture, radio/firmware operation, physical calibration, independent human test, general localization RMSE, Linux/macOS execution, or clean installation was performed. Most historical real-data report/plot entry points were not rerun. The optional PDF QA dependency pypdf is not installed in the reviewed numerical environment. Missing capture/controller/firmware sources are listed in acquisition/MISSING_SOURCE.md.
