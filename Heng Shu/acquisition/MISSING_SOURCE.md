# Acquisition source not yet supplied

This folder is an inventory, not an acquisition implementation. Acquisition/controller and firmware source are intentionally deferred for later addition in the current submission scope. Do not claim that this review candidate is a complete capture-to-analysis release.

Latest HT40 records identify these six scripts, whose expected source digests are in required_source_hashes.json:

- local_intel4_capture.py
- picoscenes_switching.py
- picoscenes_stability.py
- picoscenes_qc.py
- esp32.py
- kernel_stream.py

These files were not present in the inspected Windows project and capture directories. The original Linux project is named WiCal_RX; the observed controller was its tools/esp32.py. Retrieve the actual project source, dependency/build information and publication permissions from the acquisition computer. The digest list helps verify the version; it is not a substitute for source code or permission.

The controller identifies ESP32 firmware `sp4t_triple_sync v1.0`. Firmware source, board/build settings, GPIO mapping and firmware binary digest are absent. The observed protocol includes IDN?, RESET, DWELL, SEQ, RUN and READY responses, but this package does not reconstruct firmware from logs.

Historical HT20 data identify a separate wical-ht40-pair-runtime project and wical.local_capture module. Retrieve that source too if the requested full submission includes the earlier capture stack.

Verified software/hardware context from capture records:

- Latest: Intel 5300, PicoScenes Platform 2025.1217.1547, Ubuntu-family Linux kernel 6.5.0-15-generic, an ESP32 and three SP4T switch chains. Six switching states, 100 microsecond settling plus 25,000 microsecond dwell, nominal 1000 packets/s.
- Historical HT20: an Intel CSI-tool stack on kernel 4.2.0, channel 40 and nominal 2000 packets/s. This is a separate setup, not an interchangeable dependency list.

Recorded acquisition preparation disabled network management, changed interface/channel state, unloaded/reloaded Wi-Fi drivers, changed power/CPU settings and commanded the controller. No acquisition command is part of this package's default installation or tests. Source review, hardware access and any required explicit approval are necessary before executing that workflow.

Raw logs/configs include hardware IDs, host paths, addresses and private-network information and are not included here. Use newly sanitized configuration examples after the missing source has been reviewed.
