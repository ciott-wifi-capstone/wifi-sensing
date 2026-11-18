# Attribution and third-party status

Xiangdong Lu — USYD ELEC5021 WiCAL static sensing project.
The package preserves inherited WiCAL processing scripts and adds packaging,
tests, synthetic fixtures and result-reproduction tools. It does not assert that
all inherited code was independently authored by the student; consult project
records and university rules for acknowledgement of assisted work.

## Widar2

Kun Qian, Chenshu Wu, Yi Zhang, Guidong Zhang, Zheng Yang and Yunhao Liu,
“Widar2.0: Passive Human Tracking with a Single Wi-Fi Link,” MobiSys 2018,
pp. 350–361. https://doi.org/10.1145/3210240.3210314
Author project: https://tns.thss.tsinghua.edu.cn/wifiradar/

The local `widar2_original.py` has no accompanying verified redistribution
license. The author project/paper is a reference, not proof that this particular
Python file is official or that it may be redistributed. No code from that file
is included, extracted, rewritten or relicensed in this public folder.
The local audit used this SHA-256:
`1d39f976b819f6b07b77d5f72772707c3a001c4ffb0bddde10d4594d003d5b4c`.

`src/wical/original/sage_adapter.py` is an inherited WiCAL adapter. Its optional
`load_core()` reads five named functions from an external permission-cleared
file; the external file's module-level imports/demo are not executed.
The default pipeline uses the already-existing NumPy angle profile in
`src/wical/estimate.py`. For the restricted one-path, one-tone, AoA-only fit,
profiling out a complex gain gives |a(theta)^H x|² / 9. This is the same
profile used in the historical pipeline and checked by the native core, not a
replacement implementation of the full Widar2 tracker, multipath SAGE, ToF or
Doppler solver. Native cross-check status is recorded explicitly.

## CSI decoding and RF control

The inherited decoder implements the Intel 5300 bit-packing convention inside
the fixed 385-byte PicoScenes CSI:v3 layout observed in these captures.
Reference projects: https://github.com/dhalperi/linux-80211n-csitool-supplementary
and https://ps.zpj.io/ . Neither their code nor their datasets/firmware are
bundled. It is not a generic or officially validated PicoScenes parser.
RF switch maps come from local READY capture records; independent physical
wiring/sign verification is outstanding.

NumPy and Matplotlib are external dependencies installed separately, not
vendored. Their licenses remain those of the respective projects.
