# Attribution, rights and missing provenance

## Supplied-code origin

The project owner reports that supplied code came from a senior labmate. On 2026-11-09, the owner confirmed that permission from the labmate and relevant rights holders had been obtained to submit this selected existing-source package publicly to ciott-wifi-capstone/wifi-sensing under Heng Shu. This records the owner's authorization; no separate permission document or independent legal verification was reviewed. The labmate's preferred attribution, exact contributed files and original-versus-adapted portions remain undocumented. Do not label all files as independently authored by Heng Shu, nor invent the contributor's identity.

## Current release status

This package is prepared for Heng Shu's publicly authorized contribution to ciott-wifi-capstone/wifi-sensing. No project-wide open-source LICENSE was found in the reviewed local source. This preparation grants no MIT or other license. The owner's publication authorization does not establish a license for downstream reuse or independently resolve institutional, laboratory or third-party obligations.

The owner's confirmed public-submission scope is the selected existing research source and included method documentation. Attribution and provenance of any adapted third-party or prior-student implementation still require documentation; this review has not independently established those boundaries. The acquisition/firmware source is intentionally deferred for later addition and needs its own review. Public GitHub availability and a citation do not by themselves establish redistribution permission.

## Decoder references

The local NumPy decoder and its audit notes refer to these format sources:

- https://github.com/wifisensing/RXS-Parsing-Core
- https://github.com/wifisensing/RXS-Parsing-Core/blob/master/RxSBasicSegment.cxx
- https://github.com/wifisensing/RXS-Parsing-Core/blob/master/CSISegment.hxx
- https://github.com/dhalperi/linux-80211n-csitool-supplementary
- https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bfee.c
- https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bf_file.m

In the 2026-11-09 review, GitHub metadata returned no identified license for either upstream repository and their inspected recursive trees contained no LICENSE, COPYING or NOTICE file. The upstream Intel decoder carries Daniel Halperin's copyright notice. The PicoScenes header identifies Zhiping Jiang as creator. Those upstream C/C++/MATLAB implementations, driver binaries and firmware have not been bundled here.

The historical upstream commit used to check the local implementation is not pinned. The local files do not establish whether the NumPy implementation is wholly independent or adapted from protected source. That derivation remains unverified despite the owner's public-submission authorization; attribution alone is not a license grant.

## Method references and scope

- Ratnam et al., IEEE Transactions on Wireless Communications 23(9), 10820-10833 (2024), DOI https://doi.org/10.1109/TWC.2024.3376332 . The phase/timing module documents a multi-chain, first-packet relative-reference adaptation, not physical calibration or a verbatim full reproduction of the paper's initialization.
- WiCAL: https://arxiv.org/abs/2505.21408 . Group-internal fixed-response calibration and between-group alignment are distinct. The local linear-array diagnostics do not establish the published system's performance.
- Noncoherent subarray context: https://arxiv.org/abs/1704.06000 . The local background-projection score is not claimed to be that paper's maximum-likelihood method.
- `teacher_music_transfer_20261105` contains a historical method-transfer experiment. Its directory name alone does not identify an author, establish that code was copied, or prove permission. Confirm any actual prior-student or supervisor code reuse.

NumPy, SciPy, pandas, Matplotlib, threadpoolctl, Pillow and tabulate remain external dependencies under their own terms; their distributions are not vendored. The optional pypdf dependency is needed only by an archived PDF-QA script.

AI assistance was used for this review and packaging. Course/repository attribution requirements and historical code authorship still require verification. Reference-paper PDFs, thesis/clearance files and third-party presentation tools are excluded.
