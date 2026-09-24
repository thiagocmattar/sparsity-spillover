# O003 ? Submission supplement aligned with the revised paper

## Question

Does the lean handoff support the current manuscript's claims, including the
31M scale, and fit the user's stated 100 MB supplementary-file limit?

## Method and coverage

Extended the existing export rather than copying the author research archive.
Read the current manuscript and the source reductions behind the six paper
figures and same-checkpoint controls. The first 84 endpoint values were checked
against their original sources and left unchanged. Added 11 31M conditions,
for 95 total: 45/11/27/12 at 14M/31M/70M/410M. Added the final 31M kernel port,
architecture pin and resolved training recipes; retained all five thresholds
0, .01, .05, .1, .5 for T2/Ph and T7/Ph.

The supplement includes the byte-identical current manuscript, six original
figure assets, three readable CSV tables, updated paper/reviewer maps, 36
scale-figure execution coordinates from 33 checkpoints, complete 31M ordinary
validation and logical-product evidence, and all 33 qualification/timing
processes. Each process covers 338 validation blocks from all 500 documents
and excludes the 1,444-token tail; each timing backend has 64 inputs ? 7 passes.
There are 29,568 raw host/event timing records across two backends and 33
processes. Four Base trajectories contain 2,848 update records.

The 14M T2 factorial reductions and missing 70M skips-off reference now support
the paper's execution-control claims. The portable CLI exposes h/z-only switches
while keeping the other kernels frozen; it is explicitly documented as pairing
each mode with native execution, whereas the historical factorial benchmark
paired all four modes together within a process. No new timing claim is made.

## Figure legend, caption and result

The exact paper assets are named by Figure 1?6 in `handoff/figures/`. The scale
figure uses recipe colors, model-size shapes, and open/filled Base markers for
kernel/PyTorch execution; vertical guides show each Base loss. Its 36 points
and caption support a descriptive cross-scale quality?latency comparison, not
a fitted scaling law. The rejected absolute/relative companion is excluded.
The first layer-sparsity PDF page is used in Figure 3; its retained second page
is explicitly described as a higher-threshold companion.

Offline reconstruction produces ten PDF views; the current 40-point Figure 1
replaces the old 14M/70M three-panel reconstruction. Figure 5 retains the actual
paper drawing body. The remaining regenerated views are not typography replicas.
Visual checks covered the scale plot, current three-panel reconstruction and
four-scale Base trajectories. Corrected text encoding in reconstructed labels
and added the missing recipe legend. Original paper assets were not changed.

The independent CPU reduction recovers 12.8985% for the 14M T2 joint h,z saving
and 57.8670% of that checkpoint's native-to-enabled gap, plus 58.2992% for 70M
against custom skips-off and 17.4% against the native h,z substitution. These
are distinct denominators and are documented as such.

## Verification and archive

- Isolated copy with only distributed files: **66 passed in 5.51s**.
- All 95 identifiers, training-order hashes, count fractions and 20 analytic
  ceilings checked; 15 matched-threshold cross-scale loss comparisons checked.
- All 31M raw samples reduce to their recorded process and endpoint geometric
  means; complete numerical qualification records pass their retained gates.
- Three shipped CSV files are byte-identical to their isolated reconstructions.
- Selected CUDA token streams match **23** measured source files after
  identifier/comment cleanup and removal of unused vocabulary tile branches.
- Manuscript and six figure assets match their sources byte-for-byte.
- No weights, datasets, compiled binaries, private archive paths or credentials
  were included; gzip contents also undergo the credential-pattern scan.
- ZIP CRCs and every extracted file hash pass. `verify` and `results` run from
  a fresh extraction, independently of the original repository tree.
- Archive: `/handoff.zip`, **3,324,241 bytes (3.324241 decimal MB;
  3.170243 MiB)**; 141 files; 6,788,797 uncompressed bytes.
- Limit used: **100,000,000 bytes**; remaining space **96,675,759 bytes**.
- Archive SHA-256: `4006134efc027f5ed5d6730a1097a9344ee18384bd43b415bac926b80364fb72`.
- Manifest SHA-256: `44e3c119fe1643c3f13b531c9724185ab0d1ecc3f4c8eedc550847599d95610f`.

## Caveats and provenance

No training, CUDA compilation, GPU qualification, cloud resources or external
upload was performed. CPU checks and unchanged CUDA arithmetic do not establish
GPU equivalence of the refactored portable assembly. The 31M gate-aware wrapper
supports Base/T2 with absent a/m gates and T7 with the existing a/m gates.

Historical initialization tensors, final/recovery checkpoints and token caches
remain external. No public checkpoint URL is available in this snapshot;
therefore numerical review is immediate, while exact historical weight replay
requires those artifacts. Fresh initialization is an explicitly new replication.
Historical 14M/70M sources remain compact reductions rather than a full raw-event
archive. No 31M causal skip ablation or 410M latency is claimed.

Export sources: `01_build.py`, `release_results.py`, `release_kernels.py`,
`release_supplement.py`, and `package/`. Checks: `02_verify.py`,
`package/scripts/verify_scale.py`, `package/tests/test_companion.py`. ZIP source:
`03_zip.py`. Detailed private provenance remains in `source-map.json`; public
hashes are in the package's PROVENANCE/MANIFEST files. Verification records are
`verification.json` and `zip-verification.json`.
