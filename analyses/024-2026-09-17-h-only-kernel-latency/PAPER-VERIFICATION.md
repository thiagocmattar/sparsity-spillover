# Verification of the task.md paper figure rebuild

Completed locally on 18 September 2026. No training, evaluation, benchmarking,
or cloud resources were launched. The approved task is retained as `task.md`.

Figure 3 was subsequently revised at the user's request to show OL1(all) minus
OL1(h) at both 14M and 70M in six panels. Rows distinguish model sizes; columns
show loss, sparsity and full-model latency changes. Two topology curves use Figure 6's blue/orange
colours and uniform circular markers. Its 20 contrasts use 40 unique checkpoints.
The final style matches Figure 6's title/panel/axis/tick/legend font sizes
(14/11.5/11/10/11 points), line width and markers, widened to 12.8 inches for the
third column at the existing 5.8-inch height. Loss/sparsity retain shared scales
across model sizes; latency scales are separate to keep 14M differences legible.
Latency is the difference of retained checkpoint geometric means in µs. The global/local
title, paired-difference panel names and concise axes replace the earlier wording.
The redundant OL1 legend subtitle is removed. The final PDF was visually checked
at 2,000 pixels, with all 20 contrast records unchanged and all fonts embedded.
That revision regenerated only Figure 3 and its provenance/caption.

Figure 4 was then revised to absolute operation contributions for twelve 14M
checkpoints, replacing the eight signed contrasts. Its six bars per threshold
cover T4/T7 with P0/Ph/Pall at κ=.05/.5. All segments use the full-model
denominator and each stack sums to the checkpoint's model-wide sparsity. The
two panels share a 0–30 pp scale and use Figure 3's typography, with the exact
operation palette/order from the manuscript's `05-site-structure.pdf`.
The targeted `--only-operation-contributions` rebuild preserves all other
figure records/PDFs and manuscript files.

## Scientific checks

The additional Figure 8 combines Figure 6's style and 44 trained checkpoints
with the exact 40-point Run036 clipping latency join. Four focused tests in
`test_matched_quality_latency.py` pass: source/output identity, preservation of
the 14M trained coordinates, matching checkpoint/dose and pooled counters,
three-process geometric means, and the declared visible ranges. All clipping
latencies are visible; eight 14M and seven 70M high-loss quality points lie
above their focused quality ranges. Existing figure hashes are preserved.
The PDF was visually reviewed at 2,000 pixels, has one page and three embedded
fonts. See [the Figure 8 observation](observations/025-matched-quality-latency.md).

Seven focused tests pass via:

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s analyses/024-2026-09-17-h-only-kernel-latency -p test_paper_figures.py -v
```

The checks cover:

- All recorded source hashes, builder hashes, current PDF hashes, and the 13
  unchanged archived PDF hashes/byte counts.
- Unique identity joins for 62 checkpoints and explicit primary membership
  (32 at 14M, 22 at 70M); five κ values per available multisite family; common
  initialization and training-order hashes within size.
- Ordinary-final dense-relative loss, full-model integer sparsity denominators,
  optimized-dense speedup normalization, three process means, and 64 timing
  inputs per checkpoint. The equal-sized process geometric means reconcile to
  the reported geometric mean over all 1,344 timings.
- All 20 all-minus-h pairs at fixed size, scope, and threshold: matched
  initialization/schedule, integer-count sparsity differences, latency subtraction
  and unit conversion, identical timing workloads/indices and session provenance.
  Only the 14M T7 timing contrasts span sessions. The 14M contrasts also agree
  for all three metrics with the difference between the retained all-minus-P0
  and h-minus-P0 effects in the pressure-versus-none appendix.
- Twelve operation decompositions: each operation's numerator matches the
  retained counter and uses the full-model denominator. Nonnegative integer
  numerators sum exactly to the checkpoint aggregate; contributions sum to
  model-wide sparsity within 1e-12 percentage points. Recipe membership and
  all twelve unique checkpoint identities are verified.
- Nondominance direction, strict improvement, and preservation of ties.
- All 340 retained clipping records for the declared cohort, complete validation
  coverage, all 54 trained points inside the main range, and exactly seven
  omitted high-loss control clipping records per size.
- Five Table 2 rows with fixed h-only pressure and the exact 30-checkpoint
  historical 14M instruction/ablation cohort.

The builder also verifies 338 sequences, 692,224 input tokens, 693,668 source
tokens, and the 1,444-token excluded tail for the source loss/counter passes.
Every timing record is qualified. Ordinary-final and logical-pass losses are
retained separately; the clipping zero-dose audit is recorded without adjustments.

## Artifact and visual review

All nine publication PDFs contain one page and were rendered with Poppler at
1,500-pixel maximum dimension. Each was visually inspected; the revised quality,
pressure, and operation panels were re-rendered and reviewed after layout changes.
The architecture was compiled from the retained adaptation of the existing
manuscript TikZ diagram. `pdfinfo` confirmed one page per PDF; `pdffonts` confirmed
that all listed fonts are embedded.
Publication outputs are PDF only; temporary preview PNGs are not deliverables.
The six-panel Figure 3 was separately rendered at 2,000 pixels and visually
checked for labels, legend, line styles, and panel spacing. Numeric annotations
and the validation-loss unit suffix are absent; all four listed fonts are embedded.
The absolute-contribution Figure 4 was separately rendered at 1,900 pixels;
all three listed fonts are embedded. Visual inspection confirmed the T/P labels,
palette, shared scale, segment boundaries, title, and common legend.

Each PDF has its own observation/caption file containing the associated proposed
manuscript paragraph and links to the relevant draft section. Table 2 has its own
caption/writing file. [CAPTIONS.md](CAPTIONS.md) is the complete artifact index.
Historical observation links resolve to `.archive`; old source-hash JSON records
remain unchanged. Relative links in the new caption/index files were checked.

## Limits carried into the captions

No independent-seed uncertainty is represented. The primary cohort omits eight
local pressure settings; the instruction appendix intentionally retains them in
its older 30-point cohort. No 70M pressure-free multisite checkpoints or h-only
post-hoc sweeps are invented. The 14M h-only 7-site timing session differs from
the other 14M timings. The 70M implementation is the qualified shape-specific
port, not unchanged 14M K050 or a projection-only ablation. No regression is
extended to new checkpoints, and no 70M attention-skipping profitability claim
is inferred. Proposed manuscript text is recorded beside the figures; draft TeX
is not modified by this task.
