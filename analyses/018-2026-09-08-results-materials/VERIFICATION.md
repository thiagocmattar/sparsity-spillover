# Figure 01 training and clipping revision, 8 September 2026

Only Figure 01 was regenerated in this pass; the other seven publication PDF
hashes are unchanged. The single-panel layout now uses a smaller 7.5-pt legend
below the plot, thin grey x/y grid lines, a shorter title and validation-loss
label, dashed pressure curves, and thin post-hoc trajectories. The cutoff
annotation was removed. Every trained marker renders above the clipping paths.

The figure uses 23 series and 180 input coordinates: all 30 trained endpoints
and all 150 retained clipping evaluations, preserving each measured p=0 point.
The quality-focused window includes all trained endpoints and 100 clipping
points. Complete clipping ranges remain in Figure 06. No missing A4-OL1 or A7
clipping trajectory is inferred, and no checkpoint inference was performed.

The regression check inspects actual plotted coordinates and connection order
against the complete source checkpoint/parameter grids, including all 15
clipping sources and the training-above-clipping drawing order. Retained count,
coverage, provenance and serialization checks pass. The PDF was rendered and
inspected at paper width; an independent visual review found no material
cropping, overlap or legibility defect and scored the design 20/25. This is an
editorial score, not a scientific inference. Captions and the series table were
reconciled; temporary proofs remain under ignored tmp/analysis018-small-fixes.

The isolated staged revision passed **39 focused tests in 1.00 second**
(Analysis 018 evidence checks plus ceiling and metric tests). Figure 01 and
its overview-series table reproduce byte-for-byte from the staged source.
Only Figure 01 is staged among the publication PDFs. The optional TeX caption
compiled with resolved references and no LaTeX warnings or overfull/underfull
boxes; its figure/caption page was rendered and inspected. This staged proof
is under ignored tmp/analysis018-figure01-staged.

# Verification record - overview correction, 8 September 2026

## Overview correction

The previous plot applied strict within-series nondomination before drawing
each line. It retained all 30 trained scatter markers but left 14 outside their
recipe's connecting line. Figure 01 now follows every measured dose in order,
including dominated points, as a dose sweep. A0/A1-H remain single trained
points with separate ten-target clipping curves. The strict numerical Pareto
table is unchanged. The title retains the requested quality-sparsity framing;
the caption explicitly identifies the curves as dose sweeps, not envelopes.

All 30 trained endpoint coordinates were checked directly against their raw
same-pass validation loss and pooled per-operation integer counts. The saved
series now contains all 50 plotted input coordinates (30 trained, 20 clipped).
The regression test inspects the actual Matplotlib line coordinates against
each complete, explicitly enumerated dose grid, including previously omitted
dominated points. It does not merely compare two calls to the same filter.

The focused suite below passed **39 tests in 1.84 seconds** after the correction.
Figure 01 was rendered in color and grayscale at 115.2 dpi and inspected. It is
one page, 396 points wide, with every text span inside the page bounds. Only
Figure 01 changed; the other seven PDF hashes match the preceding commit.
The renamed tables/overview-series.md records all points in connection order;
captions, prose, evidence JSON and inventory hashes were updated together.
The revised optional TeX caption compiled with resolved references and no
LaTeX warnings or overfull/underfull boxes; its figure/caption page was rendered
and inspected. Temporary correction proofs are under ignored
tmp/analysis018-frontier-correction. All local Markdown links resolve.

## Scope and outstanding measurement

Eight figures were revised, titled and reviewed. The 54 trained conditions,
190 clipping evaluations and 30-checkpoint runtime cohort are unchanged.
The overview has ten separately defined dose series. Its complete numerical
membership is stored and tabulated. The activation grid uses seven
checkpoints, five common sites and kappa=0,.05,.5 rows.

**The requested finer activation bins are not complete.** Stored statistics
only define the four bins {0}, (0,.001], (.001,.01] and >.01. No finer-bin
counts were inferred or fabricated. The separate ACTIVATION-DIAGNOSTIC.md
proposal awaits design confirmation under AGENTS.md; no new diagnostic code,
numbered run or model execution has been started. The checklist leaves this
item open. The restored distribution figure is explicitly a coarse view.

## Numerical checks

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

Checks cover complete cohorts and validation,
integer logical counts, count-first activation pooling, magnitude-bin
partitioning, source hashes, same-size identities, exact matched differences,
ceiling units and clipping-site normalization, actual overview sweep coordinates,
all three case-study thresholds, the 30-checkpoint runtime reduction and output
inventory hashes. The saved bundle still identifies 230 direct source files.
No shared scientific code changed. The full bootstrap launch suite is not
applicable until a new diagnostic is implemented for launch.

## Visual and TeX checks

In the preceding revision, all eight PDFs were rendered at 115.2 dpi in color and grayscale and
visually inspected. Each is one page and exactly 396 points (5.5 inches) wide.
Every extracted text span lies inside its page bounds; ordinary text is at
least 8 pt, with conventionally smaller math scripts. Every figure has a title.

The overview uses separate trained-family and control-clipping dose sweeps.
Eight high-loss clipping points lie beyond its
5.04–6.15 loss window; their complete range is visible in the scale figure
and the clipping appendix. The scale ceiling lines use both color and differing
line patterns. Operation fills are solid as requested; their stack order and
legend identify operations. The activation grid uses matched axes, one shared
legend and the original measured bins. The clipping legend states source
checkpoint counts. The kernel plot now includes the full recipe legend.

The preceding revision's optional results.tex and both compact generated tables compiled in the
5.5-inch-text-width article wrapper. The final reference-resolved compile has
**no LaTeX warnings, unresolved references, overfull or underfull boxes**.
All ten preview pages were rendered and inspected as a syntax/caption/layout
proof; the wrapper's float pagination is not a final conference page budget.
The manuscript itself was not edited or rebuilt.

The final scale-line pattern change was rendered and checked separately.
Temporary renders and wrapper files stay under ignored tmp/analysis018-revision2.
Publication artwork remains PDF only. artifact_inventory.json records all
PDF/table hashes and sizes; the analysis .gitattributes preserves those bytes.

## Closeout

The coherent figure revision, updated captions/prose/tables and pending
measurement design are committed together. The open finer-bin item is not
represented as finished. Original runs, prior analyses, model weights, caches,
credentials, temporary outputs and unrelated work remain outside this commit.
