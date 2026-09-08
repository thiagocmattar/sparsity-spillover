# Complete-cohort Figure 01 update, 8 September 2026

Run 030 completed and verified 540 clipping measurements over 54 checkpoints:
300 at 14M, 120 at 70M and 120 at 410M. It adds 350 evaluations to the 190
retained points. Every source has all ten targets and complete 338-block
validation. Raw/JSON/CSV values, checkpoint hashes, integer counts and
thresholds agree exactly. The largest p=0 loss discrepancy is 0.000180712,
below the declared 0.0005 guard. All 444 final transferred files were verified
locally before both Pods were deleted. Zero Pods/endpoints remain; the existing
network volume is preserved. See Run 030's results/verification.json and
results/transfer-receipt.json.

Figure 01 now contains 38 paths and 330 input coordinates: 30 trained endpoints
plus all 300 matching clipping evaluations. Its layout, axes, legend, grid and
styles remain the approved design. The loss window contains all trained
endpoints and 222 clipping evaluations; the full-range 14M PDF is in Run 030.
Numerical frontier membership and the overview-series table use the complete
pool. The original clipping subsets in the other figures remain separate.

The working figure/evidence suite passed **41 tests in 2.44 seconds**. Actual
coordinate tests check every source/target pair, family encoding, order and
training-above-clipping placement. The publication PDF matches the reviewed
proof byte-for-byte (SHA-256 c7f4f47d95399ca81f9571f5add3e6b4e0eba3741e1f3df9dd47e925843f771c).
Independent scientific/design rubrics both score 20/25; details are in
FIGURE-REVIEW.md. The caption proof compiles without warnings or overfull/
underfull boxes and was rendered at 5.5-inch width. The three complete-range
Run 030 PDFs were also rendered and inspected.

A concurrent task committed Figure 02 revisions and its alternative 02-v2
while Run 030 was running. Those committed changes were preserved. This task
regenerated only Figure 01 within Analysis 018; hashes of the eight other
current PDFs were checked before and after the scoped update. The earlier
Figure 06/other readability edits remain outside this commit's scope.
Temporary proofs and checks are under ignored tmp/run030-figure-preview.

The isolated staged revision passed **40 focused tests in 1.81 seconds**.
Figure 01 and both overview/frontier tables reproduce byte-for-byte from the
staged source. Every Run 030 publication output hash matches its staged bytes,
including the complete raw measurement archive.

# Figure 02 intervention and paired rows, 8 September 2026

The horizontal figure now labels its two annotation rows on the left:
Intervention: for action/site descriptions and Paired: for full treatment-minus-
reference names. Each pair wraps onto two lines, with the minus sign preceding
the reference. The metric axes show only delta loss and delta model sparsity
(pp); the former "vs group reference" suffixes, Ref: prefixes and bottom
Intervention axis title are removed. Action and pair text remains 7 pt. O002,
the README and the optional analysis-owned TeX caption describe the new rows.

All 29 paired differences remain unchanged. Direct plot checks verify all 58
effect values, 29 aligned dose labels and seven pair names against the stored
treatment/reference IDs. Both row headings align with their cells. All visible
text lies within the page; annotation rows and adjacent cells do not overlap.
Color and grayscale renders and the combined figure/caption page were visually
inspected. The TeX proof has no warnings or box overflows. The PDF reproduces
byte-for-byte from the isolated source and freshly loaded retained evidence.

The isolated revision passes **40 focused tests in 1.73 seconds** (analysis
evidence, ceilings and metrics). The other eight PDFs retain their exact hashes.
No numerical evidence, model execution or manuscript changed. Unrelated working-
tree edits remain outside the commit. Temporary proofs are under ignored
tmp/analysis018-figure02-paired-rows.

# Figure 02 horizontal polish and explicit references, 8 September 2026

The user selected the original horizontal layout. Dose values and intervention
descriptions are now 7 pt; lambda/kappa labels are 7.5 pt; the shared Intervention
axis title is 8 pt. The actions consistently describe operations and sites,
including h in the four-site gate action and all seven sites in the final OL1
action. A separate Ref: row identifies each comparator. Both y-axes explicitly
say "vs group reference"; the loss axis omits nats/token. O002 and the optional
analysis-owned TeX caption define the reference identities and matching doses.

All 29 paired differences remain unchanged. The raw loss/count audit and direct
plot checks reconcile all 58 effect values, 29 aligned dose labels, seven
reference labels against the actual pair IDs, and same-lambda/same-kappa
matching where displayed. The font sizes and absence of recipe IDs from the
action labels were checked directly. Dose, parameter, action and reference rows
do not overlap; the smallest numeric-label gap is 2.17 pt. All visible text lies
inside the page. Color and grayscale renders and the combined figure/caption
page were inspected at paper width. The TeX proof has no warnings or box
overflows. The PDF reproduces byte-for-byte from the isolated source and freshly
loaded retained evidence.

The isolated revision passes **40 focused tests in 1.03 seconds**. The other
eight PDFs, including the unselected 02-v2 alternative, retain their exact
hashes. No numerical evidence, model execution or manuscript changed. Unrelated
working-tree edits remain outside the commit. Temporary proofs are under
ignored tmp/analysis018-figure02-polish.

# Figure 02-v2 alternative orientation, 8 September 2026

The user requested a separate rotated layout with Intervention on the y-axis
and two columns. Figure 02-v2 uses shared comparison rows, validation-loss
changes on the left and model-sparsity changes on the right. The original
seven-group order runs downward; doses increase downward within each group.
Numeric dose labels sit beside the plots, with separate lambda/kappa group
labels. Horizontal stems retain the same zero reference and effect ranges as
Figure 02. The new PDF is 396 by 396 points (5.5 by 5.5 inches).

All 29 comparisons were checked against their raw same-pass loss and integer
product counts. Direct plot inspection verified all 58 effect coordinates,
identical row positions across columns, the inverted categorical axis, 29
aligned dose labels, two lambda groups and four kappa groups. Numeric labels
do not overlap; all visible text lies within the page. Color and grayscale
renders were inspected at paper width. Fonts are embedded. The PDF reproduces
byte-for-byte from the isolated source and freshly loaded retained evidence.

The isolated revision passes **40 focused tests in 0.95 seconds** (analysis
evidence, ceilings and metrics), including the updated nine-PDF inventory.
All eight pre-existing PDFs retain their exact hashes. The builder produces
both orientations, and O009 records the alternative's caption and limitations.
No numerical evidence, model execution, original figure or manuscript changed.
Unrelated working-tree changes remain outside the commit scope. Temporary
proofs are under ignored tmp/analysis018-figure02-v2.

# Figure 02 intervention-axis revision, 8 September 2026

Figure 02 now places the numeric dose values below the plotted points, followed
by a separate lambda or kappa group label and the intervention name. The shared
categorical x-axis is Intervention. Uniform circles replace the mixed dose key;
the dose-free comparison has a dash. A1-H to A4 and A4 to A7 identify the actual
paired recipes. Both rows retain the same x coordinates and the original y
limits. Group widths provide space for the numeric ticks; spacing is ordinal.

All 29 retained comparisons were independently checked against the raw
same-pass loss and integer product counts. All 58 plotted y coordinates,
29 numeric labels and their x alignment, two lambda groups, four kappa groups,
uniform marker shapes and absence of a dose legend were checked directly.
No numeric labels overlap, and all visible text lies inside the figure bounds.
The PDF reproduces byte-for-byte from the isolated Figure 02 sources. Color
and grayscale renders were inspected at paper width. The captions explain the
fixed OL1 weight/budget and the full A1-H to A4 intervention, including h.
The analysis-owned TeX caption compiled without warnings or box overflows;
its combined figure/caption page was rendered and inspected.

The isolated commit scope passes **40 focused tests in 1.00 second** (analysis
evidence, ceilings and metrics). It includes the previously uncommitted complete
A1-H to A4 threshold rows needed to reproduce the 29-point figure under review.
Other figures, ongoing clipping changes and unrelated working-tree edits are
excluded from this scope. The broader working-tree suite currently reports
26 passes and 15 fixture errors because the in-progress Run 030 integration
references its absent results/clipping-points.json; Figure 02 uses retained
trained comparisons and does not depend on that new clipping file.

Source, data and PDF proofs remain under ignored
tmp/analysis018-figure02-axis-review. No experiment or manuscript was changed.

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
