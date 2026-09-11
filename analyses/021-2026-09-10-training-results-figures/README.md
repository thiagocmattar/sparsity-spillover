# Analysis 021 - Training-results figures

Rework the manuscript's figures from retained evidence, starting with an
introduction-facing Pythia-14M quality-sparsity overview. The author approved
this new analysis and the first figure on 10 September 2026. No new training,
checkpoint evaluation, or measurement is required.

## K050 investigation

The author's follow-up [investigation](investigation/README.md) reconciles all
30 K050 checkpoints with operation accounting, projection/attention counters
and raw matched timing ablations. It finds that seven-site's larger relative
speedup accompanies a slower native reference, while matched four-site OL1
bypasses more projection MMAs and has lower absolute K050 latency. The folder
contains the 30-row CSV, code trace, all ten matched pairs, stratified fits and
a five-page diagnostic PDF. This remains analysis-only; no new runs or
manuscript changes were made. See its [observation](investigation/observations/O001-kernel-investigation.md).

## Matched Q/K/V thresholding table

On 11 September, the author requested the pressure-free A7-minus-A4 comparison
in the manuscript. [O004](observations/O004-qkv-thresholding.md) records the five
matched thresholds and evidence checks. Section 4.3 and Table 2 now report the
raw endpoints, with the high-threshold difference in prose: +5.17 pp model-wide
sparsity for +0.043 validation loss. The table uses the existing Analysis 018
reduction; no new measurement or figure is added. The 27-page draft was rebuilt
and the affected pages visually checked.

## Figure 01

[Quality-sparsity trade-offs](figures/01-14m-quality-sparsity.pdf) retains
26 trained checkpoints, omits naive L1, and shows both baseline and ReLU
post-hoc clipping. Direct annotations highlight the local-pressure regime and
the matched high-threshold pressure addition. Vertical guides expose the
12.83% and 29.95% model-wide sparsity ceilings.

The compact revision is 5.5 by 3.35 inches (1.64:1), 34% shorter than the first
version. The one-row legend sits below the axes, preserving the preceding
version's data-region height. Inside the plot, labels identify baseline, ReLU,
ReLU plus pressure, the two ceiling guides and the 27.48% endpoint. A single
post-hoc clipping label sits near the upper gray path, and the orange statement
reports the matched pressure effect. The x-axis uses the manuscript's
calligraphic S_model notation. There is no internal title, comparison arrow
or utilization callout. Only the local pressure label retains a leader line.
Four-site thresholds act at a,m,h,z; seven-site thresholds add Q/K/V.
The caption defines these sites. Solid, dashed and dotted paths respectively
denote training without pressure, training with pressure and post-hoc clipping.
"Pressure" means orthogonal L1 throughout this selected view. Stronger markers
emphasize measured checkpoints over their connecting lines.

The orange annotation reports the matched seven-site pressure addition at
threshold 0.5: +12.1 percentage points of sparsity for +0.13 loss. The caption
identifies the gray/green post-hoc sweeps and reports that the 27.48% endpoint
uses 91.8% of seven-site reach.

Sources are the tracked Analysis 018 `figure_data.json` and Run 030
`results/clipping-points.json`. The compact
[figure data](data/14m-quality-sparsity.json) retain all 46 selected records,
pooled integer counters, coverage, recipe identifiers, and source hashes.
Eight clipping evaluations lie above the displayed loss range; their measured
coordinates remain in the figure paths and data. Earlier analysis artwork and
the manuscript's full recipe-level PDF are preserved.

After approving the figure polish, the author requested its adoption in the
introduction. The draft now embeds the approved PDF byte-for-byte as Figure 1
on page 2. The Pythia paragraph cites it, and the experimental section refers
back to the same figure. Its caption identifies both clipping paths and the
91.8% utilization. The rebuilt `main.pdf` has 26 pages. The earlier figure
adoption remains in commit `56d4ec6`.

## Reproduction and checks

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/01_quality_sparsity.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_overview.py -q
```

The focused checks cover exact source selection and coordinates, complete
validation, integer pooling, both unclipped p=0 measurements, and independent
ceiling accounting. All three checks passed. All 46 measurements are unchanged
from the preceding version. The compact PDF was rendered and visually checked;
all fonts are embedded. The annotation values, page dimensions, text bounds
and placement of the legend outside the axes were also verified.
See [O001](observations/O001-14m-quality-sparsity.md) for the caption,
results and interpretation limits.

## Figure 02

**Frozen by the author on 11 September 2026.** Final edits clarify the percent
axis, shorten the negative-dot annotation, and define both panels' summaries
in the caption. Layout and data remain unchanged from `444ad84`.

[OL1 conflict geometry and target-set-dependent saturation](figures/02-14m-ol1-geometry.pdf)
is a compact two-panel mechanism diagnostic, requested on 10 September 2026.
It uses every optimizer-boundary record from the five corrected four-site
Run 015 conditions and five seven-site Run 014 conditions: 7,120 observations,
with lambda = budget = 1. The 11 September revision makes the task-relative
opposing component rho_opp the primary diagnostic. Panel (a) shows its median
and interquartile range over 712 steps at each categorical threshold, multiplied
by 100 and plotted on a log percent axis. Blue circles and orange diamonds
distinguish the directly labeled target sets. Panel (b) shows five very faint
traces and one unsmoothed median per target set in the same colors. Direct
labels report each cap-active fraction, with "cap binds" at r/b=1.
There is no pooled trajectory, pooled cap annotation, legend or super-title.
No checkpoints or retrospective gradients are used.

Conflict occurs on 99.9% of steps in each group. Seven-site median cosines are
less negative at every matched threshold, while the cap is active on 97.0%
of seven-site steps versus 0.5% of four-site steps. The directions are nearly
orthogonal in magnitude but systematically negatively aligned; cosine alone
does not establish weak practical conflict. Saturation's mathematical independence
from lambda remains conditional on the cap binding. Post-projection
orthogonality is retained in O002 as an implementation check.

On 11 September, the author approved manuscript adoption. The frozen PDF
is copied without alteration into `manuscript/draft/figures/` and appears as
Figure 4 on page 7. Its caption defines the 25th-75th percentile whiskers,
per-step medians and implemented cap condition. The methods, Section 4.2,
Appendix A.2 and discussion now distinguish the target-set regimes instead
of treating lambda = 1 as universally saturated. The rebuilt draft has
26 pages; O002 records the adoption and verification.

The rho_opp diagnostic uses the retained pre-projection dot product
and squared task norm to measure the raw component removed along the task
direction, relative to its norm. Pooled medians are 0.009503 at four sites and
0.588006 at seven sites (about 62-fold separation). Seven-site medians are
higher at every threshold, and rho_opp exceeds one on 21.15% of seven-site
steps versus none at four sites. These are pre-cap, pre-learning-rate quantities,
not the magnitude of the applied update or measured loss effects.
[O002](observations/O002-ol1-geometry.md) gives the per-threshold medians and
IQRs. Across thresholds, medians span 0.25-1.21% of norm(u) at four sites and
19.78-84.78% at seven sites. The upper y-limit allows IQRs above 100% to remain
visible. No cosine or pooled fold-ratio annotation is plotted. The former
pooled ECDF figure remains in rollback commit `b2f46d0`; the per-threshold
cosine figure is preserved in `c2a2780`.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/02_ol1_geometry.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_ol1_geometry.py -q
```

All ten source event logs and manifests are tracked. The script verifies their
complete step coverage, the logged stabilized geometry, cap scaling, pressure
sites, and historical optimizer/pressure code hashes before plotting.
[The summary](data/14m-ol1-geometry.json) retains exact counts, per-condition
statistics and source hashes; the original logs retain each plotted scalar.
Five focused checks cover coverage and pooling, the stabilized cap boundary,
rejection of inconsistent geometry, the opposing-component formula against a
vector projection, percent conversion and full per-threshold IQRs, and exact
per-family traces. All eight Analysis 021 tests passed. The 6.4-by-2.8-inch PDF
uses the overview's fonts, font sizes and blue/orange colors. It was rendered and
visually checked; all text is inside the page and all fonts are embedded.
See [O002](observations/O002-ol1-geometry.md) for the caption and limitations.

## Figure 03

**Frozen by the author after the final decluttering pass on 11 September 2026.**

[Paired pressure additions](figures/03-14m-paired-interventions.pdf) answers
one question: at a fixed threshold, what changes when OL1 is added, and how
does that differ between four-site and seven-site recipes? The author initially
requested an analysis-only redesign on 11 September 2026, then approved its
manuscript adoption. The frozen PDF is copied without alteration as Figure 3
on page 6 of the 26-page draft. Section 4.2 now focuses on the marginal effect
of adding pressure; O003 records the caption, narrative changes and verification.

The 5.5-by-3.5-inch PDF contains two stacked panels sharing five equally
spaced threshold positions, with ticks on both panels and one bottom x-axis
title. A light grid and dotted zero references aid reading. Short y-axis
labels give delta validation loss and delta S_model (pp); a small shared
4-site / 7-site legend below the panels identifies both curves.
Both show the same ten matched pairs:
A4-OL1 minus A4 and A7-OL1 minus A7. Blue diamonds and orange triangles match
the quality-sparsity overview. Top: validation-loss change; bottom: model-wide
sparsity change in percentage points. Only the two complete effects at
kappa = 0.5 are annotated: +2.50 pp / +0.38 loss and +12.10 pp / +0.13 loss.
Each callout stacks loss below sparsity, leaving more horizontal space for
the data within the same figure dimensions.
A subtitle above the panels defines "Delta = (+OL1) - (no pressure), at matched
kappa". Both panel titles explicitly name the effect of adding OL1. Direction
labels, other intervention contrasts and confidence bands are omitted.

The source is Analysis 018 `figure_data.json`. The
[reduction](data/14m-paired-interventions.json) retains all 20 endpoint identities,
pooled integer counters and ten full-precision differences. It verifies matched
initialization, data order, training settings and validation cache, and reconciles
the differences against the existing contrast records. Three focused tests pass;
the PDF was rendered and visually checked, with embedded fonts and no clipped text.
See [O003](observations/O003-paired-interventions.md) for the caption, exact
results and interpretation limits.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/03_paired_interventions.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_paired_interventions.py -q
```

## Figure 04

[Activation distributions and operation accounting](figures/04-14m-distributions-and-operations.pdf)
combines the two explanations at Pythia-14M, kappa = 0.5. Two density axes show
FFN and attention nonzero shapes, with exact-zero percentages labeled directly.
Two stacked bars show the six measured operation contributions and totals.
A4-OL1 has more FFN zeros (99.31% versus 93.64%) but less model-wide sparsity
(12.71% versus 27.48%); QK/PV supply 0.06 versus 16.77 pp.

The author initially requested this analysis-only figure on 11 September,
then approved manuscript adoption. It is now Figure 5, with Section 4.4
"Local sparsity does not determine model-wide sparsity." The full threshold
density grid is in Appendix D.1; three-size operation accounting remains in
Appendix D.2. The approved artwork is copied byte-for-byte. The figure
is 7.1 by 3.35 inches, without density fills, a baseline curve or a super-title.
Blue/orange identify the recipes in the densities and bar labels; stack colors
identify operations using the existing accounting palette.

The final decluttering uses "4-site + OL1" / "7-site + OL1" throughout the visible
labels and names panel (b) "Decomposed model-wide sparsity." Zero-mass labels
read "zeros" and sit below each density plot; the subplot titles have more space
below the main headings. All percentage annotations use
one decimal place. Bar labels read 12.7% and 27.5%, and the QK/PV note sits above
the four-site bar near 20%. The extra takeaway sentence is removed. Density
plots have faint horizontal and vertical major-grid lines; bars have horizontal
guides. The legend remains three columns by two rows.
The plots use the freed top space. Figure dimensions and measurements are unchanged.

The sources are Run 031 signed histograms and Analysis 018 integer operation
counts from the same two trained checkpoints, covering complete validation.
[O005](observations/O005-distributions-and-operations.md) records the caption,
pooling, separate evaluation-pass reconciliation, results and limitations.
[The reduction](data/14m-distributions-and-operations.json) retains source hashes,
display counts and complete operation counters.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/04_distributions_and_operations.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_distributions_and_operations.py -q
```

All three focused tests pass, covering density mass conservation, the common
operation denominator and exact retained evidence. The PDF was rendered and
visually checked; labels fit inside the page and fonts are embedded.

## Figure 05

[Across-size quality costs and ceilings](figures/05-scale-transfer.pdf) uses
one horizontal row for 14M, 70M and 410M. Every panel plots model-wide sparsity
against absolute validation loss, with the same loss range [4.0, 6.4] and
size-specific sparsity ranges. Blue four-site and orange seven-site OL1 curves
retain all five trained thresholds; only kappa = 0 and 0.5 are labeled.
At 410M, only the seven-site kappa = 0 label remains to reduce annotation density.
Four-/seven-site ceiling guides and seven-site endpoint utilization annotations
replace the previous second normalized row. Compact percentages identify the
ceiling guides; endpoint annotations give only the fraction of the seven-site
ceiling used. The table is omitted from the PDF. A0 clipping remains a faint
dotted reference, clipped to the displayed loss range without dropping its
retained off-scale coordinates. The shared legend sits below the three panels.
The main figure is frozen after the author's final spacing/annotation polish.
Utilization labels use two centered lines with white backgrounds; threshold
labels also have white backgrounds to remain legible over lines. The 70M
high-threshold labels and utilization note have the author's adjusted positions.

The author approved manuscript adoption after the analysis-only polish.
The unchanged artwork is now Figure 6 on page 9, supporting Section 4.5,
"Transfer across model sizes." Full clipping trajectories remain in the
appendix. The text removes the old normalized-row explanation and retains
the common-reference ratio in the appendix endpoint table. At kappa = 0.5,
seven-site OL1 has more sparsity and lower loss than four-site OL1 at all three
sizes. Its endpoints use 91.8%, 82.2% and 92.4% of the respective seven-site
ceilings, at total loss costs +0.621, +1.116 and +0.573 relative to A0.

[O006](observations/O006-scale-transfer.md) records the caption, provenance,
coverage, normalization and limitations. The
[reduction](data/scale-transfer.json) retains the 30 trained endpoints,
30 clipping evaluations and three A0 references from Analysis 018.
The [companion table](tables/scale-transfer-endpoints.md) remains available
separately as Markdown. Three focused tests pass; the 7.4-by-2.95-inch PDF was
rendered and visually checked.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/05_scale_transfer.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_scale_transfer.py -q
```

## Figure 06

[A0 optimization across sizes](figures/06-a0-optimization.pdf) is a two-panel
appendix diagnostic: training loss and pre-clipping global task-gradient norm
versus the same token budget. Each of the three sizes contributes all 712
logged steps, ending at 1.493 billion input tokens. Both panels show faint raw
traces and thicker centered nine-step moving means, with one shared size legend.
The norm panel uses a log scale and explicitly labels the pre-clipping global
task-gradient norm. Short panel titles and very faint raw traces keep the
smoothed curves dominant. These are global norms after loss unscaling,
before the common norm-1 clip, rather than norms of AdamW update directions.

The unchanged figure is now manuscript Figure 8 on page 18 in Appendix C.4,
following the realized-protocol table. Its caption specifies the smoothing;
the surrounding text, transfer section and Discussion use the trajectories
to describe the realized optimization regimes, without a convergence claim.
[O007](observations/O007-a0-optimization.md) records the caption, source
definitions, smoothing and limitations.
The [reduction](data/a0-optimization.json) preserves all 2,136 records, source
hashes and plotted smoothing values. Three focused tests pass; the
7.4-by-2.95-inch PDF was rendered and visually checked.

Both adopted PDFs are byte-identical to their Analysis 021 sources, with hashes
in the draft's `figures/SOURCES.json`. The draft compiles to 28 pages with
resolved references and no overfull boxes; affected pages were visually checked.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/06_a0_optimization.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_a0_optimization.py -q
```

## Figure 07

[K050 speedup and projection bypass](figures/07-kernel-realization.pdf) connects
native-relative full-model speedup to S_model, then the speedup from projection
skipping to projection MMA bypass. Both panels retain the same 30 checkpoints:
one black baseline square (A0), nine gray 1-site circles (ReLU and local L1/OL1),
ten blue four-site diamonds and ten orange seven-site triangles.
Descriptive OLS fits have R² = 0.817 and 0.946.
The highest-sparsity seven-site + OL1 endpoint is labeled 1.78×; dotted
references mark 1×. One compact shared legend sits below the 55:45 panels.
The gap is tighter, both titles fit on one line,
and panel (a) annotations occupy open space away from the observations and fit.
The orange endpoint annotation is raised, and panel (b)'s reference-line note
is removed while its dotted 1× line remains.

The right panel replaces the previous ablation lollipop using the
[investigation](investigation/README.md). Its gain is the raw geometric-mean
all-skips-off latency divided by attention-dense/projection-on latency.
Its MMA bypass fraction includes hybrid SIMT substitution and padded h/z
instruction work. The extra axis note is removed, and panel (a) uses the short
label "Full-model speedup (×)". Panel (b) now says "Speedup from projection
skipping (×)": a full-model latency ratio with attention skipping disabled
in both controls. These definitions and the
separate 338-block counter/64-input timing coverage
are defined in [O008](observations/O008-kernel-realization.md).

This figure remains **analysis-only**. The [reduction](data/kernel-realization.json)
retains source hashes, 90 qualified implementation/checkpoint comparisons,
30 projection points with pooled counters and raw latency denominators, and
both fits. Three focused tests pass. The 7.4-by-3.4-inch PDF was rendered and
visually checked; the manuscript is unchanged.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/07_kernel_realization.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_kernel_realization.py -q
```
