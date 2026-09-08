# Results figures, captions and table plan

8 September 2026. Editorial plan requested after selection of the four final
training-results figures. This plans insertion into the current draft; the
approved artwork and current TeX remain unchanged. The canonical captions are
in [Analysis 018 CAPTIONS.md](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md).
A byte-identical copy of this plan is retained there as `MANUSCRIPT-PLAN.md`,
preserving the draft's local-only Git policy.

## Argument and placement

The four selected figures are sufficient for the training-results story.
Use **overview -> paired effects -> distribution reshaping -> transfer across
sizes**. Discussing distributions before transfer shows what changes in the
activations before asking which recipe relationship survives scale. File
numbers remain unchanged.

Insert four short results subsections in
[experimental-study.tex](../../manuscript/draft/experimental-study.tex), after
the paragraph beginning "Within each size, matched contrasts" and before
`\input{kernel-autoresearch}`. Each subsection should have an opening question,
an evidence paragraph and an implication. Keep definitions and run settings
in the existing methods/setup sections and their appendix.

| Proposed subsection | Analysis figure | Figure goal | Proposed label |
| --- | --- | --- | --- |
| Quality-sparsity trade-offs | 01: overview | Establish the regimes occupied by training interventions and A0 clipping. | `fig:quality-sparsity-overview` |
| Paired intervention effects | 02: blocked effects | Identify what changes when each intervention is added to its matched reference. | `fig:paired-intervention-effects` |
| Reshaping activation distributions | 05-v3: signed density grid | Show the distribution changes, including a response outside A4's intervention set. | `fig:activation-reshaping` |
| Transfer across model sizes | 03: scale transfer | Establish which recipe relationship persists at 70M/410M, with architectural reach accounted for. | `fig:scale-transfer` |

For a compact main paper, move the existing full architecture/ladder figure
to the experimental appendix and retain the recipe definitions in the setup
prose. Then the selected results become paper Figures 1-4. Let LaTeX number
them by placement; update the setup/introduction references to the ladder's
appendix location. This is a recommendation, not an already performed move.

Keep the authorized kernel subsection after these results as a separate
execution case study; its figure is additional to the four training figures.
Its historical 35-checkpoint cohort differs from the corrected 30-checkpoint
14M training cohort. Keep that boundary explicit rather than silently replacing
its figure/statistics with Analysis 018 Figure 07's 30-checkpoint summary.

## Figure goals and captions

### Overview: describe the evaluated trade-off

Use [caption 01](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md#figure-01-quality-sparsity-overview)
with `../../analyses/018-2026-09-08-results-materials/figures/01-14m-overview.pdf`.

Lead with the low-loss local-pressure regime and the higher-sparsity broader
recipes. A7-OL1 at kappa=0.5 reaches 27.4827% model-wide sparsity at loss
5.8294, trading quality relative to A0 for more zeros. The dashed A0 path
shows what evaluation-only clipping achieves without retraining. End by
asking which additions explain the differences.

There are 16 trained endpoints and a ten-target A0 clipping path, of which
six points fall inside the displayed loss range. This selected-family view
does not display the entire cohort or a fitted Pareto envelope; connections
do not establish attainable intermediate settings. Source: Analysis 018 O001.

### Paired effects: pressure's value depends on the recipe and threshold

Use [caption 02](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md#figure-02-paired-intervention-effects)
with `../../analyses/018-2026-09-08-results-materials/figures/02-blocked-intervention-effects.pdf`.

Read loss and sparsity deltas together. At kappa=0.5, adding OL1 to A4 adds
2.498 sparsity percentage points for +0.3783 loss; adding it to A7 adds
12.096 points for +0.1265 loss. At kappa=0 and 0.01, A4's pressure addition
improves both axes. These show a conditional response, not uniform benefit.

All 29 reference pairs remain explicit. Within-topology pressure additions
are matched contrasts; comparing those responses across A4 and A7 also changes
pressure sites and normalization, so it does not isolate gate-by-pressure
interaction at a fixed objective. At kappa=0, A4-to-A7's added symmetric gates
are identities; its small numerical residual is not a distinct learned gate
effect. Source: O002 and `tables/pressure-and-gate-effects.md`.

### Distributions: reshaping includes a nonlocal A4 response

Use [caption 05-v3](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md#figure-05-v3-activation-distribution-reshaping)
with `../../analyses/018-2026-09-08-results-materials/figures/05-v3-activation-density-grid.pdf`.

The complete recipes reshape both FFN and attention distributions. For
A4-OL1, q/k/v are outside the gate and pressure sets, so their changed
distributions show a nonlocal response consistent with spillover from the
combined intervention. A7-OL1 directly gates and pressures those same sites;
their changes are not spillover to untargeted q/k/v. These comparisons alone
do not isolate pressure's causal contribution or its route through the model.
Avoid reviving the older directional claim that attention must broaden or
lose near-zero mass: the high-threshold A4 attention peak here narrows.

Keep one exact-zero comparison in the text because the point masses are
deliberately absent from the curves: at kappa=0.5, A4-OL1 versus A7-OL1 has
99.31% versus 93.64% FFN zeros, but 0.22% versus 95.60% attention zeros.
A narrow nonzero peak differs from an exact-zero mass. The symlog density
axis means visual filled area is not a probability readout. Put every
checkpoint's zero masses and tail coverage in the appendix. This figure
provides no cross-size distribution replication. Source: O011 and Run 031.

### Transfer: state exactly which effect survives

Use [caption 03](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md#figure-03-transfer-across-model-sizes)
with `../../analyses/018-2026-09-08-results-materials/figures/03-scale-transfer-and-ceilings.pdf`.

Primary statement: **At kappa=0.5, A7-OL1 has lower validation loss and greater
model-wide sparsity than A4-OL1 at 14M, 70M and 410M.** At kappa=0, A4-OL1
is better on both axes at 14M/70M, whereas A7-OL1 is better at 410M. Thus the
high-threshold relationship survives, but the entire response is not invariant.

Explain the common A7 denominator in the bottom row: it removes the changing
dense-head share but retains architecture-dependent weights among block
operations. It does not make normalized sparsity an equal-quality or
equal-runtime comparison. Larger cohorts lack no-pressure A4/A7 controls,
so they cannot replicate the isolated OL1 additions in Figure 02. This is
one-seed, fixed-token recipe transfer with a different 410M peak learning
rate, not a scaling law or a demonstrated transfer of the distribution
mechanism. Source: O003.

## One compact main-text table

Place a six-row, four-column table after the scale discussion. It adds
readable effect sizes and the low-threshold exception, useful because the
full clipping-loss range compresses trained differences in Figure 03.
Proposed label: `tab:cross-scale-recipe-contrasts`. Show both boundary
thresholds of the measured grid; neither is claimed to be optimal.

| Model | kappa | Delta validation loss | Delta model-wide sparsity (pp) |
| --- | ---: | ---: | ---: |
| 14M | 0 | +0.0219 | -0.756 |
| 14M | 0.5 | -0.2086 | +14.769 |
| 70M | 0 | +0.1358 | -2.110 |
| 70M | 0.5 | -0.1736 | +5.006 |
| 410M | 0 | -0.2658 | +2.783 |
| 410M | 0.5 | -0.0703 | +9.024 |

**Proposed caption: Complete-recipe contrasts at the boundary thresholds.**
Entries are A7-OL1 minus A4-OL1 at the same size and trained threshold.
Negative loss changes and positive sparsity changes favor A7-OL1. The
high-threshold advantage is observed at every size; the zero-threshold
ordering differs. These are matched one-seed point estimates over complete
validation, not averages of independent runs. Full endpoints and all five
thresholds are reported in the appendix.

Values are rounded from
[scale-paired-recipes.md](../../analyses/018-2026-09-08-results-materials/tables/scale-paired-recipes.md).
Generate future TeX values from unrounded `figure_data.json`. Do not import
that table's recipe-specific utilization delta under Figure 03's common-A7
label. If space is tight, move this table to the appendix and retain its key
comparisons in the prose; avoid several redundant main-text result tables.

## Appendix and complete results release

Extend [experimental-appendix.tex](../../manuscript/draft/experimental-appendix.tex)
after the existing evaluation-scope subsection. Put the complete endpoint
and paired results in the PDF, with larger collections also supplied as data.

| Appendix content | Coverage and source | Purpose |
| --- | --- | --- |
| Training protocol and coverage | Analysis 018 `tables/training-protocol.md`; 54 conditions, split 30/12/12 | Document seed, matched initialization/order, 712 updates, realized 1,493,172,224 input tokens, optimizer, precision and learning rates. Distinguish lambda from kappa. |
| All trained endpoints | `tables/all-trained-endpoints.md`; 54 rows split by size | Preserve controls, no-pressure and non-selected/dominated points; show loss and model-wide sparsity, labeling any recipe-specific ceiling. |
| All paired effects | `tables/blocked-effects.md` (29 contrasts), `tables/scale-paired-recipes.md` (15 pairs) | Give exact references and all thresholds behind the main-text views. |
| Exact-zero mass and distribution coverage | Run 031's seven checkpoint histograms and Analysis 018 `activation-density-v3-data.json`; 14 unique pooled groups | Give group counts, zero fractions and out-of-view/tail mass omitted from the density curves; retain per-layer/site counts in the release. |
| Operation accounting and reach | Figure 04/O004, `tables/operation-counts.md`, `tables/architecture-counts.md`, and the architecture/ladder artifact | Explain the numerator and changing workload shares; Figure 08 is optional appendix context, not another main-text ceiling plot. |
| Complete post-hoc trajectories | Run 030's full-range `14m-posthoc-frontiers.pdf`, `70m-posthoc-frontiers.pdf`, `410m-posthoc-frontiers.pdf` and observations | Show every trained model's clipping trajectory, including poor-loss and overlapping settings; all values remain in the supplement. |

The machine-readable supplement contains Run 030's **540 clipping evaluations**
(300/120/120): `clipping-points.csv`, count-preserving `clipping-points.json`,
`frontiers.json` and raw measurements. Include Run 031's seven compressed
signed histograms, manifests and verification records, plus a field dictionary
and the operational `R_model` fraction to paper `S_model` percentage crosswalk.
There is no need to print 540 near-repetitive clipping rows in the main paper.
The appendix should link the release and explain that it includes
training-plus-clipping combinations omitted from the selected main figures.

Do not call Analysis 018's legacy `all-clipping-points.md` (190 points) or
Figure 06 (150 points from 15 checkpoints) the complete collection. Run 030
is the complete source. Its original closeout description of Figure 01 as
showing all 300 clipping points predates the user's A0-only selection;
Analysis 018 O001 and the current plotting code define the display.
The old Figure 05 and 05-v2 are retained alternatives, not additional paper
figures. No new measurement or figure redesign is needed for this plan.

## Insertion and layout checks

Use the existing analysis PDFs directly with `width=\linewidth`, preserving
titles, fonts and legends. Give each figure a nearby first reference and
the linked caption. Keep the four figures at readable paper width instead
of shrinking them into tiny composites. The density caption must retain
zero-mass exclusion, normalization and symlog details despite the removal
of the panel/footer annotations.

After insertion, compile in the actual submission template and inspect float
placement, labels and caption separation. The current `main.tex` is a reading
wrapper; its page count cannot establish submission-template fit. Do not
add significance symbols or seed confidence intervals to one-seed contrasts.
Link result-bearing TeX to its observation and generating script in source
comments. For this planning change, verify caption/figure correspondence,
table values, coverage and source links; keep the current TeX and reading PDF
in place until the planned insertion is performed.
