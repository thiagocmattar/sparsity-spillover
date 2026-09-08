# Results figures, captions and table plan

8 September 2026. Editorial plan for five selected results figures: four
training-results figures and Figure 07's kernel case study. This plans insertion into the current draft; the
approved artwork and current TeX remain unchanged. The canonical captions are
in [Analysis 018 CAPTIONS.md](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md).
A byte-identical copy of this plan is retained there as `MANUSCRIPT-PLAN.md`,
preserving the draft's local-only Git policy.

## Argument and placement

The four training figures are followed by the selected kernel realization figure.
Use **overview -> paired effects -> distribution reshaping -> transfer across
sizes -> full-model acceleration**. Discussing distributions before transfer shows what changes in the
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
| Realizing sparsity with specialized inference kernels | 07: kernel realization | Show qualified kernel-search progress and how the final selected kernel's measured speedup relates to model-wide sparsity. | `fig:kernel-autoresearch` (reuse existing label) |

For a compact main paper, move the existing full architecture/ladder figure
to the experimental appendix and retain the recipe definitions in the setup
prose. Then the selected results become paper Figures 1-5. Let LaTeX number
them by placement; update the setup/introduction references to the ladder's
appendix location. This is a recommendation, not an already performed move.

Revise the existing [kernel subsection](../../manuscript/draft/kernel-autoresearch.tex)
after the four training subsections to use the selected Analysis 018 Figure 07.
Replace its older Run 029 figure and caption, rather than adding a second
kernel figure. Update the surrounding cohort statistics together: the selected
figure uses the corrected 30-checkpoint 14M cohort, while the current TeX uses
35 historical checkpoints. The source Run 029 records remain historical.

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

### Kernel realization: connect logical sparsity to measured acceleration

Use [caption 07](../../analyses/018-2026-09-08-results-materials/CAPTIONS.md#figure-07-kernel-realization)
with `../../analyses/018-2026-09-08-results-materials/figures/07-kernel-realization.pdf`.

Question: can specialized kernels exploit the observed zeros to accelerate
the complete forward pass? Panel (a) shows the best numerically qualified
incumbent on fixed A7-OL1 kappa=0.5 across 42 eligible kernel iterations,
reaching 1.7830x. Panel (b) shows the final selected kernel K050 across all
30 included checkpoints: geometric mean speedup 1.2340x, with a descriptive
positive association with model-wide sparsity (OLS R-squared 0.8167).
"Best kernel" refers to the selected search result; it does not assert
optimality over every implementation or ablation.

Keep one paragraph on attribution: fusion and sparse paths both contribute.
Relative to the fused no-skip ablation, enabling sparse paths gives a 1.0432x
geometric ratio and helps 14/30 checkpoints. Disabling attention skipping
improves all 30 (geometric mean speedup 1.2506x), so greater logical attention
opportunity does not guarantee profitable attention skipping at T=2048.
Detailed qualification and ablation tables belong in the appendix.

This is one RTX5090, BF16, batch-one, uncached full-model case study with
full vocabulary logits, using each checkpoint's native SDPA CUDA-graph
baseline. It does not establish equal-quality gains, cached-decoding or
larger-model transfer, a causal speedup law, or superiority of agent-assisted
search over human development. Preserve the existing agent-provenance caveat.
Canonical sparsity remains the FP16 measurement used by the training figures.
Sources: [O007](../../analyses/018-2026-09-08-results-materials/observations/O007-kernel-realization.md),
`figure_data.json:runtime` and `tables/runtime-summary.md` in Analysis 018.

When updating `kernel-autoresearch.tex`, replace 35/35 with 30/30, the 1.25x
cohort mean with 1.2340x, R-squared 0.781 with 0.8167, and the cohort sparse-path
gain of 5.6% on 19/35 with 4.32% on 14/30. Replace the attention-dense cohort
mean of 1.267x with 1.2506x. The fixed-checkpoint result remains separately
scoped. The old caption's gray failed-candidate points, blue incumbent line,
P0 comparator and fitted equation are absent from Figure 07: use the new
caption in full. The main-text table recommendation below remains unchanged.

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
| Kernel qualification and ablations | Analysis 018 O007, `tables/runtime-summary.md` and `figure_data.json:runtime`; corrected 30-checkpoint cohort | Give numerical qualification, timing protocol and sparse-path/attention ablations behind Figure 07; distinguish P0's four qualified checkpoints from cohort-wide comparisons. |

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
the linked caption. Keep the five figures at readable paper width instead
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
