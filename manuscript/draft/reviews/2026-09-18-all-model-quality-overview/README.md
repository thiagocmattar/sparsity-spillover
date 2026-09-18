# All-model quality-sparsity overview and Section 4.1

## Request and result

The author requested a rewrite of **Quality--sparsity operating regimes**,
centered on the increased attainable sparsity and its quality cost, with an
overview of every executed paper recipe at 14M, 70M and 410M. The new
[three-panel figure](../../figures/11-all-model-quality-sparsity.pdf) is
Figure 3 on page 6 of the rebuilt 35-page [main.pdf](../../main.pdf).
Section 4.1 is on page 5. Figure 1 is unchanged.

The figure uses the reference's pressure colors/styles, circular markers,
ceiling guides and annotated control-clipping paths. The shared legend adds
all available no-pressure and historical single-site pressure recipes, with
naive L1 distinguished from OL1. Missing 70M/410M recipes are not fabricated.
The common Y scale exposes the cross-size loss ordering; all 74 trained
points are in range. The caption discloses the high-loss clipping tails.

## Evidence and scientific wording

The [analysis observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/028-all-model-quality-sparsity.md)
contains the question, method, coverage, caption, numerical results,
interpretation limits and reproduction commands. Its
[builder](../../../../analyses/024-2026-09-17-h-only-kernel-latency/20_plot_quality_all_sizes.py)
reconciles all 74 endpoints to retained metrics, pooled integer counts and
checkpoint manifests. The [evidence copy](../../supplementary-data/all-model-quality-sparsity.json)
contains all coordinates and source hashes. Figure and evidence copies are
byte-identical to the analysis outputs, as recorded in `SOURCES.json`.

All trained losses use ordinary final-checkpoint evaluation, preserving the
existing 62 14M/70M points exactly and applying the same convention to 410M.
The appendices explain the small pass differences from historical tables.
All sixty fixed-control clipping settings remain available in the evidence;
8/5/7 are above the focused Y range at 14M/70M/410M.

The rewritten subsection quantifies the T7/Pall high-threshold sparsity,
fractions of ceiling and loss cost, and gives a moderate-threshold example.
It states the common **712 optimizer steps and 1.493B input tokens**, then
explains the lower 410M learning-rate amplitude and tokens per parameter as
possible contributors to non-monotonic base-model loss. It does not claim
that every 410M condition is worse than 70M, or that those causes are isolated.
The limited 410M sweep and focus on detailed 14M/70M analysis are explicit.

The 14M/70M transfer statement uses the four matched pressured recipes at
kappa=.5 and their own-size optimized-base latency denominators. It reports
1.37-1.42x versus 1.89-2.05x and is scoped to the measured models/kernels,
without a general scaling-law claim. A duplicate optimization-budget
paragraph in Section 4.5 is replaced by a reference back to Section 4.1.
Appendix C.4 remains the realized training protocol; the baseline optimization
plot is now Figure 9 on page 18 (formerly Figure 8), resolved automatically.

## Verification

- Four focused tests pass: complete cohort/source reconciliation, independent
  pressure methods and recipe coverage, clipping/checkpoint/ceiling checks,
  source and output hashes, and speedup denominators.
- BibTeX and three pdflatex passes complete in an isolated build directory.
  All references and citations resolve, and there are no overfull boxes.
  Five underfull vertical boxes and one underfull horizontal box remain;
  their rendered layout was checked. No prose or figures overlap margins.
- The standalone PDF and manuscript pages 5-6 were inspected at high
  resolution. Contact sheets cover all 35 pages and show no layout defects.
- The original Figure08 artwork and all existing manuscript figure copies
  remain unchanged. Existing analysis PDFs were preserved by the generator.
  Concurrent work on Analysis024 Figure10 is outside this revision.
- No training, checkpoint reevaluation, kernel benchmarking or cloud launch
  ran. Build intermediates and raster QA files remain ignored under `tmp/pdfs/`.

[verification.json](verification.json) records the checked artifact/source
hashes, build result, label locations, cohort sizes and test scope.
