# Introduction findings and abstract

## Current abstract: author-guided findings and takeaway

The author's next instructions specify P/T at first mention, the 14M/70M model
scope, a comparison of sparsity/ceiling/latency measures, four conclusions with
data, and the exact training/kernel co-design takeaway. Only the abstract is
revised in this pass; the surgical introduction below remains unchanged.

The abstract reports T7/Pall at kappa=0.5: 27.5%/40.6% model-wide sparsity,
92%/82% of the respective ceilings, and 0.62/1.12 additional validation loss
at 14M/70M. The within-recipe execution trend and cross-recipe counterexample
follow [Observation 030](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/030-base-speedup-clipping.md)
and `22_plot_base_speedup_clipping.py`. The 14M T7/Pall and T4/Pall endpoint
latencies are 0.473366 and 0.459476 ms at kappa=0.5 despite greater T7 sparsity.

Direct checks of the existing quality export confirm lower loss and latency
for T4/Ph versus T4/Pall in all eight moderate-threshold pairs. T4/Ph at
kappa=0.05 is nondominated in loss/latency among the measured endpoints at
both sizes. The wording says "some of the best observed" because T7/Ph is
also competitive and the study does not specify a scalar trade-off objective.
The 10.7-point spillover result is specifically T4/Ph minus T4/P0 at kappa=0.05
in 14M, using pooled z exact-zero counts. The 150/29-microsecond kernel effects
are the separate conditional ablations described below, not additive savings.

All displayed numbers were checked against the retained evidence. The source
introduction hash is unchanged. The PDF rebuild has 20 pages, resolved
references/citations, no overfull or underfull vertical boxes, and the same
three underfull horizontal-box warnings. Opening pages were inspected at high
resolution and all pages in contact sheets. Related Work still follows the
introduction on page 2. The installed PDF matches the reviewed build.

## Previous pass: surgical revision

The author's follow-up requests restoring the previous wording and editing it
surgically. This version starts from the saved pre-edit passages, superseding
the compressed rewrite below. The introduction preserves the five observations
in their original order, their explanatory transitions, and the closing
co-design statement verbatim. Changes remove redundant numbering and site
definitions, bold the main clauses, add supporting references and the 16/16
and 18/20 paired results, give a concrete nonlocal example, and qualify the
scale statement by implementation and latency reference.

The abstract preserves the original opening, study description, metric
definitions and concluding sentence. It removes the repeated motivation
sentence and the three-size numerical catalog, updates the pressure comparison
to both sizes, and replaces the older regression statistics with direct kernel
evidence. It remains moderately shorter than the original, with fuller prose
than the superseded rewrite. The source evidence below remains applicable;
the current text does not report every audited value.

Checks confirm that the preceding introduction and its closing sentence are
unchanged from the author's saved version, and that the abstract retains the
specified original sentences. The 16/16 and 18/20 counts were rechecked against
the retained endpoint export. The rebuilt PDF remains 20 pages, with resolved
references/citations, no overfull or underfull vertical boxes, and the same
three underfull horizontal-box warnings. Contact sheets cover all pages;
the first two pages were inspected at higher resolution. The complete findings
paragraph fits below Figure 1 on page 2, followed by Related Work. No layout
settings or other manuscript prose were changed. The installed PDF matches
the reviewed build byte-for-byte.

## Superseded compressed version and evidence audit

The author requested a shorter, evidence-backed closing introduction paragraph,
bold main findings, and a more focused abstract. The revised introduction uses
four findings: the quality cost of high sparsity, local versus broad pressure,
nonlocal sparsity effects, and placement-dependent execution gains. A paragraph
break before the execution finding keeps it intact below Figure 1. All earlier
introduction text and the figure are preserved. The abstract is approximately
half its previous length, with two bold findings and an explicit co-design
takeaway. The one-seed and measured-hardware scope remains visible.

## Evidence

- The 14M high-threshold example comes from
  [Observation 028](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/028-all-model-quality-sparsity.md)
  and `20_plot_quality_all_sizes.py`. The retained
  `data/all-model-quality-sparsity.json` gives 91.754602% of the T7 ceiling
  and a 0.620807 increase in ordinary final validation loss over the base
  model. These round to 92% and 0.62. The analytic ceiling is not a speedup.
- The pressure comparisons follow
  [Observation 015](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/015-pressure-scope-caption.md),
  generated by `paper_effect_figures.py` / `13_rebuild_paper_figures.py`.
  Recomputing the 20 matched pairs from the same quality export confirms
  lower h-only loss in all 16 pairs at kappa <= 0.1 (0.192102--0.324167),
  and lower measured latency in 18 of 20 pairs across the full grid.
  These are descriptive endpoint comparisons, not replicated-seed estimates.
- The nonlocal example follows
  [Observation O016](../../../../analyses/021-2026-09-10-training-results-figures/pressure-scope/observations/O016-pressure-placement.md),
  `evidence.py`, and `plot_layer_sparsity.py`. The retained `data/evidence.json`
  confirms h-only realized pressure and increased pooled z exact-zero mass
  relative to P0 at kappa=0.05 for both T4 and T7 (10.666322 and 12.106378
  percentage points). The text says unpressured; z is still thresholded.
- The direct kernel evidence follows
  [Observation 034](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/034-operation-latency-manuscript.md)
  and `25_build_operation_latency_table.py`. The retained
  `data/operation-latency.json` confirms h/z conditional savings of
  150.069792/28.959677 microseconds and joint attention-product overhead of
  7.041316 microseconds. This is one 14M T7/Pall checkpoint at kappa=0.5;
  the h/z effects are separate ablations and must not be added together.

The summaries omit the previous scale/speedup generalization: its interpretation
depends on the latency reference, as documented in
[Observation 041](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/041-kernel-appendix.md).
They also omit the older 35-checkpoint regression statistics and the abstract's
three-size numerical catalog in favor of direct comparisons and an ablation.
No new experiment, measurement, or finding-registry change was made. The existing
Section 4.1 scale discussion is outside this requested prose edit and remains
unchanged.

## Verification and scope

Direct checks against the retained data confirm every numerical claim above.
A before/after comparison confirms the introduction prefix is unchanged.
The isolated latexmk build produces 20 pages, with resolved references and
citations, no overfull boxes or underfull vertical boxes, and the same three
underfull horizontal-box warnings as before. All pages were visually checked
in contact sheets, with the opening pages checked at higher resolution.
The execution finding is intact on page 2, immediately before Related Work.

The installed `main.pdf` is byte-identical to the reviewed build and includes
the author's pre-existing working-tree prose. Only this new introduction ending,
the abstract, this review, the manuscript README entry and PDF belong to this
change; other author edits remain outside its commit.
