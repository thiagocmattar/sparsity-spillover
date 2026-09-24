# O003: Focused integration of the 14M/31M/70M recipe comparison

## Question and authority

Integrate Figure 01 into the draft as the replacement for the historical
14M/70M Figure 5, following the user's [change plan](../manuscript_changes.md).
Preserve the 14M mechanistic study and distinguish threshold placement from
pressure placement. The user excluded Figure 02 from the paper; it remains
an unmodified historical analysis artifact.

## Method and coverage

`04_manuscript_evidence.py` verifies the retained 31M training manifests,
configs, canonical validation metrics, integer logical-product counts, source
hashes and 33 final timing processes. The 11 checkpoints comprise Base,
five T2/Ph and five T7/Ph conditions at kappa 0, 0.01, 0.05, 0.1 and 0.5.
All share 712 updates, 1,493,172,224 input tokens, seed 1234, initial parameter
hash and training-order hash. Validation covers all 500 documents through
338 complete 2,048-token blocks, excluding the 1,444-token tail.

The pinned 31M architecture has 30,494,720 parameters, six layers, width 256,
FFN width 1,024, eight heads of width 32 and vocabulary 50,304. Its 48.964943
tokens per parameter round to 48.96. The scalar-product denominator is
42,483,056,640 per sequence. Analytic ceilings are 0.00%, 7.58%, 9.48%,
22.75% and 37.92% for T0/T1/T2/T4/T7; these are not observed sparsities.

Each checkpoint has three independent BF16 RTX 5090 timing processes with
full 338-block numerical qualification. Timing uses 64 inputs, seven paired
passes and CUDA graphs, with 448 host samples per backend per process.
Run054 supplies Base/T2 and Run055 supplies T7, on different physical RTX
5090 devices. The fixed 31M native Base denominator is 1.0238611173053367 ms;
its specialized latency is 1.1062549786115434 ms. Canonical FP16 evaluation
loss, not the separate BF16 qualification loss, supplies the quality axis.

## Legend and caption

The paper's Figure 5 is an exact copy of Analysis038 Figure 01. Its approved
title remains **Quality–latency trade-offs across model scales**. Dashed blue
T2/Ph and solid orange T7/Ph both use h-only OL1. Circles, triangles and
squares identify 14M, 31M and 70M; hollow/filled gray markers distinguish
kernel/native Base. The three Base-loss guides now carry size labels placed
near the lower edge, clear of all measurements. Lines connect the five
measured kappa settings in order; the caption and appendix tables give that
order without adding point annotations. Both axes remain logarithmic.

The manuscript retains the old asset filename for stable inclusion and
cross-references. The immutable Analysis034 figure is now the plotter's
style reference, avoiding a circular reference to the manuscript copy.

## Results and implemented changes

All 15 matched-kappa comparisons have lower loss for T2/Ph than T7/Ph.
T7/Ph reaches a lower minimum sweep latency at every size. Only the evaluated
moderate 14M T2 settings improve loss over Base; neither 31M nor 70M does.
The illustrative 31M point uses kappa 0.1 by the stated rule of matching the
main moderate-threshold 14M example: loss 4.757208, latency 0.908418 ms,
versus Base loss 4.565514 and native latency 1.023861 ms.

| Guideline items | Applied revision |
| --- | --- |
| 1–2 | Abstract scopes execution ablations and loss improvement to 14M; introduction adds fixed-pressure comparisons at 31M/70M. Final co-design sentence/paragraph preserved. |
| 3–4 | Table 1 gains analytic 31M ceilings and an execution-coverage distinction; experimental opening states the four-size coverage and extensive 14M core. |
| 5–6 | Figure 5 replaced; correct T7/Ph comparator, guide labels, threshold order and legend. Section 4.4 has four focused paragraphs, separating sweep changes from native-Base speedups. |
| 7–8 | Discussion distinguishes thresholding from pressure; limitations state what the 31M comparison does not establish. |
| 9–11 | Appendix C adds pinned architecture, explicit T2 numerator, full training-settings column, actual executed conditions and matched protocol. |
| 12 | Figure 6 includes all 712 actual 31M Base loss/pre-clipping task-gradient records; see O004. |
| 13–14 | New 11-row 31M table uses measured pooled sparsity and fixed native-Base deltas. D.2 adds 31M token exposure while retaining the 410M stress-test interpretation. |
| 15–16 | E.1 documents the opt073 port and relocates 70M kernel history. E.2 records all 33 qualifications, versions, timing sessions and distinct native/kernel denominators. |
| 17 | Existing 14M/70M same-checkpoint controls are unchanged; no new 31M causal runtime attribution. |

The broad-pressure T7/Pall loss penalties (0.621 at 14M, 1.116 at 70M) move
to D.1. The initial 70M kernel's 3.325 ms versus reoptimized 2.732 ms history
moves to E.1. The historical 70M T2 kappa 0.5 endpoint retains its
separate-session qualification. No 14M/70M coordinate is remeasured or changed.

## Caveats

One training seed per condition; fixed token budget; architecture-specific
kernels and different timing sessions. Small latency gaps are descriptive.
The 31M comparison changes five threshold sites jointly and does not establish
spillover, site-specific gradient conflict or a same-checkpoint sparse-execution
decomposition at 31M. No fitted scaling law, universal dominance, general
regularization mechanism or automatic kernel portability is claimed.

## Verification and sources

The source-data audit checks all 11 checkpoints and 33 processes, reproduces
the displayed rounded table values, and verifies all 15 matched-kappa claims.
Protected 14M Tables 2–4, Figures 1/3/4, the spillover and OL1 sections,
Appendix A, the 410M table and both execution-control subsections are unchanged.
The historical 14M/70M results table is unchanged. Every plotted coordinate
and measured Pareto member is unchanged from the pre-integration figure.

The draft was rebuilt with `latexmk -pdf -interaction=nonstopmode
-halt-on-error main.tex`. It has 18 pages, with main text ending on page 9,
as before. No unresolved citations/references or overfull/underfull boxes are
reported. The nonfatal `Infinite glue shrinkage found in box being split`
diagnostic from the unchanged longtable also reproduces when compiling the
pre-edit Git source with the same local toolchain; its rendered table is intact.
All pages were rendered for layout review, with enlarged inspection of the
changed figure/table pages. Exact verification and output hashes are recorded
in `data/manuscript-verification.json`.

Sources: `data/manuscript-evidence.json`, `data/scale-figure.json`,
`data/base-model-training.json`, their hash-linked raw artifacts, and
`manuscript_changes.md`. Generators: `02_plot.py`, `04_manuscript_evidence.py`,
`05_base_training.py`; integration verification: `06_verify_manuscript.py`;
manuscript entry point: `manuscript/draft/main.tex`.
