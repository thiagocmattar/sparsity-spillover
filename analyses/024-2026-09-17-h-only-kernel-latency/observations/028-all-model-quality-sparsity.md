# All-model quality-sparsity overview

## Question and approved scope

Show the quality-sparsity operating range of every trained condition in the
paper at 14M, 70M and 410M. Explain how interventions approach the analytic
sparsity ceilings at a quality cost, and why detailed pressure-placement and
execution analysis focuses on 14M/70M under the realized training budget.
The author requested the figure and subsection rewrite; no new training,
checkpoint evaluation, kernel measurement or cloud work was needed.

## Figure and provenance

- PDF: [11-all-model-quality-sparsity.pdf](../figures/11-all-model-quality-sparsity.pdf).
- Builder: [20_plot_quality_all_sizes.py](../20_plot_quality_all_sizes.py).
- Auditable export: [all-model-quality-sparsity.json](../data/all-model-quality-sparsity.json).
- Tests: [test_all_model_quality.py](../test_all_model_quality.py).
- Manuscript: [training-results.tex](../../../manuscript/draft/training-results.tex),
  Section 4.1, label `fig:quality-sparsity-all-sizes`; Figure 3 on page 6.
- Adoption/build record: [review](../../../manuscript/draft/reviews/2026-09-18-all-model-quality-overview/README.md).

The export hashes 225 input files, the builder and resulting PDF. It contains
all plotting coordinates, pooled integer counts, checkpoint identities,
protocol checks, clipping visibility, recipe styles and supporting speedup
calculations. The PDF and export are copied byte-for-byte into the manuscript;
its two `SOURCES.json` manifests record their hashes.

## Method and coverage

The 1-by-3 plot contains 40/22/12 trained endpoints (74 total) at 14M/70M/410M.
These are all conditions in the paper's retained cohort, not every exploratory
run in the repository. There are ten executed recipe families at 14M, six at
70M and four at 410M. Missing families are not inferred or interpolated.

The existing 62-point Analysis024 checkpoint table supplies 14M/70M source
identities; Analysis018's `figure_data.json` supplies the twelve 410M identities
and its protocol/ceiling records. Every point is reconciled to its retained
`metrics.json`, `diagnostics/logical_products.json` and `manifest.json`.
Loss uniformly uses **ordinary reloaded final-checkpoint validation**; sparsity
is 100 times pooled block zero-product counts divided by full-model product
counts, including the dense output head in the denominator. No batch/layer
percentage averaging is used. Both evaluation passes cover all 338 complete
2,048-token validation blocks from 500 documents (692,224 input tokens), with
the 1,444-token tail excluded. Logical-count evaluation is FP16.

All 62 previous 14M/70M loss/sparsity coordinates remain exact. For 410M the
uniform ordinary-final loss differs slightly from historical logical-pass
losses; both values remain in the export. Historical tables are preserved,
and the manuscript appendix explains the pass difference.

All 74 records confirm 712 completed optimizer steps, 1,493,172,224 input tokens
and model/data-order seeds 1234. Initial parameter hashes and data schedule
hashes match within each model size. Cross-size parameter counts and realized
learning rates are retained with their evidence; the 410M peak is about
3e-4 versus 1e-3 at 14M/70M. The common token budget corresponds to 106.14,
21.20 and 3.68 tokens per parameter. This does not match optimization regimes
across model sizes or establish convergence.

Run030 supplies sixty fixed-control post-hoc settings: Base and ReLU at each
size, clipping targets p=0,.1,...,.9 at a,m,h,z. Their checkpoint content hashes
match the trained controls; all counts and original clipping-pass losses are
retained. No clipping latency is inferred for 410M.

## Presentation and legend

All three panels share validation-loss limits 4.0-6.2; every trained endpoint
lies inside the view. X limits differ by size to accommodate their ceilings.
Eight/five/seven high-loss clipping evaluations at 14M/70M/410M exceed the
displayed Y range. They remain in the export, and the caption refers to the
complete appendix paths. The plot contains no nondominated-point marking.

Ten distinct colors identify recipes. T0/P0 and T1/P0 are Base and ReLU;
T1/P1 has separately labeled historical naive L1 and OL1 variants. Multisite
Ph/Pall means h-only/all-thresholded-site OL1; P0 means no pressure. The
existing Figure08 teal/blue/purple/orange colors and dashed/solid pressure
styles are retained. New no-pressure and single-site families have distinct
colors. Circular markers, one shared legend, subtle post-hoc annotations and
T4/T7 ceiling guides follow the reference figure. Lines connect separately
trained kappa settings (0,.01,.05,.1,.5), or lambda settings (.05,.1,.5,1) for
T1/P1, and do not represent training trajectories.

## Publication caption

**Quality-sparsity trade-offs across model sizes.** All 40/22/12 trained
conditions at 14M/70M/410M, with a shared loss scale. Colors distinguish
executed recipes; lines connect separate threshold settings
(kappa=0,.01,.05,.1,.5), or pressure weights (lambda=.05,.1,.5,1) for T1/P1.
Its L1 and OL1 variants are shown separately; all other pressured recipes use
OL1. Dotted paths apply post-hoc clipping at a,m,h,z to the fixed base and ReLU
controls. Vertical guides mark the T4/T7 analytic sparsity ceilings. All
trained points are visible; 8/5/7 high-loss clipping evaluations continue above
the view (complete paths in the three appendix clipping figures). Trained
losses use ordinary final-checkpoint evaluation; sparsity uses pooled FP16
logical counts over all 338 complete validation blocks.

## Results and associated manuscript writing

The subsection leads with increased attainable sparsity and its quality cost.
For T7/Pall at kappa=.5:

| Model | S_model (%) | Fraction of T7 ceiling (%) | Validation loss | Loss increase over Base |
|---|---:|---:|---:|---:|
| 14M | 27.4827 | 91.75 | 5.829390 | 0.620807 |
| 70M | 40.6019 | 82.15 | 5.215976 | 1.116209 |
| 410M | 80.6155 | 92.40 | 5.120901 | 0.573438 |

The cost varies: 14M T7/Ph at kappa=.05 has 10.13% sparsity and loss 5.195,
close to the base model's 5.209. Each endpoint has one training seed; this is
an operating-point observation, not evidence of a generalization improvement.

The 410M **base model** has higher loss (4.547) than 70M (4.100). Some
interventions reverse the ordering, so the manuscript does not assert that
every 410M recipe performs worse. The differing learning-rate amplitude and
tokens per parameter are plausible explanations, not isolated causes. The
common 712 steps and token budget are explicit; Appendix C.4 and the baseline
optimization figure support the discussion (currently Figure 9, previously 8).
The limited 410M coverage remains visible, while detailed paired/runtime
analysis focuses on 14M/70M.

All four matched T4/T7, Ph/Pall recipes at kappa=.5 have greater model-wide
sparsity and greater speedup relative to their **own size's optimized base**
at 70M: 1.89-2.05x versus 1.37-1.42x at 14M. The supporting ratios use saved
full-model latency, with denominators 0.651573035 ms (14M) and 3.313247016 ms
(70M). Eight records, denominator sessions and kernels are exported. Figure 1
contains these raw latency points. These ratios compare different trained
models and quality levels; they are not same-checkpoint native-to-sparse
implementation speedups.

## Interpretation limits

Architectural ceilings and observed zero-product opportunity are not measured
runtime gains. Larger ceilings themselves explain part of the cross-size
sparsity difference. Runtime sessions differ across cohorts; 70M uses the
qualified shape-specific port. The observed speedup trend does not isolate
parameter count, match kernel optimization effort, establish a scaling law,
or extend runtime results to 410M. FP16 quality/counts and BF16 timings remain
different measurements. No uncertainty across seeds can be estimated.

## Reproduction and verification

From the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/20_plot_quality_all_sizes.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s analyses/024-2026-09-17-h-only-kernel-latency -p test_all_model_quality.py -v
```

Four focused tests pass: source/count/loss reconciliation and protocol;
complete recipe coverage with distinct pressure methods; control-clipping
identity/coverage, visibility and analytic ceilings; source/artifact hashes
and optimized-base speedup denominators. The generator verifies that existing
analysis PDFs are unchanged. The manuscript builds with resolved references
and no overfull boxes. The standalone figure, manuscript pages 5-6 and the
full 35-page layout were visually checked; see the adoption record.
