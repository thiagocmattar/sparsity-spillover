# Figure 9: Operation contributions across size and threshold

PDF: [09-14m-70m-operation-contributions.pdf](../figures/09-14m-70m-operation-contributions.pdf).
Builder: [17_plot_operation_grid.py](../17_plot_operation_grid.py).
Exact numerators, denominators, totals, membership and hashes:
[14m-70m-operation-contributions.json](../data/14m-70m-operation-contributions.json).

## Question and requested design

How does the composition of model-wide logical sparsity change across retained
pressure recipes, model sizes and thresholds? This is a new version of
[Figure 4](../figures/04-operation-sparsity-changes.pdf), with 14M in the first
row, 70M in the second row, and κ=0, .05 and .5 from left to right. All six
panels show absolute operation contributions, with one scale per model-size
row: 0-30 pp at 14M and 0-45 pp at 70M. The three columns within each row
share limits. The Y label is "Contribution to S_model (pp)".

The plot retains Figure 4's six-operation palette and stack order: QKV, QK,
PV, attention output, FFN up and FFN down. One shared operation legend serves
all panels. Every panel uses T4/Ph, T4/Pall, T7/Ph and T7/Pall in that order.
DejaVu Sans and 14/11.5/11/10/11-point title/panel/axis/tick/legend sizes are
preserved, on a 12.8-by-6.4-inch canvas. Only the requested pressure recipes
are displayed, without numerical annotations.

## Method and matched coverage

The figure uses 24 unique retained final checkpoints from the declared primary
cohort in [paper-checkpoints.json](../data/paper-checkpoints.json): twelve at
14M and twelve at 70M. Every panel has the same four recipes: T4/Ph, T4/Pall,
T7/Ph and T7/Pall. Pressure-free multisite controls are excluded at both sizes,
as requested. No values are imputed or drawn as zero placeholders.

Every segment is `C_o = 100 * N_zero,o / N_model`, where counts are pooled
before division across all six layers and all 338 complete validation blocks.
The full-model denominator includes the dense LM head; its value is
6,363,055,915,008 at 14M and 35,251,191,545,856 at 70M. It is common to all
segments and recipes within each model size. The six operation numerators
sum exactly to the aggregate block zero-product numerator; their contributions
sum to the checkpoint's model-wide sparsity. These are absolute levels,
not absolute values of signed differences or within-operation zero rates.

The canonical FP16 logical counters cover all 500 MiniPile validation documents
as 338 complete 2,048-token blocks (692,224 input tokens), excluding the
1,444-token tail. Future causal QK/PV positions are excluded from the product
counts. All actual zero operands in the six counted operations are included,
including naturally occurring zeros outside selected gate sites. One seed and
one final checkpoint per setting are represented; no new evaluation was run.

## Publication caption

**Operation contributions to model-wide sparsity on Pythia 14M and 70M.**
Rows show 14M (a-c) and 70M (d-f); columns show κ=0, 0.05 and 0.5. Each
stack decomposes the checkpoint's model-wide sparsity into operation
contributions `C_o = 100 N_zero,o / N_model` in percentage points. All
segments use the full-model denominator for that size, including the dense
LM head, and sum to model-wide sparsity expressed as a percentage. Colours
identify operations; X labels identify threshold topology and pressure scope.
Ph means h-only OL1, and Pall means OL1 on every selected threshold site.
Each panel contains the same four T4/T7 × Ph/Pall recipes. The Y scale is shared
across thresholds within each row and differs between sizes (0-30 pp at 14M;
0-45 pp at 70M). Integer counts
are pooled over six layers and 338 complete validation sequences from 500
documents, excluding the 1,444-token tail. These are logical zero-product
opportunities, not measured runtime savings; each condition has one seed.

## Result and associated proposed manuscript writing

Proposed placement: [Section 4.4, activation reshaping](../../../manuscript/draft/training-results.tex),
alongside or replacing the operation portion of `fig:activation-reshaping`.
This task creates an analysis variant without editing manuscript TeX.

> At κ=0, internal-attention QK/PV contributions are negligible under both
> pressure scopes. At κ=.05, T7/Pall has larger QK/PV contributions than
> T7/Ph at both sizes, but slightly lower total sparsity. At κ=.5 the
> internal-attention contribution becomes substantial: QK/PV supply 16.773 pp
> under T7/Pall versus 6.385 pp under T7/Ph at 14M, and 10.424 pp versus
> 2.887 pp at 70M. The corresponding full totals are 27.483% versus 16.664%
> at 14M and 40.602% versus 32.233% at 70M. The higher aggregate 70M sparsity
> therefore does not mean that every operation contributes a larger share
> of the full-model denominator.

## Caveats, provenance and verification

Architecture and the dense LM-head share differ by size. Comparing absolute
contributions across sizes is descriptive, not an isolated causal effect of
model size. Changing pressure scope also changes the equal-tensor pressure
normalization. There are no independent-seed uncertainty estimates. Scalar
zero products do not establish alignment with kernel skipping predicates,
and this figure makes no latency or speedup claim. It compares the two pressure
scopes; pressure-free multisite recipes are outside its scope.

The builder reuses the exact palette recorded for Figure 4 and saves complete
provenance. Three focused tests check source/output hashes, the original
eight overlapping Figure 4 pressure decompositions, all 24 plotted identities,
matched Ph/Pall membership, per-operation integer counts, full-model
denominators and all stack totals. Integer sums are exact; floating-point
contribution sums reconcile within 1e-12 pp. The existing seven paper-evidence checks
also pass. The one-page PDF was rendered at 2,100 pixels and visually reviewed;
all fonts are embedded. All twelve pre-existing PDFs are preserved.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/17_plot_operation_grid.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s analyses/024-2026-09-17-h-only-kernel-latency -p test_operation_grid.py -v
```
