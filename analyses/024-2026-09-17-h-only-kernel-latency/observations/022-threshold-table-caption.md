# Table 2: Threshold placement at fixed h-only pressure

Table: [TABLE-2-threshold-placement.md](../TABLE-2-threshold-placement.md).
Manuscript placement: [Section 4.3](../../../manuscript/draft/training-results.tex),
`sec:qkv-thresholding-results` and replacement of `tab:qkv-thresholding` for this
fixed-pressure comparison. No additional main figure is generated.

## Question, method, and coverage

What changes when q,k,v thresholding is added while pressure remains only on h?
Ten checkpoint pairs cover five κ settings at 14M and 70M. Each cell is
Y(T7/P_h)−Y(T4/P_h), using ordinary-final validation loss and full-model
integer-pooled logical sparsity. Each source checkpoint covers 338 validation
blocks; source identities and unrounded differences are preserved in the derived table.

## Publication caption

**Threshold placement at fixed h-only pressure.** Each row compares 7-site
minus 4-site thresholding at the indicated κ, holding OL1 pressure on h.
Columns report changes in model-wide sparsity (percentage points) and validation
loss (nats/token) for 14M and 70M. Positive loss changes are worse; positive
sparsity changes indicate more logical zero products. Each setting has one final
checkpoint with matched initialization and training order within size. The five
thresholds are not independent-seed replicates. Values are derived from unrounded
source records and rounded only for display.

## Result and proposed manuscript writing

> Under fixed h-only pressure, broader threshold placement adds 6.436 pp
> sparsity at 14M and 2.997 pp at 70M for κ=.5, with loss changes of +0.0094
> and +0.0315 nats/token. At κ=.05, the corresponding changes are +1.090 pp
> and −0.0006 loss at 14M, versus +0.328 pp and +0.0707 loss at 70M.
> This comparison separates threshold placement from pressure placement;
> changing from 4-site OL1(all) to 7-site OL1(all) would change both scopes.

## Caveats and provenance

Even κ=0 has separate trained checkpoints, so tiny differences are retained rather
than forced to zero. These paired single-seed settings do not establish population
effects. Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[unrounded pairs](../data/paper-derived.json), and
[checkpoint records](../data/paper-checkpoints.json).
The paragraph is proposed manuscript text, retained for review.
