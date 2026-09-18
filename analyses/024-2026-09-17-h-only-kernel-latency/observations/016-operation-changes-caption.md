# Figure 4: Pressure changes sparsity unevenly across operations

PDF: [04-operation-sparsity-changes.pdf](../figures/04-operation-sparsity-changes.pdf).
Manuscript placement: [Section 4.4](../../../manuscript/draft/training-results.tex),
`sec:activation-reshaping`, alongside/replacing the operation portion of
`fig:activation-reshaping`.

## Question, method, and coverage

Which operation counts explain the aggregate pressure response? Eight 14M contrasts
use κ=.05/.5, both threshold scopes, and the pressure increments h-minus-none and
all-minus-h. Every segment is 100 times the difference in exact pooled zero-product
counts for one operation divided by the same full-model denominator,
6,363,055,915,008 products over 338 validation blocks. Counts include all six layers
and the denominator includes the untargeted LM head. All six operation differences
sum exactly in integers to the aggregate numerator difference; their displayed
contributions sum to ΔS_model within floating-point precision. No segments are
inferred from aggregate sparsity, activation percentages, or latency.

## Publication caption

**Pressure changes sparsity unevenly across operations.** Two 14M panels show
κ=.05 and .5; their Y scales differ to retain the smaller changes. Bars compare
P_h−P_0 and P_all−P_h within each threshold scope. Here color encodes operation,
not recipe: QKV projection, FFN-up, FFN-down, attention-output projection, QK,
and PV. Each segment is ΔC_o=100 ΔN_0,o/N_model, with the same full-model
denominator in every segment. Positive and negative contributions stack on
opposite sides of zero; negative segments indicate reduced sparsity under the
treatment. The black marker is their signed sum, equal to the net change in
model-wide sparsity. Full-validation integer counters are used. These quantities
are logical zero-product opportunities, not measured runtime savings.

## Result and proposed manuscript writing

> Narrow h pressure changes sparsity beyond FFN-down. For 7-site thresholding
> at κ=.05, h-only versus no pressure adds 0.712 pp from FFN-down and a net
> 0.287 pp from the other operations, for +0.999 pp overall. At κ=.5 its
> +1.277 pp net change is driven mainly by +1.619 pp in QK, partly offset by
> −0.471 pp in PV. Broadening 7-site pressure at κ=.05 increases QK/PV
> contributions by +0.684 pp while projection contributions decrease by
> −0.947 pp. At κ=.5, QK/PV instead supply +10.388 pp of the +10.819 pp
> total increment. Aggregate sparsity therefore conceals changes of opposite
> signs across operations.

## Caveats and provenance

This is evidence of a nonlocal response in the retained trained endpoints,
not causal identification of a particular propagation path. Logical product
counts do not specify whether zeros align with kernel skipping predicates.
No 70M operation figure or new measurement is inferred from these 14M bars.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[source checkpoint counters](../data/paper-checkpoints.json), and
[integer differences, denominators, and net checks](../data/paper-derived.json).
Associated paragraph is proposed manuscript writing only.
