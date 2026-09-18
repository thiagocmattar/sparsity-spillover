# Appendix A2: Effects of adding narrow and broad pressure

PDF: [A2-pressure-versus-none.pdf](../figures/A2-pressure-versus-none.pdf).
Manuscript placement: [Results appendix](../../../manuscript/draft/results-appendix.tex),
supporting [Section 4.2](../../../manuscript/draft/training-results.tex), `sec:paired-results`.

## Question, method, and coverage

How does each pressure recipe compare with no pressure at each trained threshold?
The 20 14M contrasts are two scopes × five thresholds × two pressure increments.
Rows show h-minus-none and all-minus-none; columns show ordinary-final loss,
logical sparsity in percentage points, and full-model K050 latency in microseconds.
All pairs use exact checkpoint keys at fixed threshold scope and κ. The five
thresholds are separate categorical positions, with shared Y limits within columns.

## Publication caption

**Effects of adding narrow and broad pressure at 14M.** Rows compare OL1(h) and
OL1(all) against no pressure, with threshold scope and κ held fixed. Columns
show loss change, model-wide sparsity change (percentage points), and full-model
latency change (microseconds). Blue diamonds and orange triangles distinguish
4-site and 7-site thresholding; short and long dashes distinguish h-only and
all-site increments. Zero lines show no difference. Each κ is a separate trained
setting, not an independent seed; connecting lines do not imply training trajectories
or attainable interpolated models. Loss and integer-pooled logical sparsity use
complete FP16 validation. Timing uses full-model K050 on RTX 5090, BF16, batch one,
2,048 tokens and 50,304 logits, with geometric means of 1,344 raw host samples per
checkpoint. The 7-site h-only comparisons span Run033 and Run029 physical GPU/host
sessions; other pairs are within Run029. Timing differences are descriptive and
do not provide seed uncertainty or establish small cross-session wins.

## Result and proposed manuscript writing

> At κ≤.1, h-only pressure lowers 14M loss relative to no pressure for both
> threshold scopes. At κ=.5, that loss contrast becomes positive. The runtime
> response also depends on κ: the largest h-only latency reductions occur at
> intermediate thresholds, while broad 7-site pressure changes latency little
> over much of the grid despite its large high-threshold sparsity increment.
> Explicit threshold contrasts reveal this dependence without treating the
> threshold grid as a distribution of replicated experiments.

## Caveats and provenance

Pressure-free 70M counterparts are unavailable, so this appendix is 14M only.
Neither a causal effect of sparsity alone nor a direct Q/K/V-pressure effect is
identified. No uncertainty intervals are shown because cross-session measurement
variation and training uncertainty are not interchangeable.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[20 exact contrast pairs and sessions](../data/paper-derived.json), and
[checkpoint metadata](../data/paper-checkpoints.json).
Associated writing is proposed text for manuscript adoption.
