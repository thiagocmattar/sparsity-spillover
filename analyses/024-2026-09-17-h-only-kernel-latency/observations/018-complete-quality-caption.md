# Appendix A1: Complete trained and post-hoc quality-sparsity trajectories

PDF: [A1-complete-quality-sparsity.pdf](../figures/A1-complete-quality-sparsity.pdf).
Manuscript placement: [Results appendix](../../../manuscript/draft/results-appendix.tex),
supporting [the overview](../../../manuscript/draft/introduction.tex),
`fig:quality-sparsity-overview`.

## Question, method, and coverage

What lies outside Figure 1's main range? Show the full 54-checkpoint primary cohort
and every retained post-hoc evaluation attached to it: 220 at 14M and 120 at 70M,
ten doses for each of 34 checkpoints. Unlike the main panel, this appendix includes
clipping of multisite checkpoints as well as dense/ReLU. The 20 h-only multisite
checkpoints have no retained clipping sweep; none is fabricated. We retain dose
ties as separate records even when plotted at identical coordinates. Absolute
trained losses are ordinary-final; clipped losses are the measured FP16 eager
sweep losses. The largest zero-dose versus ordinary-final loss difference across
these 34 checkpoints is 0.000425389 nats/token; no curve has been shifted to force
agreement. This comparison differs from Run030's original eager-to-eager audit,
whose maximum was 0.000180712 over its original 54-checkpoint cohort.

## Publication caption

**Complete trained and post-hoc quality-sparsity trajectories.** Separate 14M
and 70M panels use absolute validation loss and a shared full-range Y axis.
The same 32/22 trained endpoints as Figure 1 are shown alongside all 340 retained
clipping evaluations for this declared cohort (220/120 by size). Dotted paths
connect target fractions 0,.1,...,.9 with separate magnitude thresholds calibrated
at a,m,h,z in each layer on ten training blocks. Existing trained gates are
preserved, including q,k,v gates in 7-site models; no retraining is performed.
Dense/ReLU clipping paths are emphasized; the other evaluated paths are shown
faintly. The 20 multisite OL1(h) checkpoints have no retained post-hoc sweeps.
Fourteen control clipping records outside Figure 1's Y range are included here.
Trained lines connect increasing κ, not training steps or attainable interpolated
models. All measured losses and pooled logical counts use complete validation
(338 blocks from 500 documents, excluding the 1,444-token tail). Coverage guides
describe computational reach, not measured speedup. There is one trained final
checkpoint per setting and no seed uncertainty.

## Result and proposed manuscript writing

> The full clipping paths document the cost of increasing sparsity beyond the
> main figure's quality range. Several high-dose evaluations reach losses above
> nine nats/token. These evaluated outcomes are retained even when dominated or
> outside the range used to compare trained endpoints. Clipping is an additional
> operation on a fixed trained checkpoint; it neither supplies missing trained
> recipes nor establishes the quality of unevaluated intermediate settings.

## Caveats and provenance

The clipping cohort is incomplete for the new h-only models, and the 70M
pressure-free multisite training cohort is absent. Small eager/ordinary evaluation
differences should not be read as meaningful improvements. Comparison uses full
measured trajectories rather than only per-checkpoint Pareto records.

Source: [paper_quality_figures.py](../paper_quality_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[all clipping IDs and zero-dose audit](../data/paper-checkpoints.json),
[figure selection](../data/paper-derived.json), and
[Run030 protocol](../../../runs/030-2026-09-08-all-models-posthoc-clipping/README.md).
The paragraph is proposed manuscript writing, not an edit to the draft.
