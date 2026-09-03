# O003 - A0 global task-gradient norms across model scales

## Question

How does the global full-model task-gradient L2 norm evolve with training
tokens, before and after clipping, for the matched A0 controls at Pythia-14M,
70M, and 410M?

## Method and coverage

Sources: the A0 training event streams from Runs 004, 018, and 019. Each stream
contains all 712 optimizer boundaries from one MiniPile pass, with 2,097,152
input tokens per boundary and 1,493,172,224 input tokens in total. The figure
plots every recorded `adamw_gradient_norm_pre_clip` and
`adamw_gradient_norm_post_clip` value without smoothing. All three runs used
global L2 clipping at 1.0; no A0 boundary overflowed or skipped its optimizer
update.

## Figure caption and legend

A0 global task-gradient L2 norm versus cumulative training tokens for
Pythia-14M, 70M, and 410M. Magenta solid lines show the norm immediately before
global clipping; blue dashed lines show the norm after clipping. The gray
dotted line marks the shared clip threshold of 1.0. Axes are shared across the
three panels and the norm axis is logarithmic. Panel annotations give the
number and fraction of clipped optimizer boundaries.

## Result

The clip threshold is exceeded at 5/712 boundaries for 14M (0.7%), 8/712 for
70M (1.1%), and 56/712 for 410M (7.9%). The maximum pre-clip norms are 2.0019,
3.5132, and 26.9615, respectively. The post-clip histories reconcile with the
threshold: clipped boundaries are reduced to approximately 1.0, while
unclipped boundaries retain their pre-clip norm.

## Caveats and nonclaims

These are global, unnormalized norms over differently sized parameter vectors;
their absolute cross-scale values are therefore not directly comparable as a
per-parameter gradient statistic. The figure documents the executed optimizer
mechanics and increased clipping incidence at 410M, but does not by itself
identify a learning-rate error or explain the 410M validation loss. Each scale
has one seed.

Source script: `01_build.py`.

Output: `figures/03-a0-gradient-norm-vs-tokens.pdf`.
