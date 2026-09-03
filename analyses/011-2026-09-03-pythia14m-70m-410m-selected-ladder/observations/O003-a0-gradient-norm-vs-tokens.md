# O003 - A0 global task-gradient norms across model scales

## Question

How do training task loss, the global full-model task-gradient L2 norm before
and after clipping, and the learning-rate schedule evolve with training tokens
for the matched A0 controls at Pythia-14M, 70M, and 410M?

## Method and coverage

Sources: the A0 training event streams from Runs 004, 018, and 019. Each stream
contains all 712 optimizer boundaries from one MiniPile pass, with 2,097,152
input tokens per boundary and 1,493,172,224 input tokens in total. The figure
plots every recorded `task_loss`, `adamw_gradient_norm_pre_clip`,
`adamw_gradient_norm_post_clip`, and `learning_rate` value without smoothing.
The task loss is the mean causal-language-model loss across the microbatches in
that boundary. All three runs used global L2 clipping at 1.0; no A0 boundary
overflowed or skipped its optimizer update.

## Figure caption and legend

A0 optimization trajectories versus cumulative training tokens for
Pythia-14M, 70M, and 410M. Columns identify model size. The top row shows
training task loss. In the middle row, magenta solid lines show the global
task-gradient norm immediately before clipping and blue dashed lines show it
after clipping. The gray dotted line marks the shared clip threshold of 1.0.
The bottom row shows the recorded learning rate in green on a shared absolute
scale. Axes are shared within rows and the norm axis is logarithmic.
Annotations give the final boundary loss, clipping frequency, and peak/final
learning rates.

## Result

The final boundary task losses are 5.2439 at 14M, 3.9830 at 70M, and 4.4483 at
410M; their minima are 5.1956, 3.9285, and 4.4214. The clip threshold is
exceeded at 5/712 boundaries for 14M (0.7%), 8/712 for 70M (1.1%), and 56/712
for 410M (7.9%). The maximum pre-clip norms are 2.0019, 3.5132, and 26.9615,
respectively. The post-clip histories reconcile with the threshold: clipped
boundaries are reduced to approximately 1.0, while unclipped boundaries retain
their pre-clip norm. The learning-rate schedules all peak at boundary 9. The
14M and 70M schedules peak at `1e-3` and finish at `1e-4`; the 410M schedule
peaks at `3e-4` and finishes at `3e-5`.

## Caveats and nonclaims

These are global, unnormalized norms over differently sized parameter vectors;
their absolute cross-scale values are therefore not directly comparable as a
per-parameter gradient statistic. The figure documents the executed optimizer
mechanics and increased clipping incidence at 410M, but does not by itself
identify a learning-rate error or explain the 410M validation loss. Each scale
has one seed.

Source script: `01_build.py`.

Output: `figures/03-a0-gradient-norm-vs-tokens.pdf`.
