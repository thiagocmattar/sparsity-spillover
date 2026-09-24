# O004: Base training at 14M, 31M, 70M and 410M

## Question

How does the 31M Base trajectory compare with the existing Base loss and
pre-clipping task-gradient histories under the fixed token budget?

## Method and coverage

`05_base_training.py` adds the 712 actual train events from Run054 Base to
the three verified histories from Analysis021. All four models have 712
updates, 1,493,172,224 input tokens and seed 1234: 2,848 records total.
The script verifies contiguous steps and token counts, finite task loss and
pre-clipping gradient norms, no skipped/overflowed updates, and consistency
of the recorded pre/post-clipping norms with clipping at 1.0.

The gradient field is `adamw_gradient_norm_pre_clip`: the global L2 norm
of accumulated, unscaled task gradients immediately before clipping and
AdamW. Base has no pressure correction. No pressure-gradient or post-clipping
norm is substituted. The historical raw and smoothed coordinates, axis
limits and original nine-update smoothing are preserved exactly.

## Caption

**Base-model (T0/P0) training at 14M, 31M, 70M and 410M.** (a) Task loss;
(b) global task-gradient norm before clipping, on a logarithmic axis. Faint
lines show individual updates; solid lines show centered arithmetic means
over nine updates, shortening the window at each endpoint. Colors identify
model sizes. The horizontal axis is cumulative input tokens. Norms are
measured after accumulation and are not normalized by parameter count.

## Result and caveats

The 31M Base finishes with training loss 4.489116 and pre-clipping gradient
norm 0.361599. The new trajectory fits the existing axes without truncation.
It adds context for the intermediate scale, not evidence of convergence or
a causal size effect. The 410M model retains its lower learning-rate schedule;
all runs have one seed and the same fixed token budget.

## Artifact and sources

PDF: `figures/03-base-model-training.pdf`; identical manuscript copy:
`manuscript/draft/figures/19-base-model-optimization.pdf`.
Data and source hashes: `data/base-model-training.json`.
Original generator/data: Analysis021 `06_a0_optimization.py` and
`data/a0-optimization.json`. New evidence: Run054
`artifacts/attempts/001-20260924-131935-fc20305e/events.jsonl` and `manifest.json`.
The generated PDF and its manuscript placement were rendered and inspected.
