# O004 - OL1 orthogonalization trajectories

## Question

Within the same Adam-relative geometry, how does OL1 change task-pressure
interaction at each optimizer boundary?

## Sources and coverage

- The four verified Run 009 OL1 `events.jsonl` histories.
- One seed, Pythia-14M, ReLU topology `A1-H`, pressure only at `h`, trust budget
  `1.0`, and lambda `{0.05, 0.1, 0.5, 1.0}`.
- All 712 completed boundaries and 1,493,172,224 training input tokens per
  condition. No selected boundary overflowed or skipped its optimizer step.

The source validator is described in O003. It additionally requires each OL1
projection flag to equal the sign test on the stored adaptive dot product.

## Reduction

Every optimizer boundary contributes two points in the same adaptive geometry:

```text
before = cosine(d_task, d_pressure)
after  = cosine(d_task, d_safe).
```

Here `d_task` is the task-only AdamW direction and `d_pressure` uses the same
task second-moment denominator. When their dot is negative, OL1 projects
`d_pressure` onto the boundary orthogonal to `d_task`; otherwise it leaves the
direction unchanged.

The orange and green rolling summaries use the same centered 51-boundary
window as Figure 2: arithmetic mean with p05-p95 fill. Every one of the 712
boundary-level values remains visible.

## Figure caption and legend

**Figure 3. OL1 Adam-relative interaction before and after conflict
projection.** Columns show lambda `0.05`, `0.1`, `0.5`, and `1.0`. Every point
is one optimizer boundary. Orange squares show the cosine between the task-only
AdamW direction and the pre-projection pressure direction. Green triangles show
the cosine after OL1 projection. Lines are centered 51-boundary rolling means;
shaded regions are rolling p05-p95 intervals. Negative cosine denotes conflict.
All panels share `[-0.19, 0.02]`, covering every value. Each condition
contributes 712 boundaries, one every 2,097,152 input tokens. One seed was
evaluated.

## Observed pattern

The pre-projection cosine is negatively biased throughout training, with a
larger negative displacement at larger lambda. Projection was applied on
`697/712` boundaries at lambda `0.05` and `711/712` boundaries at each larger
lambda. On those projected boundaries, the post-projection cosine is numerical
zero; the largest absolute residual across all projected cases is `2.4e-8`.

The few nonprojected boundaries had nonnegative adaptive dots and retain their
original pressure directions. The figure therefore shows exactly the property
implemented by OL1: a negative Adam-relative component is removed, while an
already compatible component is preserved.

## Caveats and nonclaims

- The post-projection trajectory is near zero by construction. It validates the
  implemented geometry but is not evidence that orthogonalization improves
  final loss or sparsity.
- Orthogonality is defined against the task-only adaptive direction, excluding
  decoupled weight decay. It is not orthogonality to the current raw task
  gradient or a guarantee of finite-step loss preservation.
- Rolling bands are temporal p05-p95 summaries, not uncertainty intervals.
- The evidence covers one seed, one model scale, and one training recipe. No
  finding or manuscript claim is promoted from this figure.

## Provenance

- Plot and reduction: `../04_plot_gradient_conflict_trajectories.py`
- Figure: `../figures/03-ol1-orthogonalization-trajectories.pdf`
- Aggregate gradient tables: `../gradient_tables.md`
- Mixed-geometry context: `O003-gradient-conflict-trajectories.md`
