# O003 - Naive-L1 context and OL1 projection trajectories

## Question

How do naive L1's raw task-pressure interaction and OL1's Adam-relative
interaction before and after projection evolve during training?

## Sources and coverage

- The four verified Run 004 naive-L1 and four verified Run 009 OL1
  `events.jsonl` histories selected from their run verification records.
- One seed, Pythia-14M, ReLU topology `A1-H`, pressure only at `h`, and lambda
  `{0.05, 0.1, 0.5, 1.0}`.
- All 712 optimizer boundaries and 1,493,172,224 training input tokens per
  condition. No selected boundary overflowed or skipped its optimizer step.
- The runs have identical initial-parameter and training-schedule hashes, and
  their cumulative token counts match at every boundary.

`04_plot_gradient_conflict_trajectories.py` rejects invalid evidence status,
unmatched initialization, schedule, or per-boundary token counts, incomplete or
non-contiguous event histories, a lambda-grid mismatch, non-finite interaction
values, raw cosines outside `[-1, 1]`, raw conflict flags inconsistent with the
dot sign, or OL1 projection flags inconsistent with the adaptive dot sign.

## Reduction

Every optimizer boundary is plotted. Each lambda panel therefore contains 712
naive-L1 raw-cosine points, 712 OL1 pre-projection adaptive-cosine points, and
712 OL1 post-projection adaptive-cosine points.

For each series, a centered 51-boundary window computes the arithmetic mean and
the p05 and p95 quantiles. The window covers 106,954,752 input tokens. Rolling
curves are drawn only where the complete window is available; the first and
last 25 boundaries remain visible as individual points.

The three series do not share one geometry:

- Naive L1 uses the raw cosine between the separately accumulated task and
  pressure gradients.
- OL1 before projection uses the cosine between the task-only AdamW adaptive
  direction and the pressure direction under the same task second-moment
  denominator.
- OL1 after projection replaces the pressure direction with its projected
  version.

Run 004 did not retain a task-only AdamW direction for naive L1. The naive-L1
series is therefore context, not a direct adaptive-space comparator.

## Figure caption and legend

**Figure 2. Per-boundary gradient-conflict trajectories with naive-L1 context
and OL1 projection.** Columns show lambda `0.05`, `0.1`, `0.5`, and `1.0`.
Every point is one optimizer boundary. Blue circles show Run 004's raw
task-pressure gradient cosine. Orange squares show Run 009's Adam-relative
task-pressure cosine before projection, and green triangles show the same OL1
cosine after projection. Lines are centered 51-boundary rolling means; shaded
regions are rolling p05-p95 intervals. The horizontal line marks zero;
negative cosine denotes conflict. All panels share `[-0.7, 0.8]`, which covers
every plotted point. Each condition contributes 712 boundaries, one every
2,097,152 input tokens. The naive-L1 and OL1 series use different geometries,
as stated in the legend and reduction. One seed was evaluated.

## Observed pattern

The naive-L1 raw cosine is broad and noisy around its rolling mean. Its mean
becomes more negative during later training as lambda increases, most clearly
at lambda `0.5` and `1.0`.

The OL1 Adam-relative cosine has a narrower but consistently negative
pre-projection trajectory. Its negative displacement grows with lambda. After
projection, the OL1 cosine lies at numerical zero on projected boundaries;
compatible positive directions are left unchanged. The green trajectory thus
shows the operational removal of the negative adaptive component.

## Caveats and nonclaims

- Raw naive-L1 cosine and Adam-relative OL1 cosine are not directly comparable
  values. The figure juxtaposes them only to provide training context and show
  the OL1 transformation.
- A naive-L1 pressure-versus-task-only-Adam cosine cannot be recovered because
  naive-L1 moments combine task and pressure and historical gradient tensors
  were not retained.
- Rolling p05-p95 regions summarize local temporal variation. They are not
  confidence intervals, and the 712 boundaries are not independent replicates.
- OL1's post-projection zero is an algorithmic property. It does not prove that
  the finite correction preserves training or validation loss.
- The evidence covers one seed, one model scale, and one training recipe. No
  finding or manuscript claim is promoted from this figure.

## Provenance

- Plot and reduction: `../04_plot_gradient_conflict_trajectories.py`
- Figure: `../figures/02-gradient-conflict-trajectories.pdf`
- Aggregate gradient tables: `../gradient_tables.md`
- Geometric interpretation: `O002-gradient-interference-and-ol1-geometry.md`
