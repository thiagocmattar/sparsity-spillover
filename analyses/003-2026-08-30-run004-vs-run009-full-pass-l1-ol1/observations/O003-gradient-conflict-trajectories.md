# O003 - Gradient-conflict trajectories

## Question

How does raw task-pressure gradient interaction evolve during training for
naive L1 and OL1 at each matched lambda?

## Sources and coverage

- The four verified Run 004 naive-L1 and four verified Run 009 OL1
  `events.jsonl` histories selected from their run verification records.
- One seed, Pythia-14M, ReLU topology `A1-H`, pressure only at `h`, and lambda
  `{0.05, 0.1, 0.5, 1.0}`.
- All 712 completed optimizer boundaries and 1,493,172,224 training input tokens
  per condition. No selected boundary overflowed or skipped its optimizer step.
- The runs have the same initial-parameter and training-schedule hashes.

`04_plot_gradient_conflict_trajectories.py` rejects invalid evidence status,
unmatched initialization or schedule, incomplete or non-contiguous event
histories, a lambda-grid mismatch, non-finite interaction values, cosines
outside `[-1, 1]`, or conflict flags inconsistent with the raw dot-product
sign.

## Reduction

Each 712-boundary history is divided in order into 24 contiguous bins. Each bin
contains 29 or 30 boundaries. The horizontal coordinate is the bin's median
cumulative input-token count.

The top row reports the median and interquartile range of the raw task-pressure
gradient cosine within each bin. The bottom row reports

```text
count(raw task-pressure dot < 0) / boundaries in bin.
```

This is a count-first boundary fraction. The bins summarize local temporal
variation; their bands are not confidence intervals, and the boundaries are
not independent replicates.

## Figure caption and legend

**Figure 2. Raw task-pressure gradient interaction over one full MiniPile
training pass.** Columns show lambda `0.05`, `0.1`, `0.5`, and `1.0`. The top
row shows the within-bin median raw gradient cosine; shaded bands span the
within-bin interquartile range. The bottom row shows the percentage of
boundaries in each bin whose raw task-pressure dot product is negative. Blue
solid lines with circles denote Run 004 naive L1; orange dashed lines with
squares denote Run 009 OL1. The horizontal reference lines mark zero cosine
and 50% conflict incidence. All cosine panels share `[-0.5, 0.5]`; all conflict
panels share `[0%, 100%]`. Each condition contributes 712 boundaries, grouped
into 24 contiguous bins of 29 or 30. The OL1 raw task gradient is globally
clipped before this diagnostic, whereas Run 004 records the naive-L1 task
component before the combined gradient is clipped. Global clipping multiplies
the task gradient by a positive scalar, so it does not change the raw cosine or
dot-product sign shown here. One seed was evaluated.

## Observed pattern

At lambda `0.05`, both methods remain centered close to zero cosine and finish
near a 50% conflict rate. At lambda `0.1`, the median cosine becomes modestly
negative and the conflict rate rises during the second half of training.
Lambda `0.5` and `1.0` show a stronger late-training shift: median cosine moves
further below zero, and most boundaries in the final bins have negative dots.
The effect is strongest at lambda `1.0`.

The binned naive-L1 and OL1 raw trajectories broadly track one another within
each lambda. At the larger lambdas, OL1 is generally less negative late in
training, consistent with the overall raw-cosine summaries in
`gradient_tables.md`. This is a comparison of the shared raw-gradient metric;
it does not compare naive L1 with OL1's AdamW-relative direction cosine.

## Caveats and nonclaims

- The trajectory is descriptive evidence from one seed, one model scale, and
  one training recipe. Bin-to-bin variation is not replicate uncertainty.
- Lambda does not algebraically change the angle between the recorded
  unweighted component gradients at a fixed parameter state. Differences emerge
  as the conditions follow different training trajectories.
- A negative pressure-component dot does not imply that the complete naive-L1
  gradient opposes the task gradient. O002 shows that the combined raw gradient
  remained task-aligned at every recorded naive-L1 boundary.
- The figure does not show the OL1 adaptive cosine before projection. That
  quantity is defined only on the OL1 task-only AdamW state and is not
  reconstructible for the naive-L1 trajectories.
- No finding or manuscript claim is promoted from this figure.

## Provenance

- Plot and reduction: `../04_plot_gradient_conflict_trajectories.py`
- Figure: `../figures/02-gradient-conflict-trajectories.pdf`
- Aggregate gradient tables: `../gradient_tables.md`
- Geometric interpretation: `O002-gradient-interference-and-ol1-geometry.md`
