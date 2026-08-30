# Analysis 003 - Run 004 naive L1 versus Run 009 OL1

## Status

Completed as a descriptive cross-run analysis of two verified full-pass cohorts.
It covers both final endpoints and optimizer-boundary gradient interactions. No
finding was promoted and no manuscript text was changed.

## Question

At matched activation, topology, pressure site, lambda, initialization, data
order, training budget, and complete-validation coverage, how do operational
OL1 endpoints from Run 009 compare with naive-L1 endpoints from Run 004 in
per-site exact/near-zero mass, measured `R_model`, and final validation loss?
How does task-pressure interaction vary across the lambda grid, and what
property does operational OL1 enforce?

## Source cohort and matching

- Run 004 supplies the GeLU and ReLU controls and four ReLU `A1-H` naive-L1
  conditions at lambda `{0.05, 0.1, 0.5, 1.0}`.
- Run 009 supplies four ReLU `A1-H` OL1 conditions on the same lambda grid and
  reuses, rather than reruns, Run 004's controls.
- Every condition uses the same random initialization seed and parameter hash,
  realized training schedule, 712 optimizer boundaries, 1,493,172,224 training
  input tokens, and final validation over all 338 complete blocks (692,224 input
  tokens). The 1,444-token tail is excluded and reported.
- The executing difference within each lambda pair is `l1_naive` versus
  `orthogonal_l1` with trust budget `1.0`. Independently scheduled Pods are
  scientifically matched but not bitwise identical.

`01_compare.py` validates those identities, recalculates every activation and
logical fraction from stored integer counts, and writes `comparison.json` plus
the generated [tables](tables.md). Source paths and SHA-256 values are retained
in the machine-readable comparison.

`03_gradient_interaction.py` independently validates all 5,696 pressure
boundaries, reduces the stored scalar gradient diagnostics, and writes
`gradient_interaction.json` plus the generated
[gradient tables](gradient_tables.md). The event-file paths and hashes are
stored in that artifact.

## Results

Naive L1 and OL1 produced closely matched `h` exact-zero mass and `R_model` at
each lambda. OL1 minus naive-L1 final validation loss was `-0.008087`,
`-0.006083`, `-0.002453`, and `+0.018753` at lambda `0.05`, `0.1`, `0.5`, and
`1.0`, respectively. The lowest naive-L1 loss was `5.102276` at lambda `1.0`;
the lowest OL1 loss was `5.110235` at lambda `0.5`.

Exact zeros outside `h` were negligible in every condition, so measured
`R_model` was driven almost entirely by `h` zero operands in MLP W2. At the
largest lambda, OL1 had slightly less `h` exact-zero mass and `R_model` than
naive L1 and a higher final loss. These are endpoint descriptions, not a claim
of a method effect under replicate uncertainty.

The figure plots all ten unique endpoints. The four lambda values are written
beside their respective naive-L1 and OL1 points; the two reused controls appear
once. The validation-loss axis uses conventional ascending numerical order, is
explicitly labeled as an endpoint detail, and does not start at zero.

### Gradient interference and the OL1 guarantee

Along the naive-L1 trajectories, the fraction of boundaries with a negative
raw task-pressure dot product increased from `49.16%` at lambda `0.05` to
`70.93%` at lambda `1.0`. The median raw cosine moved from `+0.002505` to
`-0.105177`. These are interactions between the separate, unweighted component
gradients. Lambda therefore does not change their angle algebraically at a
fixed parameter state. Indeed, the first-boundary cosines agree across all
eight conditions within `1.8e-8`; the later differences arise along
lambda-dependent training trajectories.

For an isolated pressure update
`delta_pressure = -alpha * lambda * g_pressure`, the first-order task-loss
change is

```text
-alpha * lambda * <g_task, g_pressure>.
```

A negative dot therefore gives the pressure component a locally task-increasing
contribution. It does not follow that the complete naive-L1 direction opposes
the task gradient. In these runs,
`<g_task, g_task + lambda * g_pressure>` remained positive on all 2,848
naive-L1 boundaries. Among boundaries with component conflict, the median
reduction in this raw task-alignment term was `0.169%`, `0.354%`, `1.504%`, and
`2.675%` across the lambda grid; the largest reduction was `14.110%` at lambda
`1.0`. These calculations precede global clipping and AdamW and are not
measurements of realized loss change.

OL1 instead compares the pressure direction with the task-only AdamW adaptive
direction under a shared task second-moment denominator. It projects the
pressure direction onto the half-space of directions with nonnegative dot
product against that task direction. When the dot is negative, the projected
direction lies on the orthogonal boundary up to numerical tolerance; when it
is nonnegative, OL1 leaves the pressure direction unchanged. This is the
Euclidean projection onto the non-opposing half-space, so it changes the
original pressure direction as little as possible in that geometry. It is not
a proof that the finite correction cannot increase task or validation loss.

The trust budget bounds the correction-to-task direction ratio; it does not
remove the pressure-weight hyperparameter. With budget `1.0`, the cap was
active on `0/712`, `1/712`, `2/712`, and `76/712` boundaries at lambda `0.05`,
`0.1`, `0.5`, and `1.0`. Operational OL1 is therefore bounded, not
hyperparameter-free. Its correction is the largest requested scalar along the
projected ray that satisfies this bound, not the globally largest
pressure-reducing step that is safe for task loss.

The gradient-trajectory figure uses the raw interaction shared by both methods.
It divides each 712-boundary history into 24 contiguous bins of 29 or 30
boundaries. The top row shows the within-bin median cosine and interquartile
range; the bottom row shows the fraction of boundaries with a negative dot.
At lambda `0.05`, both methods remain centered near zero and finish near a 50%
conflict rate. Larger lambdas develop a clearer negative cosine and a rising
conflict rate late in training. The binned naive-L1 and OL1 raw trajectories
are similar at each lambda, although they are not stepwise identical.

## Outputs

- `comparison.json` - count-reconciled machine-readable endpoint and paired data.
- `tables.md` - exact-zero, near-zero, `R_model`, validation-loss, and matched-delta tables.
- `gradient_interaction.json` - boundary-level scalar reductions and source hashes.
- `gradient_tables.md` - raw-gradient, combined-gradient, adaptive-direction,
  projection, and trust-cap tables.
- `figures/01-r-model-vs-final-validation-loss.pdf` - quality/logical-opportunity scatter.
- `figures/02-gradient-conflict-trajectories.pdf` - matched raw-cosine and
  negative-dot-rate trajectories.
- `observations/O001-r-model-vs-final-validation-loss.md` - caption, provenance,
  observed pattern, and limits.
- `observations/O002-gradient-interference-and-ol1-geometry.md` - gradient
  interpretation, operational guarantee, and nonclaims.
- `observations/O003-gradient-conflict-trajectories.md` - trajectory reduction,
  caption, observed pattern, and limits.

## Limits

This is one seed and one Pythia-14M scale with no replicate uncertainty.
`R_model` is exact-zero logical-product opportunity rather than removed FLOPs
or measured runtime speedup. `attention_output` is post-`W_o` before residual
addition and is not the pre-`W_o` site `z`. Boundary observations are a
temporally dependent training sequence, not independent replicates. The
AdamW-relative pre-projection interaction is available only along the OL1
trajectories; naive-L1 moments combine task and pressure and cannot recover a
task-only AdamW direction. The stored artifacts contain scalar interactions,
not the historical gradient vectors. No finding or manuscript claim is
promoted by this analysis.
