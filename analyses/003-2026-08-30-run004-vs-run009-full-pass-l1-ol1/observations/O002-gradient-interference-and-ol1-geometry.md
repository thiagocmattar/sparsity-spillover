# O002 - Gradient interference and OL1 geometry

## Question

Does the activation-L1 pressure gradient oppose the task gradient more often
along higher-lambda naive-L1 trajectories? What does operational OL1 guarantee
when such opposition is present?

## Sources and coverage

- The four Run 004 naive-L1 and four Run 009 OL1 `events.jsonl` files selected
  by their verified condition records.
- One seed, Pythia-14M, ReLU topology `A1-H`, pressure only at `h`, and lambda
  `{0.05, 0.1, 0.5, 1.0}`.
- Each condition contributes all 712 completed optimizer boundaries: 2,848
  naive-L1 and 2,848 OL1 boundaries. No boundary overflowed or skipped its
  optimizer step.
- Run 004 records separate, unweighted task and pressure gradients before they
  are combined, globally clipped, and consumed by AdamW. Run 009 additionally
  records the task-only AdamW adaptive direction, the pressure direction under
  the same task second-moment denominator, projection, and trust scaling.

`03_gradient_interaction.py` rejects incomplete step sequences, invalid evidence
status, condition or lambda mismatches, skipped boundaries, non-finite values,
projection decisions inconsistent with the adaptive dot sign, or final OL1
ratios above the declared trust budget. It stores the source paths and SHA-256
hashes in `gradient_interaction.json`.

## Method

Raw gradient conflict is the boundary event

```text
<g_task, g_pressure> < 0.
```

For naive L1, the analysis also evaluates the alignment of the complete raw
gradient with the task gradient:

```text
<g_task, g_task + lambda * g_pressure>
  = ||g_task||^2 + lambda * <g_task, g_pressure>.
```

On boundaries with component conflict, the reported alignment offset is

```text
-lambda * <g_task, g_pressure> / ||g_task||^2.
```

An offset of `0.02`, for example, means that the conflicting component reduces
this raw task-alignment term by 2%. It does not measure an AdamW displacement or
an observed loss change.

For OL1, the adaptive interaction is computed before projection between

```text
d_task = m_hat / (sqrt(v_hat) + adam_eps)
d_pressure = g_pressure / (sqrt(v_hat) + adam_eps),
```

using task-only AdamW moments. A negative adaptive dot triggers projection.

## Results

The naive-L1 raw conflict fraction increased monotonically over the tested
lambda grid:

| lambda | negative raw dot | median raw cosine |
| ---: | ---: | ---: |
| 0.05 | 350/712 (49.16%) | +0.002505 |
| 0.1 | 387/712 (54.35%) | -0.017370 |
| 0.5 | 454/712 (63.76%) | -0.065280 |
| 1.0 | 505/712 (70.93%) | -0.105177 |

The recorded gradients are unweighted, so lambda does not directly alter the
dot sign or cosine at a fixed parameter state. All eight conditions have the
same initialization, and their first-boundary raw cosines agree within
`1.8e-8`. The later separation is a property of the resulting training
trajectories.

The complete naive-L1 raw gradient remained task-aligned on every boundary:
`<g_task, g_task + lambda * g_pressure>` was positive in all 2,848 cases. On
boundaries where the pressure component conflicted, its median alignment offset
was `0.169%`, `0.354%`, `1.504%`, and `2.675%` as lambda increased. The maximum
offset was `14.110%` at lambda `1.0`; no offset reached 100% and reversed the
combined raw direction.

In the OL1 runs, the task and pressure adaptive directions had a negative dot
before projection on `697/712` boundaries at lambda `0.05` and `711/712`
boundaries at each larger lambda. Projection was applied on exactly those
boundaries. Among projected cases, the largest absolute post-projection cosine
was `2.4e-8`, consistent with numerical residual. The trust cap was active on
`0`, `1`, `2`, and `76` boundaries across the same lambda grid.

## Interpretation

For the isolated pressure displacement
`delta_pressure = -alpha * lambda * g_pressure`, a first-order expansion gives

```text
L_task(theta + delta_pressure) - L_task(theta)
  = -alpha * lambda * <g_task, g_pressure> + O(alpha^2).
```

A negative raw dot therefore makes the pressure component locally oppose task
descent. The observation motivates separating the two objectives, but it does
not show that the full naive-L1 update raises task loss: the task component
dominated the combined raw direction throughout these runs.

Operational OL1 enforces a different and narrower property. It projects the
pressure direction onto the half-space whose dot product with the task-only
AdamW adaptive direction is nonnegative. Conflicting directions move to the
orthogonal boundary; compatible directions remain unchanged. In Euclidean
norm, this is the smallest change that satisfies the constraint. The trust
budget then bounds the correction norm relative to the task direction.

This is an AdamW-relative non-opposition guarantee, not a loss-safety theorem.
The adaptive task direction contains momentum and need not equal the current
task gradient; the geometry excludes decoupled weight decay; and a finite
orthogonal correction can affect loss through curvature. The method is also not
hyperparameter-free: lambda controls the requested correction and the trust
budget controls its cap.

## Caveats and nonclaims

- The conflict trend is descriptive evidence from one seed, one model scale,
  and one training recipe. It does not establish behavior in larger models.
- The 712 boundaries within a condition are temporally dependent observations,
  not independent replicates.
- The OL1 adaptive pre-projection statistics are measured on OL1 trajectories.
  They cannot be transferred to the naive-L1 trajectories, whose AdamW moments
  combine task and pressure.
- Historical gradient tensors were not retained. The analysis uses the stored
  scalar norms, dots, cosines, projection flags, and trust ratios.
- The projection is the minimum-change non-opposing direction in the declared
  adaptive geometry. It is not the globally largest pressure-reducing step that
  provably preserves training or validation loss.
- No finding or manuscript claim is promoted from this single-seed analysis.

## Provenance

- Reduction: `../03_gradient_interaction.py`
- Reduced artifact: `../gradient_interaction.json`
- Generated tables: `../gradient_tables.md`
- Endpoint comparison: `O001-r-model-vs-final-validation-loss.md`
