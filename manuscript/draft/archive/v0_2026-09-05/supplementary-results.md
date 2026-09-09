# Evidence included in the rewritten manuscript

The authoritative reduction is
[Analysis 013](../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md).
Its [complete numeric table](../../analyses/013-2026-09-04-matched-intervention-manuscript/tables.md)
and [figure data](../../analyses/013-2026-09-04-matched-intervention-manuscript/figure_data.json)
retain values, counts, source identities, and selected operating points.

## Main-text selection

- Figure 1: matched 14M local pressure, fixed-gate pressure scope, and A4/A7 branches.
- Table 1: gate set, pressure set, and complete update recipe for every condition.
- Table 2: all five paired A4/A7 and within-configuration pressure effects.
- Figure 2: full trained grids and uniform-clipping paths at 14M/70M.
- Table 3: maximum opportunity under five explicit quality allowances.
- Figure 3: actual activation zeros and operation contributions at high dose.
- Main runtime paragraph: completed correctness checks and measured slowdown,
  with the complete results in the appendix.

## Complete appendix coverage

The appendix includes the exact activation sites, count denominator, OL1 rule,
data identities, precision and schedule, all 35 trained 14M conditions,
12 selected 70M conditions, 12 selected 410M conditions, and all 60 clipping
evaluations. The two larger-scale control rows also serve as zero-target
clipping references; rows are not presented as independent experimental units.
The separate Run 021 LR screen contributes three selection rows, including
the reused Run 019 A0 baseline. It does not inflate the main study's 59 count.

The 410M section preserves its full absolute frontier, reversed dose response,
quality gap to A0, and lower tokens-per-parameter exposure. A common token
budget leaves its optimization maturity unresolved. All six Run 023 runtime
sentinels appear with both batches and operation coverage. No unfavorable
410M or sentinel condition is removed.

## Material excluded from the main argument

The analytic selected-site reach ceiling, exploratory near-zero spillover
plots, raw gradient-norm curves, and OL1 conflict trajectories are left in
their owning run/analysis records. They answer secondary questions and would
diffuse this paper's intervention-centered contribution. No archived evidence
is deleted or relabeled as a different intervention.
