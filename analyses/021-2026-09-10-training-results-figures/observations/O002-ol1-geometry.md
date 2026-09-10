# O002 - OL1 conflict geometry and target-set-dependent saturation

## Question

Does OL1 encounter conflicting adaptive task and pressure directions during
the Pythia-14M multisite experiments, remove the opposing component, and reach
its norm budget at lambda = 1? This is the author-requested mechanism check,
not a comparison of validation quality or an L1N-versus-OL1 optimizer study.

## Sources and coverage

- [Generating script](../02_ol1_geometry.py).
- [Figure PDF](../figures/02-14m-ol1-geometry.pdf).
- [Machine-readable summary and provenance](../data/14m-ol1-geometry.json).
- [Analysis 018 cohort](../../018-2026-09-08-results-materials/figure_data.json):
  all 14M A4-OL1 and A7-OL1 conditions, without selecting by outcome.
- [Corrected four-site Run 015](../../../runs/015-2026-08-31-pythia14m-corrected-a4-ol1/README.md)
  and [seven-site Run 014](../../../runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/README.md):
  the ten selected attempt directories contain the tracked `events.jsonl`
  and `manifest.json` files identified and hashed in the summary.

There are ten conditions: kappa in {0, 0.01, 0.05, 0.1, 0.5} at each site set,
with lambda = 1, budget b = 1, and projection/budget epsilon = 1e-12. Each
condition has all 712 optimizer steps, giving 7,120 step-condition observations.
None has a skipped optimizer step or gradient overflow. Four-site pressure
captures a,m,h,z in all six layers (24 tensors); seven-site pressure adds
post-RoPE Q/K and V (42 tensors). These counts describe activation captures;
the global optimizer geometry pools 69 eligible parameter tensors and skips
seven in every record. Corrective Run 015 is used, not the mistargeted Run 012.

The models were pretrained from random initialization with model/data-order
seed 1234 and a common 712-update budget (1,493,172,224 input tokens per
condition). This figure reads training logs only; validation evaluations and
final checkpoints do not enter the calculation.

## Method

The executed [OL1 implementation](../../../src/sparsity_research/pressure.py)
constructs u from AdamW's bias-corrected task moments and w by preconditioning
the unweighted pressure gradient with the task second moment. It pools dot
products and squared norms over all eligible parameters. Projection occurs
only if the global dot product is negative and the squared task norm exceeds
epsilon. All negative-dot observations in this cohort satisfy that guard.

Panel (a) plots median logged `task_pressure_cosine_before` at each kappa,
separately for four- and seven-site conditions. Each point summarizes all 712
training steps in that condition; thin bars span the 25th to 75th percentiles.
These are empirical training-step intervals, not confidence intervals or
uncertainty across seeds. The five thresholds occupy equally spaced categorical
positions, with slight horizontal offsets to keep overlapping bars readable.
Connecting lines organize evaluated settings without interpolating models.
There is no binning, smoothing, layer averaging or subsampling. Conflict is counted
using `task_pressure_dot_before < 0`, not the separate raw-gradient conflict
diagnostic. The logged cosines use the implementation's stabilized denominators.
Post-projection cosines remain in the numerical audit and caption rather than
occupying a plot dimension.

Panel (b) plots `pressure_to_task_ratio_raw / step_budget`, where
r = lambda * norm(w_tilde) / (norm(u) + epsilon). Cap activity is counted
from the actual `trust_scale < 1`. The implementation uses
s = min(1, b / (r + epsilon)) for positive r; the script verifies this rule.
It also recovers norm(w_tilde) from the recorded ratio to check consistency
of the post-projection dot product and cosine. The measured post-projection
cosine is retained, including numerical residuals.

The additional diagnostic is rho_opp = max(0, -c) / (q + epsilon), where
c is `task_pressure_dot_before` and q is reconstructed by squaring the logged
`task_direction_norm`. It measures the norm of the raw pressure component
removed along u, relative to norm(u), before the cap and learning rate. For a
conflicting step, the implemented projection is w_tilde = w + rho_opp * u;
thus norm(w_tilde - w) / norm(u) equals rho_opp, up to numerical rounding.
All cohort task norms satisfy the projection guard q > epsilon, which the
script checks. Aligned steps contribute zero to this diagnostic. Its summaries
include all steps, rather than only conflicting steps.

This uses the raw pre-projection dot product directly. Multiplying the plotted
ratio r/b by the negative cosine would mix post-projection and pre-projection
norms. Lambda is 1 here; rho_opp describes the unweighted raw direction and
does not include the subsequent trust scale. Quantiles and counts above one
are retained by family and threshold in the machine-readable summary.

Blue denotes four-site conditions and orange denotes seven-site conditions
throughout. In panel (b), five thin traces show the individual conditions
within each family; a thick line shows their median at each optimizer step,
without smoothing. The y=1 dotted reference is directly labeled "budget binds."
Panel (a) uses blue circles and orange diamonds and marks zero cosine with a
horizontal dotted reference. There is no
pooled trajectory or cap annotation, uncertainty band, threshold-specific
color, legend or super-title. The revised 5.9-by-2.45-inch layout gives slightly
more width to the budget panel. The former pooled ECDF figure is preserved
in commit `b2f46d0`, the author-requested rollback point.

Before reduction, the script verifies complete step coverage, pressure identity,
and eight distinct historical optimizer/training source files against the run
manifest hashes, respecting recorded LF canonicalization. The source code used
to interpret the logs matches the executed code. All required historical scalars
are present; no retrospective reconstruction from final weights is attempted.

## Results

| Scope | Observations | Conflicting steps | Cap-active steps | Cap-active fraction |
| --- | ---: | ---: | ---: | ---: |
| Four-site | 3,560 | 3,556 | 18 | 0.51% |
| Seven-site | 3,560 | 3,555 | 3,453 | 96.99% |
| Pooled | 7,120 | 7,111 | 3,471 | 48.75% |

Conflict occurs on 99.89% of four-site observations and 99.86% of seven-site
observations (both round to 99.9%). The median pre-projection cosines are
-0.03459 and -0.01157, respectively; their interquartile ranges are
[-0.05155, -0.02009] and [-0.01548, -0.00731]. These pooled statistics do not
describe the threshold dependence now exposed by panel (a):

| Kappa | Four-site median cosine | Seven-site median cosine |
| ---: | ---: | ---: |
| 0 | -0.02321 | -0.01210 |
| 0.01 | -0.02718 | -0.01393 |
| 0.05 | -0.03741 | -0.01294 |
| 0.1 | -0.04767 | -0.01133 |
| 0.5 | -0.02168 | -0.00281 |

Seven-site pressure has a less negative median cosine at every matched
threshold. Four-site medians become more negative through kappa=0.1, then
return toward zero at 0.5. Seven-site medians remain around -0.012 through 0.1,
then approach zero at 0.5. That last condition still has a wide IQR
[-0.04958, -0.00178]; its near-zero median does not describe the entire run.

The directions are geometrically close to orthogonal but systematically
negatively aligned. Cosine alone does not establish that practical conflict
is weak: it omits the raw pressure-to-task norm ratio. No random-direction
reference distribution is established by this analysis.

The pooled median pre-projection cosine is -0.01736 (-0.01737 when restricted
to conflicting steps). Under conflict, the 99th percentile of the absolute
post-projection cosine is 7.39e-9, and its maximum is 1.46e-8.
The nine non-conflicting observations retain exactly the same logged cosine
before and after projection. Their positive cosines are tiny (at most
0.000461), and remain included in the per-condition summaries.

Budget activity differs sharply by site set. Four-site conditions are almost
always below budget; their five cap counts are 3, 7, 5, 2, and 1 out of 712,
in increasing threshold order. Seven-site conditions are capped at all 712
steps for thresholds 0 through 0.1 and at 605/712 steps for threshold 0.5.
Median r/b across all observations within each family is 0.2533 at four
sites and 57.4851 at seven sites. These are different summaries from the
two time-varying within-family median traces. The original pooled counts
remain in the numerical record, but the revised figure omits their cap rate
and trajectory because they obscure the target-set difference.

These results support frequent adaptive-direction conflict and frequent
seven-site saturation. They do **not** support the proposed blanket statement
that lambda = 1 saturates all multisite conditions. This new qualification was
reported to the author. The ideal correction is independent of further
increases in lambda only while the cap binds. Thus the draft's statement
motivating a globally "saturated lambda" needs qualification; conditional
mathematical saturation and the guaranteed relative norm budget remain valid.
This figure revision does not edit manuscript wording or consolidated findings.

## Opposing-component diagnostic

The diagnostic separates the target sets much more sharply than raw cosine.
Values below are medians and interquartile ranges across the 712 steps of
each condition; they are fractions of the task-direction norm.

| Kappa | Four-site median rho_opp | Four-site IQR | Seven-site median rho_opp | Seven-site IQR |
| ---: | ---: | --- | ---: | --- |
| 0 | 0.01056 | [0.00808, 0.01469] | 0.46594 | [0.36871, 0.91241] |
| 0.01 | 0.01211 | [0.00848, 0.01563] | 0.52505 | [0.43554, 0.87687] |
| 0.05 | 0.01122 | [0.00976, 0.01287] | 0.66841 | [0.54357, 1.01275] |
| 0.1 | 0.00909 | [0.00819, 0.01011] | 0.84784 | [0.74267, 1.19236] |
| 0.5 | 0.00248 | [0.00203, 0.00267] | 0.19781 | [0.10060, 0.24482] |

Pooled within each family, median rho_opp is 0.009503 at four sites
(IQR 0.007007-0.012623) and 0.588006 at seven sites (IQR 0.365727-0.887890):
about a 62-fold median difference. In task-relative norm units, these are
0.95% and 58.80%. Rho_opp exceeds one on 753/3,560 seven-site steps (21.15%)
and on no four-site steps. It is larger at seven sites at every matched
threshold, despite the less negative median cosine.

This establishes a substantial difference in the raw opposing component
removed by projection. It does not quantify the opposing component of the
applied, capped update, nor the validation benefit of projection. A hypothetical
update without projection would also change the norm supplied to the cap.
The requested primary figure keeps cosine versus threshold and budget versus
step; rho_opp is retained here for the author's next choice of presentation.

## Caption

**Task and pressure directions are systematically negatively aligned;
their angular relationship varies with threshold and their norm-budget
regime differs sharply by target set.** (a) Median pre-projection cosine
between adaptive task direction u and preconditioned pressure direction w
across all 712 steps at each threshold. Blue circles/orange diamonds denote
four-/seven-site conditions; bars show the empirical 25th-75th percentiles
over training steps, not confidence intervals. Thresholds occupy categorical
positions; lines connect evaluated settings. Seven-site medians are less
negative at every matched threshold, while 99.9% of step-level alignments
are negative. Under conflict, projection reduces the task-aligned component
to numerical tolerance (99th-percentile absolute post-projection cosine 7.39e-9).
(b) Pre-cap norm ratio r/b at lambda = b = 1. Thin lines show individual
conditions; thick lines show the median within each target set at each step.
The cap is active on 0.5% of four-site steps and 97.0% of seven-site steps.
All 712 steps from each of five thresholds per target set are included
(7,120 observations, Pythia-14M). Quantities pool OL1's eligible parameters
before group learning rates and exclude decoupled weight decay; they do not
guarantee task-loss descent or preservation.

## Caveats and verification

- This is one training seed with correlated adjacent steps, not 7,120
  independent experimental replicates. Percentages describe observed steps.
- All ten first steps have learning rate zero, while Adam moments and the
  pre-learning-rate geometry are still computed. They remain in the requested
  geometry scope; cap activity does not imply a nonzero parameter correction
  on those ten steps.
- Projection residuals are an implementation check, not independent evidence
  of improved loss. These diagnostics do not establish a validation benefit,
  task-loss preservation, or equivalence to ordinary L1 pressure.
- Panel (a) summarizes steps within each condition; panel (b) summarizes five
  conditions at each step. Their medians answer different questions.
  Gate placement and pressure
  targets both change between these families, so their contrast does not
  isolate pressure target selection at an otherwise fixed architecture.
- Historical raw-gradient conflict fields describe a different geometry from
  the AdamW-preconditioned quantities shown here. L1N has no equivalent
  separately constructed update pair in these runs and is excluded.
- Five focused tests verify full coverage and pooled counts, the stabilized
  cap boundary, rejection of inconsistent logs, the opposing-component formula
  against a vector projection, and all per-threshold quantiles and unsmoothed
  per-family traces. All eight Analysis
  021 tests pass. The PDF was rendered and checked for text bounds and embedded
  fonts; source logs and immutable run artifacts are unchanged.
