# O002 - OL1 task-pressure geometry and budget saturation

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

Panel (a) plots the logged `task_pressure_cosine_before` and
`task_pressure_cosine_after` directly, with no layer averages, subsampling,
jitter, or replacement of measured residuals by zero. Conflict is counted
using `task_pressure_dot_before < 0`, not the separate raw-gradient conflict
diagnostic. The logged cosines use the implementation's stabilized denominators.

Panel (b) plots `pressure_to_task_ratio_raw / step_budget`, where
r = lambda * norm(w_tilde) / (norm(u) + epsilon). Cap activity is counted
from the actual `trust_scale < 1`. The implementation uses
s = min(1, b / (r + epsilon)) for positive r; the script verifies this rule.
It also recovers norm(w_tilde) from the recorded ratio to check consistency
of the post-projection dot product and cosine. The measured post-projection
cosine is retained, including numerical residuals.

Thin solid traces denote four-site conditions, and thin dashed traces denote
seven-site conditions. The thick dark line is the arithmetic median across
all ten conditions at each optimizer step, without smoothing. No uncertainty
band is shown. The y=1 reference marks the nominal budget boundary. Panel (a)
includes x=0, y=0 and y=x references. No threshold-specific colors are used.

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

Conflict occurs on 99.87% of observations. Its median pre-projection cosine is
-0.01737: conflict is frequent, but the typical negative alignment is small.
Under conflict, the maximum absolute post-projection cosine is 1.46e-8.
The nine non-conflicting observations retain exactly the same logged cosine
before and after projection. Their positive cosines are tiny (at most
0.000461), so they cluster at the origin at the displayed scale.

Budget activity differs sharply by site set. Four-site conditions are almost
always below budget; their five cap counts are 3, 7, 5, 2, and 1 out of 712,
in increasing threshold order. Seven-site conditions are capped at all 712
steps for thresholds 0 through 0.1 and at 605/712 steps for threshold 0.5.
The pooled median r/b across all observations is 0.8463. This is a different
summary from the time-varying median trace.

These results support frequent adaptive-direction conflict and frequent
seven-site saturation. They do **not** support the proposed blanket statement
that lambda = 1 saturates all multisite conditions. This new qualification was
reported to the author; no manuscript wording or consolidated finding has
been changed.

## Caption

**OL1 encounters conflicting pressure, with budget saturation concentrated
in the seven-site conditions.** (a) Logged cosine similarity between the
adaptive task direction u and preconditioned pressure direction w, before
and after conflict-conditioned projection. Conflict occurs on 99.9% of
optimizer steps, with median conflicting cosine -0.017; projection makes the
directions approximately orthogonal when they conflict. (b) Pre-cap norm
ratio r/b over training at lambda = b = 1. Thin solid/dashed lines denote the
five four-/seven-site conditions; the thick line is the pooled median across
conditions at each step. The cap is active on 48.8% of steps overall: 0.5%
at four sites and 97.0% at seven sites. All ten Pythia-14M conditions and all
712 optimizer steps per condition are included. Quantities pool OL1's eligible
parameters and are defined before group learning rates, excluding decoupled
weight decay. They diagnose pressure geometry and budget activity, not
guaranteed task-loss descent or preservation.

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
- The two families have very different norm-ratio distributions. Their pooled
  median can lie between the families, where no individual run lies. It is a
  summary across conditions, not a representative trajectory or fitted model.
- Historical raw-gradient conflict fields describe a different geometry from
  the AdamW-preconditioned quantities shown here. L1N has no equivalent
  separately constructed update pair in these runs and is excluded.
- Four focused tests verify full coverage and pooled counts, the stabilized
  cap boundary, rejection of inconsistent logs, and every plotted coordinate
  and unsmoothed trace. The PDF is rendered for visual inspection; source
  logs and immutable run artifacts are unchanged.
