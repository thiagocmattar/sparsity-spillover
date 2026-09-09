# O001: Quality constraints and counted opportunity

Question: Which evaluated points offer the greatest sparsity under explicit
same-size A0 loss allowances, and what explains the local/model-wide ranking?

Method: `01_audit_training.py` reconstructs Analysis 018 scientific data from
the original sources, verifies exact equality excluding live manuscript source
fingerprints, checks integer operation sums, and recomputes all 29 paired and
15 cross-size differences. It considers all 54 canonical endpoints and all 540
actual Run 030 clipping evaluations. It computes retrospective .05/.10/.20 loss
budgets from full precision, without interpolation. `04_make_tables.py` formats
the retained and derived tables. Exact record IDs and hashes are in
`training-audit.json` and its source inventory.

Coverage: all500 MiniPile validation documents, 338 complete2048-token blocks,
692224 input positions and691886 next-token targets; excluded tail1444.
Canonical loss/count passes use FP16. There is one seed per training condition,
712 updates and1493172224 training input tokens. Counts pool before division.

Results: the best14M local endpoint is naive L1 at lambda1: loss5.10227534 and
3.949334% sparsity. A7-OL1 kappa.5 reaches27.482684% but costs+.62083357 loss
versus A0. Under a+.05 budget, all-evaluated maxima are6.490231%,7.4099% and
14.8833% at14M/70M/410M; all use clipping. No sparsified trained70M/410M
endpoint qualifies at+.05 or+.10. Empty regimes remain explicit.

The14M kappa.5 A4-OL1 and A7-OL1 operation contributions are respectively
12.656881/10.709399 projection points and.056568/16.773285 attention points.
The added attention contribution outweighs reduced projection contributions.
For all 54 canonical endpoints, S_model/R_arch(A7) equals S_block exactly to
floating-point tolerance under the declared graph. Architectural reach's
integer numerator and denominator are unchanged by its manuscript rename.

Numerical sensitivity: actual p=0 reruns differ from canonical endpoints by at
most5.51685e-5/1.80712e-4/1.69217e-4 loss at14M/70M/410M and
.00027111/.00033215/.00090735 sparsity percentage points. The audit records
strict and conservative dominance separately. Canonical trained points dominate
A0's p>=.5 clipping points beyond these observed drifts on both axes at all
sizes. Small-p results must not be collapsed into a universal dominance claim.

Caption/legend: quality-budget tables identify the A0 reference, allowed loss
increment, recipe/parameter and clipping target. Operation-summary columns use
the common full-model denominator, including a dense head with no zero credit.
Neither table represents runtime. Revised overview/scale captions are in O004.

Caveats: budgets and drift sensitivity are retrospective presentation choices.
Drift is not a confidence interval or seed variability. No latency is available
for the clipped winners. Complete-recipe comparisons change pressure targets
and old-term weighting; no fixed-coefficient experiment was run.
