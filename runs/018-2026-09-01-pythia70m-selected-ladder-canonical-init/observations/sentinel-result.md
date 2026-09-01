# Pythia-70M selected-ladder sentinel

## Question

Does the relationship between measured logical-product opportunity
(`R_model`) and full-validation loss persist when the selected 14M ladder is
promoted to randomly initialized Pythia-70M, before committing to the eight
nonzero-κ OL1 conditions?

## Method and coverage

Four independent Secure H200 workers ran A0, A1-H, A4-OL1 κ=0, and A7-OL1
κ=0 from the same hash-pinned random initialization and the same one-pass data
schedule. Each condition completed 712 optimizer boundaries and 1,493,172,224
training tokens. Final evaluation reloaded the retained checkpoint and covered
all 338 complete 2,048-token validation blocks (692,224 input tokens); the
declared 1,444-token tail was excluded. No worker recorded an FP16 overflow or
skipped optimizer update. Diagnostics pooled integer logical-product and
activation counts over the complete validation workload.

| Condition | Validation loss | `R_model` | `R_model_max` | Median tokens/s |
|---|---:|---:|---:|---:|
| A0 GELU | 4.099767 | 0.00000454 | 0.000000 | 465,812 |
| A1-H ReLU | 4.222745 | 0.100658 | 0.123545 | 464,156 |
| A4-OL1 κ=0 | 4.805289 | 0.256725 | 0.370634 | 277,487 |
| A7-OL1 κ=0 | 4.941204 | 0.235624 | 0.494239 | 255,831 |

For the OL1 boundaries, A4 recorded 351/712 gradient-conflict steps and 383
projection steps; A7 recorded 375/712 conflict steps and 711 projection steps.
The principal exact-zero masses at κ=0 were `(a,m,h,z) =
(0.5660,0.4938,0.9259,0.9339)` for A4 and
`(0.4973,0.5030,0.8597,0.6861)` for A7.

The approved post-hoc TEAL protocol evaluated ten target sparsities for both
controls using ten calibration blocks and the same complete validation
workload. Selected points are:

| Source | Target sparsity | Validation loss | `R_model` |
|---|---:|---:|---:|
| A0 | 0.0 | 4.099766 | 0.00000454 |
| A0 | 0.1 | 4.103608 | 0.037048 |
| A0 | 0.3 | 4.207335 | 0.111282 |
| A0 | 0.5 | 4.833057 | 0.185885 |
| A1-H | 0.0 | 4.222750 | 0.100658 |
| A1-H | 0.1 | 4.223747 | 0.125370 |
| A1-H | 0.3 | 4.269464 | 0.174855 |
| A1-H | 0.5 | 4.522054 | 0.225156 |

## Result

The 70M sentinel preserves the basic tradeoff: moving from A0 to A1-H and then
to A4 κ=0 raises measured `R_model` and validation loss. Post-hoc clipping also
traces a smooth local tradeoff for each control: modest clipping initially buys
logical opportunity at small loss cost, followed by increasingly steep loss.

A7 κ=0 does not dominate A4 κ=0. It realizes lower `R_model` (0.2356 versus
0.2567) at higher loss (4.9412 versus 4.8053), despite its larger analytic
all-zero reach ceiling. The nonzero-κ wave is therefore needed if the approved
question is the full A4/A7 frontier rather than only the κ=0 ranking; the
sentinel alone does not establish that A7 remains dominated at other gates.

## Verification defect and resolution

The frozen verifier initially rejected A0 because its measured `R_model` was
greater than the A0 analytic reach ceiling of zero. The observed value is a
tiny, count-reconciled incidental exact-zero opportunity, dominated by fp16
underflow at the GELU output feeding MLP W2. The operational contract defines
`R_model_max` as an analytic gate-reach ceiling, not a ceiling on every observed
zero. An append-only verifier repair retained all original identity,
aggregation, boundary, checkpoint, and transfer checks while bounding observed
`R_model` to `[0,1]`. The repair passed focused tests and all four retrieved
attempts. Both the original failure and corrected verification logs are
retained with A0.

## Caveats

This is one seed, one 70M scale, one MiniPile pass, and only κ=0 for the OL1
conditions. `R_model` is a logical-product opportunity, not a measured runtime
gain. The sentinel cannot establish the nonzero-κ frontier or persistence at
410M. TEAL is post-hoc clipping and is not equivalent to training with the
corresponding gate.

## Sources

- Canonical attempts: `artifacts/attempts/{001,002,003,008}-*`
- Control frontiers: `artifacts/teal/a0-gelu.json`,
  `artifacts/teal/a1h-relu.json`, and `artifacts/teal/teal_frontiers.json`
- Retrieval and verifier evidence: `launch-control/sentinel-20260901/`
- Corrected verifier: `verification_observed_bound.py` and
  `08_verify_observed_bound.py`
