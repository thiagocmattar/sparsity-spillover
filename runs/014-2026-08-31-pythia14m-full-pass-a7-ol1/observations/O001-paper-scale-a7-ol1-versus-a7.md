# O001 - Paper-scale A7-OL1 versus matched A7

## Question

Does seven-site post-gate OL1 improve the Pythia-14M full-pass A7
quality--logical-opportunity frontier at matched `kappa`?

## Method and coverage

Run 014 independently trained five `A7-Z-POST` conditions at
`kappa in {0, 0.01, 0.05, 0.1, 0.5}` with `lambda=1` and trust budget `1`.
The seven pressure sites were `a,m,h,q_post,k_post,v,z` in all six layers.
Each condition completed 712 optimizer boundaries and 1,493,172,224 input
tokens from the same random initialization, realized data order, optimizer
schedule, and gate configuration as its Run 013 no-pressure comparator.

Every endpoint covers all 500 MiniPile validation documents, all 338 complete
2,048-token blocks, and 692,224 input tokens; the 1,444-token incomplete tail
is excluded and reported. `R_model` and exact-zero fractions divide pooled
integer counts from the complete workload. The cohort verifier hash-locks the
five Run 014 rows to the five verified Run 013 attempts.

Every optimizer boundary also verifies the realized 42-tensor pressure capture
(seven sites by six layers). Across all 3,560 boundaries there were no
nonfinite gradients, FP16 overflows, or skipped optimizer updates, and the
final pressure/task correction ratio never exceeded the trust budget.

## Result

| `kappa` | A7-OL1 loss | A7-OL1 `R_model` | A7 loss | A7 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.480184 | 7.0542% | 5.468401 | 7.2177% | +0.011783 | -0.1634 pp |
| 0.01 | 5.475797 | 7.7234% | 5.458822 | 7.6176% | +0.016975 | +0.1057 pp |
| 0.05 | 5.462811 | 9.8630% | 5.437888 | 9.1269% | +0.024923 | +0.7361 pp |
| 0.1 | 5.429497 | 11.7968% | 5.428681 | 10.4250% | +0.000816 | +1.3717 pp |
| 0.5 | 5.829390 | 27.4827% | 5.702923 | 15.3868% | +0.126466 | +12.0959 pp |

The effect is threshold-dependent. OL1 slightly reduces `R_model` at
`kappa=0`, and the `kappa=0.01` and `0.05` OL1 rows are not new points on the
combined A7/A7-OL1 frontier. At `kappa=0.1`, however, OL1 adds 1.3717
percentage points of logical opportunity for only `+0.000816` validation
loss. At `kappa=0.5`, it adds 12.0959 points for a much larger `+0.126466`
loss. The `kappa=0.1` and `0.5` A7-OL1 rows are therefore new nondominated
points among the ten matched A7/A7-OL1 endpoints.

The added attention-site exact-zero fractions show how strongly the largest
threshold changes the endpoint:

| `kappa` | `q_post` | `k_post` | `v` |
| ---: | ---: | ---: | ---: |
| 0 | 0.0000% | 0.0000% | 0.0000% |
| 0.01 | 1.3726% | 1.4956% | 2.3376% |
| 0.05 | 5.4275% | 5.8171% | 10.5257% |
| 0.1 | 8.2275% | 8.5445% | 18.8192% |
| 0.5 | 93.5450% | 94.5413% | 98.7133% |

OL1 conflict was observed on 352, 372, 370, 364, and 420 of 712 boundaries
as `kappa` increased. Projection was applied on 709--712 boundaries, and all
but 107 `kappa=0.5` boundaries saturated the trust cap. These are measured
training-time interactions, not checkpoint reconstructions.

## Interpretation

This one-seed result partially supports the scoped hypothesis. Seven-site OL1
does not provide a uniform matched-threshold gain, but it creates a useful
moderate-threshold endpoint at `kappa=0.1` and substantially extends the
high-opportunity frontier at `kappa=0.5` with an explicit quality cost. The
result refutes a blanket claim that OL1 improves every A7 threshold.

## Caveats

- One seed and one Pythia-14M scale do not establish replication or scaling.
- All seven pressure sites change jointly, so site-specific causal effects are
  not identified.
- Thresholded training changes forward values and gradient support; matching
  `kappa` does not hold the realized pressure magnitude fixed.
- The Run 012 four-site OL1 implementation captured only `h` despite declaring
  four pressure sites. This does not affect the direct Run 013/014 comparison,
  but it limits later P09/P10/P16 synthesis until Run 012 is audited or rerun.
- `R_model` is a logical zero-operand opportunity, not removed FLOPs or
  measured sparse-kernel acceleration.

## Sources

- Run 014 cohort verification: `artifacts/verification.json`
- Per-attempt integer counters: `artifacts/attempts/*/diagnostics/logical_products.json`
- Per-attempt activation counts: `artifacts/attempts/*/diagnostics/activation_statistics.json`
- Per-boundary OL1 records: `artifacts/attempts/*/events.jsonl`
- Matched A7 source: `runs/013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json`
- Generating verifier: `03_verify.py` and `verification.py`
- Transfer and teardown provenance: `prelaunch/scientific-transfer-closeout.json`

No PDF figure was generated for this run-local observation. A paper-facing
frontier figure, if selected, belongs in a separately approved numbered
analysis.
