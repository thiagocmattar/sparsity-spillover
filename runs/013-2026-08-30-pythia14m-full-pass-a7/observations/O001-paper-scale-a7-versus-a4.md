# O001 - Paper-scale A7 versus matched A4

## Question

Does adding symmetric post-RoPE `q_post`, `k_post`, and `v` threshold gates to
the one-sided A4 topology improve the Pythia-14M full-pass
quality--logical-opportunity frontier at matched `kappa`?

## Method and coverage

Run 013 independently trained five `A7-Z-POST` conditions at
`kappa in {0, 0.01, 0.05, 0.1, 0.5}` from the same random initialization and
realized data order used by Run 011 A4. Each condition completed 712 optimizer
boundaries and 1,493,172,224 input tokens. Every reported endpoint covers all
500 MiniPile validation documents, all 338 complete 2,048-token blocks, and
692,224 input tokens; the 1,444-token incomplete tail is excluded and reported.

`R_model` is recomputed from pooled integer logical-product counts over the
complete validation workload. Exact-zero site fractions likewise divide
pooled counts, not averaged percentages. The A7/A4 comparison is hash-locked by
the Run 013 cohort verifier to Run 011's verified attempts.

## Result

| `kappa` | A7 loss | A7 `R_model` | A4 loss | A4 `R_model` | A7 loss delta | A7 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.468401 | 7.2177% | 5.470497 | 7.2120% | -0.002096 | +0.0056 pp |
| 0.01 | 5.458822 | 7.6176% | 5.466500 | 7.4137% | -0.007678 | +0.2039 pp |
| 0.05 | 5.437888 | 9.1269% | 5.434110 | 8.2059% | +0.003778 | +0.9210 pp |
| 0.1 | 5.428681 | 10.4250% | 5.419642 | 8.9530% | +0.009038 | +1.4720 pp |
| 0.5 | 5.702923 | 15.3868% | 5.659680 | 10.2155% | +0.043244 | +5.1713 pp |

The `kappa=0` A7 and A4 endpoints are close, as expected when the added
symmetric gates are identity operators. At `kappa=0.01`, A7 is better on both
measured axes. From `kappa=0.05` upward, the additional attention gating yields
increasing logical opportunity at increasing validation-loss cost. Within A7,
`kappa=0.1` has the lowest loss, while `kappa=0.5` reaches the largest measured
`R_model` and realizes 51.37% of the 29.9524% A7 analytic ceiling.

The final count-pooled exact-zero fractions at the three added sites rise
monotonically:

| `kappa` | `q_post` | `k_post` | `v` |
| ---: | ---: | ---: | ---: |
| 0 | 0.0000% | 0.0000% | 0.0000% |
| 0.01 | 0.3985% | 0.4066% | 1.6544% |
| 0.05 | 1.8397% | 1.7161% | 7.3125% |
| 0.1 | 3.1501% | 3.2271% | 11.3550% |
| 0.5 | 17.4347% | 18.1338% | 31.2552% |

## Interpretation

This one-seed result supports the scoped hypothesis that the post-RoPE
attention gates can extend the A4 logical-opportunity frontier: the smallest
positive threshold improves both reported axes, and larger thresholds provide
clear opportunity gains with explicit quality costs. It does not establish
that every A7 dose dominates A4, attribute the effect to Q, K, or V
individually, or demonstrate runtime speedup.

## Caveats

- One seed and one Pythia-14M scale do not establish replication or scaling.
- The three attention gates are added jointly, so site-specific causal effects
  are not identified.
- Thresholded training changes both forward values and gradient support.
- Independently scheduled GPUs share scientific identities but are not assumed
  bitwise replicas.
- `R_model` is a logical zero-operand opportunity, not removed FLOPs or measured
  sparse-kernel acceleration.

## Sources

- Run 013 cohort verification: `artifacts/verification.json`
- Per-attempt integer counters: `artifacts/attempts/*/diagnostics/logical_products.json`
- Per-attempt activation counts: `artifacts/attempts/*/diagnostics/activation_statistics.json`
- Matched A4 source: `runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json`
- Generating verifier: `03_verify.py` and `verification.py`
- Transfer and teardown provenance: `prelaunch/scientific-transfer-closeout.json`

No PDF figure was generated for this run-local observation. A paper-facing
frontier figure, if selected, belongs in a separately approved numbered
analysis.
