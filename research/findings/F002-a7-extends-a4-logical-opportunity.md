# F002 - A7 extends A4 logical opportunity with dose-dependent quality cost

Status: tentative

Approved: 2026-08-31

## Statement

For one matched seed and one full MiniPile pretraining pass at Pythia-14M,
expanding the operational `A4-Z = {a,m,h,z}` topology with symmetric threshold
gates at post-RoPE `q_post`, post-RoPE `k_post`, and `v` produces a
dose-dependent change in the quality--logical-opportunity tradeoff. The
resulting `A7-Z-POST` topology is a near-null expansion at `kappa=0`. At
`kappa=0.01`, A7 improves complete-validation loss by 0.007678 and measured
`R_model` by 0.2039 percentage points relative to matched A4. At
`kappa in {0.05, 0.1, 0.5}`, A7 increases `R_model` by 0.9210, 1.4720, and
5.1713 percentage points, respectively, with validation-loss costs of
0.003778, 0.009038, and 0.043244.

This is evidence that thresholding the tested attention operands can extend
the logical-opportunity range beyond A4, with an explicit dose-dependent
quality cost. It is not evidence that A7 uniformly dominates A4, that any one
of Q, K, or V causes the effect independently, or that the additional logical
opportunity produces runtime speedup.

## Evidence

The finding covers five matched A4/A7 pairs. Each condition trained from the
same random initialization and realized data order for 712 optimizer
boundaries. Every endpoint uses all 338 complete 2,048-token blocks from all
500 MiniPile validation documents (692,224 input tokens); the 1,444-token
incomplete tail is excluded and reported.

| `kappa` | A4 loss | A4 `R_model` | A7 loss | A7 `R_model` | A7 loss delta | A7 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | 7.2120% | 5.468401 | 7.2177% | -0.002096 | +0.0056 pp |
| 0.01 | 5.466500 | 7.4137% | 5.458822 | 7.6176% | -0.007678 | +0.2039 pp |
| 0.05 | 5.434110 | 8.2059% | 5.437888 | 9.1269% | +0.003778 | +0.9210 pp |
| 0.1 | 5.419642 | 8.9530% | 5.428681 | 10.4250% | +0.009038 | +1.4720 pp |
| 0.5 | 5.659680 | 10.2155% | 5.702923 | 15.3868% | +0.043244 | +5.1713 pp |

Deltas are A7 minus matched A4, so a negative loss delta is favorable.

Within A7, the lowest validation loss is 5.428681 at `kappa=0.1` and
`R_model=10.4250%`. The `kappa=0.5` endpoint reaches the largest measured
`R_model`, 15.3868%, while raising loss to 5.702923. Exact-zero mass at the
three added attention-operand sites rises monotonically from effectively zero
at `kappa=0` to 17.4347% at `q_post`, 18.1338% at `k_post`, and 31.2552% at
`v` for `kappa=0.5`. Analysis 008 retains the full count-pooled eight-site
table.

## Sources and provenance

- Primary analysis: `analyses/008-2026-08-31-full-pass-frontier-with-a7/`
- Primary observation:
  `analyses/008-2026-08-31-full-pass-frontier-with-a7/observations/O001-full-pass-frontier-with-a7.md`
- Generating script:
  `analyses/008-2026-08-31-full-pass-frontier-with-a7/01_build.py`
- Machine-readable reduction:
  `analyses/008-2026-08-31-full-pass-frontier-with-a7/figure_data.json`
- A4 source:
  `runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json`
- A7 source:
  `runs/013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json`

Analysis 008 recomputes every Run 013 `R_model` from integer logical-product
counts and every reported site fraction from pooled integer activation counts
before division. Run 013's cohort verifier hash-locks the five A7 attempts and
their matched Run 011 A4 sources.

## Definitions

- `A4-Z` activates one-sided threshold gates at `a`, `m`, `h`, and `z`, where
  `z` is the concatenated attention context immediately before `W_o`.
- `A7-Z-POST` retains those four gates and adds symmetric thresholds at
  post-RoPE `q_post`, post-RoPE `k_post`, and `v`. One common `kappa` applies
  at all seven sites, and no sparsity-pressure objective is used.
- `R_model` is the count-first fraction of model-wide logical scalar products
  with at least one exactly-zero operand, including the dense LM head in its
  denominator. It is an opportunity measure, not removed FLOPs or speedup.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scaling behavior.
- The three attention gates are introduced jointly, so individual Q/K/V
  effects are not identified.
- Thresholded training changes both forward values and gradient support.
- Separately scheduled GPUs are matched by scientific identities but are not
  bitwise replicas.
- Connected frontier lines describe dose order; they are not fitted response
  curves.
- No sparse kernel or wall-clock acceleration was evaluated.

## What would change the conclusion

The finding should be strengthened if additional matched seeds reproduce the
dual-axis improvement at `kappa=0.01` and the opportunity--quality tradeoff at
larger thresholds, and if larger Pythia scales show the same paired direction.
It should be narrowed or discarded if matched replications remove the
`kappa=0.01` improvement, if the larger-dose opportunity gains disappear, if
the result depends on one anomalous seed, or if corrected integer-count or
coverage provenance changes any endpoint materially.

## Manuscript effect

This finding supports only the scoped Pythia-14M statement added to
`manuscript/introduction.tex` and the evidence crosswalks in
`manuscript/README.md` and `manuscript/experiment-control/README.md`.
`manuscript/methodology.tex` already names the executed post-RoPE Q/K/V sites
and `A7-Z-POST` topology, so no method-definition change is required. The
finding does not support a general scale-independent or runtime-speedup claim.
