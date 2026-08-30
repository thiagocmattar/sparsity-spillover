# F001 - A4-OL1 improves the moderate-threshold Pythia-14M frontier

Status: tentative

Approved: 2026-08-30

## Statement

For one matched seed and one full MiniPile pretraining pass at Pythia-14M,
adding four-site OL1 pressure (`lambda=1`, `step_budget=1`) to the operational
`A4-Z = {a,m,h,z}` one-sided threshold topology lowers complete-validation loss
and increases measured `R_model` at each matched `kappa` in
`{0, 0.01, 0.05, 0.1}`. The advantage contracts as `kappa` increases and does
not extend to `kappa=0.5`: there, validation loss is 0.062986 higher than A4
while `R_model` increases by only 0.0119 percentage points.

This is evidence that conflict-aware pressure can improve the
quality--logical-opportunity frontier of the tested architecture-wide topology
at moderate thresholds. It is not evidence of measured runtime speedup or of
the same effect at larger model scales.

## Evidence

The finding covers five matched A4/A4-OL1 pairs. Each condition trained from
the same random initialization and realized data order for 712 optimizer
boundaries. Every reported endpoint uses all 338 complete 2,048-token blocks
from all 500 MiniPile validation documents (692,224 input tokens); the
1,444-token incomplete tail is excluded and reported.

| `kappa` | A4 loss | A4 `R_model` | A4-OL1 loss | A4-OL1 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | 7.2120% | 5.215749 | 8.4094% | -0.254749 | +1.1974 pp |
| 0.01 | 5.466500 | 7.4137% | 5.204504 | 8.5858% | -0.261996 | +1.1721 pp |
| 0.05 | 5.434110 | 8.2059% | 5.195590 | 9.0356% | -0.238520 | +0.8297 pp |
| 0.1 | 5.419642 | 8.9530% | 5.228687 | 9.3435% | -0.190955 | +0.3905 pp |
| 0.5 | 5.659680 | 10.2155% | 5.722666 | 10.2274% | +0.062986 | +0.0119 pp |

Within A4-OL1, the lowest validation loss is 5.195590 at `kappa=0.05` and
`R_model=9.0356%`. The `kappa=0.1` endpoint trades a slightly higher loss of
5.228687 for `R_model=9.3435%`. Exact-zero mass rises monotonically at all four
selected sites; the full count-pooled site table is retained by Analysis 007.

## Sources and provenance

- Primary analysis: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/`
- Primary observation:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/observations/O001-full-pass-frontier-with-a4-ol1.md`
- Generating script:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/01_build.py`
- Machine-readable reduction:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/figure_data.json`
- A4 source: `runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json`
- A4-OL1 source:
  `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/artifacts/verification.json`

Analysis 007 recomputes every Run 012 `R_model` from integer logical-product
counts and every selected-site exact-zero fraction from pooled integer
activation counts before division. Run 012's verifier records the matched A4
deltas above and locks the source attempts by hash.

## Definitions

- `A4-Z` activates one-sided threshold gates at `a`, `m`, `h`, and `z`, where
  `z` is the concatenated attention context immediately before `W_o`.
- A4-OL1 adds orthogonal L1 pressure at those same four post-gate sites without
  changing topology, gate operator, threshold, optimizer schedule, or data.
- `R_model` is the count-first fraction of model-wide logical scalar products
  with at least one exactly-zero operand, including the dense LM head in its
  denominator. It is an opportunity measure, not removed FLOPs or speedup.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scaling behavior.
- The four-site joint intervention cannot attribute the outcome to one site.
- Separately scheduled GPUs are matched by scientific identities but are not
  bitwise replicas.
- Connected frontier lines describe dose order; they are not fitted response
  curves.
- No sparse kernel or wall-clock acceleration was evaluated.

## What would change the conclusion

The finding should be strengthened if additional matched seeds reproduce the
moderate-threshold improvement, especially at `kappa=0.05` and `0.1`, and if a
larger Pythia scale shows the same paired direction. It should be narrowed or
discarded if matched replications remove the loss or `R_model` advantage, if
the result depends on one anomalous seed, or if corrected integer-count or
coverage provenance changes any paired endpoint materially.

## Manuscript effect

This finding supports only the scoped Pythia-14M statement added to
`manuscript/introduction.tex` and the evidence crosswalk in
`manuscript/README.md`. It also requires the executed `z` site and `A4-Z`
topology to be explicit in `manuscript/methodology.tex`. It does not support a
general scale-independent or runtime-speedup claim.
