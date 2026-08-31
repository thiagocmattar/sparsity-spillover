# O001 - Corrected four-site A4-OL1 versus matched A4

## Question

Does operational OL1 pressure at every active A4 site improve the Pythia-14M
full-pass validation-loss versus logical-opportunity frontier at matched
`kappa`?

## Method and coverage

Run 015 independently trained five `A4-Z` conditions at
`kappa in {0, 0.01, 0.05, 0.1, 0.5}`. Each condition used one-sided gates and
post-gate OL1 at `a,m,h,z` in all six layers, `lambda=1`, and trust budget `1`.
The pressure scalar was the unweighted mean of all 24 site/layer tensor means.
Every optimizer boundary verified the realized tensor count and canonical-name
hash before differentiation.

Each condition completed 712 optimizer boundaries and 1,493,172,224 input
tokens from the same random initialization, realized data order, optimizer
schedule, gate threshold, and validation cache as its Run 011 comparator.
Every endpoint covers all 500 validation documents, all 338 complete
2,048-token blocks, and 692,224 input tokens; the 1,444-token incomplete tail
is excluded and reported. `R_model` and exact-zero fractions divide pooled
integer counts from the complete workload.

## Result

| `kappa` | corrected A4-OL1 loss | corrected `R_model` | A4 loss | A4 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.458170 | 7.8100% | 5.470497 | 7.2120% | -0.012327 | +0.5980 pp |
| 0.01 | 5.458277 | 8.5867% | 5.466500 | 7.4137% | -0.008222 | +1.1730 pp |
| 0.05 | 5.489803 | 10.4564% | 5.434110 | 8.2059% | +0.055693 | +2.2506 pp |
| 0.1 | 5.548323 | 11.3943% | 5.419642 | 8.9530% | +0.128681 | +2.4413 pp |
| 0.5 | 6.037987 | 12.7134% | 5.659680 | 10.2155% | +0.378307 | +2.4979 pp |

The corrected result supports a threshold-dependent, not blanket, benefit.
At `kappa=0` and `0.01`, four-site OL1 lowers validation loss while raising
`R_model`. At `0.05`, `0.1`, and `0.5`, it raises `R_model` by about
2.25--2.50 percentage points with progressively larger quality costs. Across
the ten matched Run 011/015 endpoints, Run 011 `kappa=0.1` is the lowest-loss
nondominated point, while corrected Run 015 `kappa=0.05`, `0.1`, and `0.5`
extend the logical-opportunity frontier.

The selected-site exact-zero fractions show the endpoint change:

| `kappa` | `a` | `m` | `h` | `z` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 68.8906% | 39.2961% | 74.2046% | 69.6248% |
| 0.01 | 73.4909% | 43.4637% | 80.8333% | 85.2595% |
| 0.05 | 88.5048% | 60.4390% | 93.2746% | 97.3920% |
| 0.1 | 91.3095% | 75.9117% | 97.1490% | 99.2833% |
| 0.5 | 98.9060% | 96.7655% | 99.9404% | 99.9757% |

All 3,560 boundaries realized the required 24-tensor objective. There were no
nonfinite gradients, FP16 overflows, or skipped updates. Conflict was observed
on 318, 340, 393, 438, and 426 boundaries as `kappa` increased. Projection was
applied on 710--712 boundaries per condition; the trust cap was saturated on
only 1--7 boundaries, and the maximum final correction ratio never exceeded
`1.0`. These are measured training-time interactions, not checkpoint
reconstructions.

## Interpretation

This one-seed result replaces the missing four-site evidence but does not
repair Run 012 retroactively. Run 012 remains valid only as A4-Z gates plus
OL1@h. The corrected intervention improves both matched outcomes only at the
two smallest thresholds; at larger thresholds its value is an explicit
quality--opportunity tradeoff. A numbered cross-run analysis is still required
before restoring or creating a finding or changing result-bearing manuscript
text.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scale transfer.
- All four pressure sites change jointly, so site-specific effects are not
  identified.
- Thresholded training changes both forward values and gradient support;
  matching `kappa` does not match realized pressure magnitude.
- `R_model` is a logical zero-operand opportunity, not removed FLOPs or
  measured sparse-kernel acceleration.
- Independently scheduled GPUs share scientific identities but are not
  presumed bitwise replicas.

## Sources

- Run 015 cohort verification: `artifacts/verification.json`
- Per-attempt integer counters:
  `artifacts/attempts/*/diagnostics/logical_products.json`
- Per-attempt activation counts:
  `artifacts/attempts/*/diagnostics/activation_statistics.json`
- Per-boundary OL1 records: `artifacts/attempts/*/events.jsonl`
- Matched A4 source:
  `runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json`
- Generating verifier: `03_verify.py` and `verification.py`
- Transfer and teardown provenance:
  `prelaunch/scientific-transfer-closeout.json`

No PDF figure was generated for this run-local observation. A paper-facing
frontier figure belongs in a separately approved numbered analysis.
