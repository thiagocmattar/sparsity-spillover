# Analysis 001 - Run 008 versus Run 010

## Question

How does adding Run 007's operational OL1 correction at all seven post-threshold
A7-Z-POST sites change validation loss, logical opportunity, and per-site exact
zero mass relative to Run 008 at the same kappa?

## Method

`01_compare.py` loads the terminal `verification.json` artifacts from Runs 008
and 010 and rejects either source unless it is verified valid evidence. It also
requires identical initial parameters, realized training schedule, optimizer
boundary count, token budget, complete-validation pass count, and kappa grid.

For each matched kappa, deltas are Run 010 minus Run 008. Lower validation-loss
deltas and higher `R_model`/exact-zero deltas are favorable in their respective
estimands. Paired dominance requires no worse validation loss and no lower
`R_model`, with at least one strict improvement. The joint ten-point Pareto
frontier is evaluated under the same loss/`R_model` ordering.

## Sources

- `runs/008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local/artifacts/verification.json`
- `runs/010-2026-08-30-pythia14m-a7-z-post-mixed-threshold-ol1-local/artifacts/verification.json`

## Result

The runs match on initialization, realized schedule, 2,905 optimizer boundaries,
380,764,160 training input tokens, 15 complete validation passes, and the full
kappa grid. The executing difference is the approved seven-site OL1 correction
in Run 010.

| kappa | validation loss, 008 -> 010 (delta) | `R_model`, 008 -> 010 (delta pp) | noise-aware paired interpretation |
| ---: | ---: | ---: | --- |
| 0 | 5.657176 -> 5.649200 (-0.007975) | 7.8823% -> 7.7741% (-0.1082) | no resolved quality or model-level logical difference |
| 0.01 | 5.656776 -> 5.645979 (-0.010798) | 8.4122% -> 8.9898% (+0.5776) | no resolved quality or model-level logical difference |
| 0.05 | 5.662928 -> 5.669881 (+0.006954) | 10.2549% -> 12.6962% (+2.4413) | same resolved quality; directional logical increase |
| 0.1 | 5.648046 -> 5.678580 (+0.030534) | 12.0822% -> 18.4278% (+6.3456) | same resolved quality; clear logical increase |
| 0.5 | 6.036204 -> 6.029251 (-0.006953) | 27.5723% -> 29.9524% (+2.3800) | same resolved quality; branch-input closure and block saturation |

Per-site entries below are exact-zero percentage-point changes from Run 008 to
Run 010. Positive values mean OL1 produced more exact-zero mass.

| kappa | `a` | `m` | `h` | `q_post` | `k_post` | `v` | `z` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.6937 | +0.3843 | -0.0903 | +0.0000 | +0.0000 | +0.0000 | -13.3691 |
| 0.01 | +0.4281 | +0.0953 | +0.0673 | +3.6079 | +3.5145 | +2.0988 | -12.2130 |
| 0.05 | +0.5956 | +0.0263 | -0.7988 | +11.6100 | +11.4708 | +14.7281 | -5.2625 |
| 0.1 | +0.7386 | +0.1409 | -0.4490 | +34.6090 | +32.7735 | +36.0358 | -2.0229 |
| 0.5 | +31.9932 | +31.5429 | +0.0010 | +0.0612 | +0.0454 | +0.0493 | +0.0000 |

At moderate kappa, OL1's logical gain comes predominantly from much higher
exact-zero mass at signed Q/K/V, while `a` and `m` move only slightly and `z`
is lower than in Run 008. At kappa 0.5, Q/K/V, `h`, and `z` were already almost
saturated in Run 008; OL1 instead closes the remaining `a` and `m` mass and
drives `R_block` from 92.054% to effectively 100%.

The exact joint ten-point loss--`R_model` frontier contains Run 010 at kappa
0.01, 0.05, 0.1, and 0.5, plus Run 008 at kappa 0.1. Those dominance labels are
arithmetic bookkeeping and are not treated as evidence of material superiority
under the realization-noise assumption below.

## Interpretation under realization noise

The user directs that first- and second-decimal differences be treated as prone
to realization noise. Under that assumption, none of the validation-loss
differences, whose magnitudes range from 0.007 to 0.031, supports a quality
improvement or degradation claim. The apparent paired dominance at kappa 0.01
and 0.5 is therefore not scientifically meaningful on loss.

The robust result is redistribution of exact-zero mass:

- at kappa 0 and 0.01, the large effect is 12--13 percentage points less zero
  mass at `z`; sub-percentage-point changes at `a`, `m`, and `h` are ignored;
- at kappa 0.05, Q/K/V gain roughly 11--15 percentage points while `z` loses
  about 5 points;
- at kappa 0.1, Q/K/V gain roughly 33--36 percentage points, the clearest
  moderate-threshold shift; changes below one point at `a`, `m`, and `h` are
  ignored;
- at kappa 0.5, `a` and `m` each gain roughly 32 percentage points and the
  block becomes essentially all-zero-reachable; other sites were already
  saturated.

Accordingly, the supported interpretation is that all-site OL1 can strongly
relocate sparsity toward signed Q/K/V at moderate thresholds and close the
remaining branch-input gap at kappa 0.5. It is not supported that OL1 improves
validation quality, nor that the small kappa-0/0.01 `R_model` differences are
meaningful. The kappa-0.1 logical increase is the clearest coarse model-level
shift; the kappa-0.05 and 0.5 increases are directionally consistent but remain
single-realization estimates.

## Limits

This is one matched seed at one model scale with no replicate uncertainty.
`R_model` is logical-product opportunity rather than runtime speedup, and the
joint intervention does not identify the causal contribution of an individual
pressure site. No finding or manuscript claim is promoted by this analysis.
