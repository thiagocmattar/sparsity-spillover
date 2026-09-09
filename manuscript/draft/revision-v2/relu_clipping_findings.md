# ReLU pretraining followed by post-hoc clipping

Source: Analysis 020/02_cross_size.py, data/cross_size_interventions.csv and
cross-size-audit.json. This is an identity-checked join of 36 FP16 endpoints
and 360 clipping evaluations; the main display selects 96 distinct evaluations.
Every checkpoint has ten targets and full 338-block validation. Run 030 source
hashes match; canonical checkpoint identities come from Analysis 019. The CSV
under revision-v2/data is an exact copy. No new checkpoint evaluation occurred.

The clipping rule acts at a,m,h,z after the trained nonlinearity, calibrated
per site/layer on the same ten-block training sample. For A1-H this extends
inference reach from h-only to A4. p is a calibration policy, not achieved
sparsity. Every increment below uses the measured p=0 record, not a substituted
canonical endpoint. Raw and block-normalized values use the common model/block
denominators, with A7 reach as the normalization for all curves at each size.

## Matched policies (G = GELU A0; H = ReLU A1-H)

### 14M

Canonical ReLU base loss cost: +0.061061249.

| p | G loss | H loss | G S (%) | H S (%) | G delta loss | H delta loss | G delta S (pp) | H delta S (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 5.208594 | 5.269633 | 0.000001 | 2.714133 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.1 | 5.214453 | 5.274073 | 1.278706 | 3.568403 | 0.005859 | 0.004439 | 1.278705 | 0.854269 |
| 0.2 | 5.250270 | 5.307940 | 2.554786 | 4.418365 | 0.041676 | 0.038307 | 2.554785 | 1.704232 |
| 0.3 | 5.361901 | 5.405056 | 3.831641 | 5.271871 | 0.153307 | 0.135422 | 3.831640 | 2.557737 |
| 0.4 | 5.605384 | 5.602965 | 5.125544 | 6.129325 | 0.396790 | 0.333331 | 5.125543 | 3.415191 |
| 0.5 | 6.077731 | 5.976274 | 6.441923 | 6.994451 | 0.869137 | 0.706641 | 6.441922 | 4.280318 |
| 0.6 | 6.821173 | 6.521474 | 7.764415 | 7.892904 | 1.612579 | 1.251840 | 7.764414 | 5.178771 |
| 0.7 | 7.766948 | 7.306175 | 9.123471 | 9.051787 | 2.558354 | 2.036542 | 9.123470 | 6.337654 |
| 0.8 | 8.336118 | 8.107618 | 10.585958 | 10.558519 | 3.127524 | 2.837985 | 10.585957 | 7.844386 |
| 0.9 | 8.825610 | 8.869450 | 11.941881 | 11.878925 | 3.617016 | 3.599817 | 11.941880 | 9.164792 |

### 70M

Canonical ReLU base loss cost: +0.122983675.

| p | G loss | H loss | G S (%) | H S (%) | G delta loss | H delta loss | G delta S (pp) | H delta S (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 4.099766 | 4.222750 | 0.000454 | 10.065758 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.1 | 4.103608 | 4.223747 | 3.704772 | 12.536992 | 0.003842 | 0.000998 | 3.704318 | 2.471233 |
| 0.2 | 4.130721 | 4.234510 | 7.409856 | 15.009890 | 0.030955 | 0.011760 | 7.409402 | 4.944131 |
| 0.3 | 4.207335 | 4.269464 | 11.128199 | 17.485463 | 0.107569 | 0.046714 | 11.127744 | 7.419705 |
| 0.4 | 4.397795 | 4.346144 | 14.844233 | 19.985072 | 0.298029 | 0.123394 | 14.843779 | 9.919314 |
| 0.5 | 4.833057 | 4.522054 | 18.588525 | 22.515643 | 0.733291 | 0.299304 | 18.588070 | 12.449885 |
| 0.6 | 5.657785 | 4.914340 | 22.410886 | 25.107710 | 1.558019 | 0.691591 | 22.410432 | 15.041951 |
| 0.7 | 6.927989 | 5.722641 | 26.348400 | 27.893498 | 2.828223 | 1.499891 | 26.347946 | 17.827739 |
| 0.8 | 8.072236 | 6.912035 | 30.462129 | 31.035075 | 3.972470 | 2.689285 | 30.461675 | 20.969316 |
| 0.9 | 9.179397 | 9.133301 | 34.581403 | 33.891163 | 5.079631 | 4.910551 | 34.580949 | 23.825405 |

### 410M

Canonical ReLU base loss cost: +0.103837621.

| p | G loss | H loss | G S (%) | H S (%) | G delta loss | H delta loss | G delta S (pp) | H delta S (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 4.547456 | 4.651294 | 0.011048 | 21.049725 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.1 | 4.553016 | 4.652483 | 7.463172 | 25.984306 | 0.005560 | 0.001189 | 7.452124 | 4.934581 |
| 0.2 | 4.581109 | 4.662035 | 14.883300 | 30.904799 | 0.033653 | 0.010741 | 14.872252 | 9.855074 |
| 0.3 | 4.672734 | 4.693632 | 22.242787 | 35.810314 | 0.125278 | 0.042338 | 22.231739 | 14.760589 |
| 0.4 | 4.931758 | 4.773735 | 29.538790 | 40.713066 | 0.384302 | 0.122441 | 29.527742 | 19.663341 |
| 0.5 | 5.585540 | 4.975709 | 36.804329 | 45.619300 | 1.038083 | 0.324415 | 36.793281 | 24.569575 |
| 0.6 | 6.721288 | 5.444486 | 44.098063 | 50.582074 | 2.173832 | 0.793192 | 44.087015 | 29.532349 |
| 0.7 | 7.758184 | 6.466273 | 51.563621 | 55.642240 | 3.210728 | 1.814979 | 51.552573 | 34.592515 |
| 0.8 | 8.311326 | 8.391250 | 60.012937 | 61.770014 | 3.763870 | 3.739956 | 60.001889 | 40.720288 |
| 0.9 | 9.027046 | 9.029800 | 70.047141 | 69.162896 | 4.479590 | 4.378505 | 70.036093 | 48.113171 |

## Observed regions and crossings

- At 14M, A0 has lower total loss at p=0..0.3; ReLU has lower loss at
  p=0.4..0.8, with higher sparsity only at p=0.4..0.6. At p=0.9 A0 has
  lower loss and slightly higher sparsity. The tiny p=.4 loss margin is
  .0024194, above measured p=0 drift but not a seed-robust equivalence claim.
- At 70M, A0 has lower loss at p=0..0.3; ReLU has lower loss at p=.4.. .9,
  with higher sparsity at .4.. .8. At .9 its sparsity is .690 pp lower.
- At 410M, A0 has lower loss at p=0.. .3; ReLU has lower loss and higher
  sparsity at .4.. .7. A0 again has lower loss at .8 and .9, and higher
  sparsity at .9. The extreme-target reversal is retained.

Thus p=.5 is a recurring favorable comparison at all sizes, not universal
dominance. Incremental loss damage is smaller for ReLU at all positive tested
p, but that need not overcome the initial ReLU cost. These are policy-matched
comparisons, not equal achieved sparsity or equal sitewise zero allocation.
Quantile targets can act on existing zero mass; a plateau does not establish
tolerance to removing additional nonzero values. The widening raw-sparsity
region also reflects head share and block-operation weights.

## Examples at comparable observed sparsity

No interpolation; the signed mismatch is H minus G. The audit also retains
all cross-target pairs within one percentage point. There is no uniformly
close informative pair at every size; the 410M example explicitly allows
a larger mismatch.

| Size | G p | H p | G loss | H loss | G S (%) | H S (%) | Mismatch (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 14M | 0.6 | 0.6 | 6.821173 | 6.521474 | 7.764415 | 7.892904 | +0.128489 |
| 70M | 0.6 | 0.5 | 5.657785 | 4.522054 | 22.410886 | 22.515643 | +0.104757 |
| 410M | 0.6 | 0.5 | 6.721288 | 4.975709 | 44.098063 | 45.619300 | +1.521237 |

## Endpoint versus measured p=0 discrepancies

Both use FP16 and full validation on identical checkpoint content. Separate
wrappers/calibration/evaluation paths allow numerical drift. None is silently
interpreted as a gain from a zero-target intervention.

| Checkpoint ID | Loss drift | Sparsity drift (pp) |
|---|---:|---:|
| 14M:A0:None | +2.10474934e-05 | +7.29523685e-08 |
| 14M:A1-H:None | -8.12598234e-07 | +3.52993912e-06 |
| 14M:A4-OL1:0.0 | +1.88012095e-05 | +7.21665824e-08 |
| 14M:A4-OL1:0.01 | +2.21799817e-05 | +1.41671708e-05 |
| 14M:A4-OL1:0.05 | +5.30904567e-05 | +2.08695321e-06 |
| 14M:A4-OL1:0.1 | +5.51685074e-05 | +1.0670659e-05 |
| 14M:A4-OL1:0.5 | +3.39640668e-06 | +7.63458638e-06 |
| 14M:A7-OL1:0.0 | +8.87368557e-07 | +7.20845465e-06 |
| 14M:A7-OL1:0.01 | -7.54615964e-06 | +1.03595821e-05 |
| 14M:A7-OL1:0.05 | +2.3472944e-05 | -4.58869926e-05 |
| 14M:A7-OL1:0.1 | -2.61618541e-05 | +3.73730018e-05 |
| 14M:A7-OL1:0.5 | -2.90341631e-05 | -0.000271105868 |
| 70M:A0:None | +0 | +0 |
| 70M:A1-H:None | +0 | +0 |
| 70M:A4-OL1:0.0 | -1.56770796e-06 | +3.18825308e-05 |
| 70M:A4-OL1:0.01 | +9.03229036e-05 | +9.21174537e-07 |
| 70M:A4-OL1:0.05 | +0.000180712113 | -0.000174423707 |
| 70M:A4-OL1:0.1 | +0.000151541811 | +0.000332153507 |
| 70M:A4-OL1:0.5 | +8.86920641e-05 | +6.50602348e-05 |
| 70M:A7-OL1:0.0 | +1.37316405e-05 | -2.03666421e-05 |
| 70M:A7-OL1:0.01 | +7.18112528e-06 | -8.34126982e-05 |
| 70M:A7-OL1:0.05 | +1.93274233e-05 | -1.68606925e-05 |
| 70M:A7-OL1:0.1 | -2.61431615e-05 | +0.000119403303 |
| 70M:A7-OL1:0.5 | -0.000120160142 | +7.22456742e-05 |
| 410M:A0:None | +0 | +0 |
| 410M:A1-H:None | +0 | +0 |
| 410M:A4-OL1:0.0 | -5.05475603e-06 | -0.000160082583 |
| 410M:A4-OL1:0.01 | -1.90805401e-05 | -2.96868144e-05 |
| 410M:A4-OL1:0.05 | -2.26991416e-05 | -1.03575478e-05 |
| 410M:A4-OL1:0.1 | -4.90027772e-06 | -3.27363291e-05 |
| 410M:A4-OL1:0.5 | +0.000169217234 | -2.41792576e-05 |
| 410M:A7-OL1:0.0 | -9.05849525e-06 | +2.6210494e-05 |
| 410M:A7-OL1:0.01 | +1.68233228e-05 | +1.33125694e-05 |
| 410M:A7-OL1:0.05 | -3.26308978e-06 | -0.000107931589 |
| 410M:A7-OL1:0.1 | +1.1611267e-05 | +3.22447545e-05 |
| 410M:A7-OL1:0.5 | +2.50480584e-06 | +0.000907354105 |
