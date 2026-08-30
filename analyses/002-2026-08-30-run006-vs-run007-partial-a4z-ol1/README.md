# Analysis 002 - Run 006 versus completed Run 007 conditions

## Question

How does adding Run 007's operational OL1 correction at all four post-threshold
A4-Z sites change validation loss, logical opportunity, and per-site exact-zero
mass relative to Run 006 at the same kappa?

## Method and evidence scope

`01_compare.py` loads Run 006's terminal valid `verification.json` and
independently revalidates each selected Run 007 attempt. It requires completed
status; identical initialization and realized schedule; 581 optimizer updates
and 76,152,832 training tokens; the matched data, model, optimizer, seed, gate,
and kappa; OL1 at `a,m,h,z` with lambda 1 and trust budget 1; all 338 complete
2,048-token validation blocks (692,224 input tokens and the 1,444-token tail
excluded); complete OL1 event diagnostics; finite activation/logical metrics;
and matching transfer and checkpoint inventories.

Four of five planned kappas have usable pairs. Run 007 attempt 002 failed at
kappa 0.01, but completed attempt 003 supplies a valid replacement endpoint.
Both Run 007 attempts at kappa 0.5 failed with accelerator out-of-memory errors,
so kappa 0.5 is unpaired and excluded rather than imputed. Run 007 as a whole is
therefore not relabeled as a terminally valid cohort; this analysis is explicitly
partial evidence from revalidated completed attempts.

## Paired results

Deltas are Run 007 minus Run 006. Logical deltas are percentage points. The
quality interpretation follows the user's assumption that first- and
second-decimal validation-loss differences are prone to realization noise.

| kappa | validation loss, 006 -> 007 (delta) | `R_block`, 006 -> 007 (delta pp) | `R_model`, 006 -> 007 (delta pp) | noise-aware paired interpretation |
| ---: | ---: | ---: | ---: | --- |
| 0 | 5.661601 -> 5.707263 (+0.045661) | 26.4443% -> 29.5214% (+3.0770) | 7.9207% -> 8.8423% (+0.9216) | no resolved quality difference; small model-level and clear block-level logical increase |
| 0.01 | 5.656006 -> 5.699371 (+0.043365) | 26.9400% -> 31.9754% (+5.0354) | 8.0692% -> 9.5774% (+1.5082) | no resolved quality difference; directional model-level and clear block-level logical increase |
| 0.05 | 5.673034 -> 5.748999 (+0.075966) | 28.7171% -> 36.7239% (+8.0068) | 8.6014% -> 10.9997% (+2.3982) | no resolved quality difference; material logical increase |
| 0.1 | 5.651189 -> 5.797230 (+0.146041) | 30.4748% -> 40.7275% (+10.2527) | 9.1279% -> 12.1988% (+3.0709) | no resolved quality difference under the stated assumption; clearest available logical increase |
| 0.5 | 6.036731 -> unavailable | 34.9301% -> unavailable | 10.4624% -> unavailable | no Run 007 endpoint; no paired inference |

The exact-zero table reports absolute percentages as `Run 006 -> Run 007`, with
the change in percentage points in parentheses.

| kappa | `a` exact-zero mass | `m` exact-zero mass | `h` exact-zero mass | `z` exact-zero mass |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 49.2311% -> 83.3899% (+34.1588) | 50.4059% -> 46.0478% (-4.3581) | 80.0789% -> 78.3510% (-1.7278) | 71.0152% -> 79.0627% (+8.0474) |
| 0.01 | 49.7273% -> 87.3926% (+37.6653) | 50.9610% -> 50.2988% (-0.6623) | 81.9225% -> 84.6686% (+2.7461) | 73.8115% -> 93.5110% (+19.6995) |
| 0.05 | 50.6076% -> 91.4214% (+40.8138) | 52.1434% -> 66.6204% (+14.4770) | 89.8973% -> 95.5849% (+5.6876) | 84.3148% -> 98.3462% (+14.0314) |
| 0.1 | 52.3409% -> 93.2269% (+40.8860) | 53.9994% -> 79.6533% (+25.6538) | 96.6644% -> 98.4812% (+1.8168) | 93.8568% -> 99.1304% (+5.2736) |
| 0.5 | 68.0493% -> unavailable | 68.5429% -> unavailable | 99.9990% -> unavailable | 100.0000% -> unavailable |

## Interpretation under realization noise

All four observed validation-loss deltas point numerically upward, from +0.044
through +0.146. Under the stated realization-noise assumption, none resolves a
quality degradation on its own. The common direction is still a caution rather
than evidence of quality neutrality, especially because there is only one seed.

The logical and exact-zero changes are much coarser. OL1 increases `R_block` at
every available kappa, with the gain growing monotonically from 3.1 to 10.3
percentage points. `R_model` gains grow from 0.9 to 3.1 points. At kappa 0 and
0.01 the redistribution is dominated by roughly 34--38 points more exact-zero
mass at `a`, together with 8--20 points more at `z`; `m` and `h` can move
slightly downward. At kappa 0.05 and 0.1 all four thresholded sites increase,
including about 41 points at `a`, 14--26 points at `m`, and 5--14 points at `z`.

The clearest available paired result is therefore a material increase in
logical opportunity at the higher completed kappas without a resolved quality
change under the user's noise model. The result is not a complete dose-response
claim: kappa 0.5 is missing, and no extrapolation to it is warranted. The exact
eight-point loss--`R_model` frontier is arithmetic bookkeeping only; it does not
turn the single realization into an uncertainty-controlled comparison.

## Limits

This is one matched seed at one model scale. The Run 007 cohort is incomplete,
and its missing highest-kappa endpoint followed two OOM failures, so missingness
may be related to resource demand. `R_block` and `R_model` are logical-product
opportunities rather than runtime speedups. Because OL1 is applied jointly at
all four sites, the analysis cannot identify an individual site's causal
contribution. No finding or manuscript claim is promoted by this analysis.
