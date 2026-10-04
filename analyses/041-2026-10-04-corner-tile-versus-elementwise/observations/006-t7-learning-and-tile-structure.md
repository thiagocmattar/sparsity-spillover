# Does T7/Ph learn under the stronger corner restriction?

Question: determine whether the two approved T7/Ph corner conditions learn and
retain final activity, and describe their recorded h/z tile evolution.

Method: use Run062's native FP16 full-validation events at updates
0/1/32/128/256/512/712 and complete activation/tile diagnostics at0/32/128/712.
Both conditions share random initialization, data/order, seed1234,712-update
budget, MB32/GAS32/T2048, optimizer and h-only orthogonal L1 pressure. Preflight
state was discarded before fresh training. Every diagnostic pools integer
counts over the six layers and keeps QKV tiles headwise. All corner certificates
pass; neither condition has skipped/overflowed optimizer updates.

Coverage at every recorded point: all500 MiniPile validation documents,
338 complete2048-token blocks,692224 inputs and691886 predictions;1444 tail
tokens are excluded. Interpolation lines connect only the recorded points.

Caption: left shows full-validation NLL; dotted horizontal lines are the
canonical final scalar losses, not scalar learning trajectories. Middle/right
show empty8x16 h/z tiles at the diagnostic cadence. Blue circles indicate
kappa0; purple dashed squares indicate kappa.1. One training seed.

Results: loss falls11.023044 → 5.721071 at kappa0 and11.026098 → 5.835102 at
kappa.1. Scalar final losses are5.219827/5.237395, leaving penalties
.501244/.597707 (perplexity+65.077%/+81.795%). All42 final gate ports retain
some activity in each condition; no entire layer/site is empty.

| kappa | Update | h empty tiles (%) | z empty tiles (%) |
|---:|---:|---:|---:|
| 0 | 0 | 51.885 | 50.838 |
| 0 | 32 | 69.435 | 43.321 |
| 0 | 128 | 71.775 | 53.664 |
| 0 | 712 | 61.764 | 48.389 |
| .1 | 0 | 62.902 | 99.273 |
| .1 | 32 | 74.384 | 54.091 |
| .1 | 128 | 88.059 | 78.906 |
| .1 | 712 | 92.842 | 92.461 |

The very sparse initial z at kappa.1 becomes more active early and ends highly
structured. Final per-layer h occupancy ranges88.002–95.756% empty and z
85.489–96.330%. These observations support learning viability under the
restriction and a substantial quality cost at this fixed horizon.

Caveats: surviving activity and falling loss do not prove representational
redistribution, preserved quality or a longer-horizon outcome. One seed and a
complete operator change also limit causal interpretation. No runtime result is
inferred from these curves.

Artifacts: [PDF](../figures/06-t7-learning-and-tile-structure.pdf),
[recorded curves, counts, layer ranges and hashes](../results/t7-training.json),
[quality comparison](../results/t7-comparison.md).
Source script: [03_t7_training.py](../03_t7_training.py). PDF rendered and
visually inspected; earlier figures are preserved.
