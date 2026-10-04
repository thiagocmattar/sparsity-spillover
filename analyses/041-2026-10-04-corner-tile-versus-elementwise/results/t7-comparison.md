# 14M T7/Ph: scalar versus 8x16 corner tiles

Canonical native FP16 autocast with FP32 parameters. Integer counts pooled over all six layers; QKV tiles remain headwise.

| Gate | kappa | FP16 loss | Perplexity | R_model (%) | a zeros (%) | a empty tiles (%) | m zeros (%) | m empty tiles (%) | h zeros (%) | h empty tiles (%) | q_post zeros (%) | q_post empty tiles (%) | k_post zeros (%) | k_post empty tiles (%) | v zeros (%) | v empty tiles (%) | z zeros (%) | z empty tiles (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0 | 5.219827 | 184.902 | 8.399 | 49.378 | 0.000 | 49.745 | 0.000 | 93.475 | 1.513 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 64.394 | 0.003 |
| tile | 0 | 5.721071 | 305.232 | 7.255 | 55.017 | 55.017 | 54.471 | 54.471 | 61.764 | 61.764 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 48.389 | 48.389 |
| scalar | 0.1 | 5.237395 | 188.179 | 10.904 | 53.521 | 0.000 | 54.547 | 0.000 | 99.133 | 57.705 | 4.073 | 0.000 | 6.306 | 0.000 | 8.403 | 0.000 | 96.820 | 52.100 |
| tile | 0.1 | 5.835102 | 342.100 | 12.616 | 53.415 | 53.415 | 53.388 | 53.388 | 92.842 | 92.842 | 11.378 | 11.378 | 10.404 | 10.404 | 22.259 | 22.259 | 92.461 | 92.461 |

| kappa | Tile minus scalar loss | Perplexity change (%) |
|---:|---:|---:|
| 0 | +0.501244 | +65.077 |
| 0.1 | +0.597707 | +81.795 |

One seed, fixed budget, complete-operator comparison. Corner equality and signed interior preservation also change. No causal claim about representation redistribution.

R_model is a logical zero-product opportunity, not a speedup. All500 validation documents; excluded1444-token tail.
