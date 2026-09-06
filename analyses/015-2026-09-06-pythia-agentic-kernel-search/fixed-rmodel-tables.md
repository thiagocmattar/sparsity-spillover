# Fresh-process fixed-R_model optimization

All ratios compare independently launched processes at the identical checkpoint and canonical `R_model`.

## Condition summaries

| Size | Condition | R_model | Baseline | Optimized | Baseline speedup | Optimized speedup | Optimized / baseline | Repeat range | Wins |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 14M | A4-OL1 kappa=0.5 | 12.713% | P0 | K001 | 1.1112x | 1.1322x | 1.0202x | [0.9932, 1.0287] | 2/3 |
| 14M | A7-OL1 kappa=0.5 | 27.483% | P0 | K001 | 1.1036x | 1.1184x | 1.0102x | [1.0093, 1.0218] | 3/3 |
| 70M | A4-OL1 kappa=0.5 | 35.596% | K009 | K016 | 1.0337x | 1.0474x | 1.0132x | [1.0093, 1.0188] | 3/3 |
| 70M | A7-OL1 kappa=0.5 | 40.602% | K009 | K016 | 1.0018x | 1.0093x | 1.0075x | [0.9897, 1.0869] | 2/3 |
| 410M | A4-OL1 kappa=0.5 | 71.591% | K004 | K010 | 1.0121x | 1.0192x | 1.0063x | [1.0037, 1.0092] | 3/3 |
| 410M | A7-OL1 kappa=0.5 | 80.616% | K004 | K010 | 1.0110x | 1.0176x | 1.0065x | [1.0057, 1.0113] | 3/3 |

## Architecture summaries

| Size | Qualified pairs | Median optimized / baseline | Range | Wins |
|---|---:|---:|---:|---:|
| 14M | 6 | 1.0152x | [0.9932, 1.0287] | 5/6 |
| 70M | 6 | 1.0113x | [0.9897, 1.0869] | 5/6 |
| 410M | 6 | 1.0064x | [1.0037, 1.0113] | 6/6 |

The range is the minimum-to-maximum across three process-paired repeats, not a confidence interval.
The comparison changes implementation coverage/dispatch while holding the trained checkpoint and canonical logical opportunity fixed.

## 14M final robustness-policy contrast

K013 is the final all-checkpoint robustness policy, while K001 is the direct P0 kernel optimization used in the primary 14M comparison.

| Condition | R_model | P0 speedup | K013 speedup | K013 / P0 | Repeat range |
|---|---:|---:|---:|---:|---:|
| A4-OL1 kappa=0.5 | 12.713% | 1.1143x | 1.0389x | 0.9370x | [0.9323, 0.9433] |
| A7-OL1 kappa=0.5 | 27.483% | 1.1129x | 1.0481x | 0.9477x | [0.9373, 0.9646] |
