# Paper-facing tables

All losses are paired with the eager logical-product pass unless the column explicitly names training loss.

## Baseline exposure and optimization context

| Model | Parameters | Training tokens | Tokens / parameter | Peak LR | Mean A0 train loss, steps 649--712 | A0 validation loss | Clipped A0 boundaries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pythia-14M | 14,067,712 | 1,493,172,224 | 106.142 | 1e-03 | 5.249282 | 5.208594 | 5 / 712 |
| Pythia-70M | 70,426,624 | 1,493,172,224 | 21.202 | 1e-03 | 4.009096 | 4.099766 | 8 / 712 |
| Pythia-410M | 405,334,016 | 1,493,172,224 | 3.684 | 3e-04 | 4.493717 | 4.547456 | 56 / 712 |

## Matched trained endpoint boundary

| Model | A4-OL1 k=0 loss / R_model | A4-OL1 k=.5 loss / R_model | A7-OL1 k=0 loss / R_model | A7-OL1 k=.5 loss / R_model |
|---|---:|---:|---:|---:|
| Pythia-14M | 5.458276 / 7.8100% | 6.037982 / 12.7134% | 5.480181 / 7.0542% | 5.829407 / 27.4827% |
| Pythia-70M | 4.805361 / 25.6725% | 5.389480 / 35.5962% | 4.941206 / 23.5624% | 5.215925 / 40.6019% |
| Pythia-410M | 5.692075 / 38.3387% | 5.190966 / 71.5914% | 5.426296 / 41.1215% | 5.120692 / 80.6155% |

The A4-OL1/A7-OL1 columns compare complete recipes: A7 adds post-RoPE q/k/v gates and pressures those sites. They do not isolate gate placement.
