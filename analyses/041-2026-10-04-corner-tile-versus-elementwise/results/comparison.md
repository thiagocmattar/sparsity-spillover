# T2/Ph: scalar versus causal corner tile gates

Full FP16 validation, 338 complete 2048-token blocks; excluded tail 1444 tokens. Percentages pool integer counts over all six layers. Empty tile means all 128 values are exactly zero.

| Gate | kappa | Loss | Perplexity | R_model (%) | A zeros (%) | A empty8x16 (%) | M zeros (%) | M empty8x16 (%) | H zeros (%) | H empty8x16 (%) | Z zeros (%) | Z empty8x16 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elementwise | 0 | 5.127033 | 168.516 | 4.6301 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 93.1227 | 3.0442 | 60.4571 | 0.0153 |
| corner_tile | 0 | 5.553770 | 258.209 | 2.9404 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 57.5056 | 57.5056 | 44.9262 | 44.9262 |
| elementwise | 0.1 | 5.152346 | 172.836 | 5.1708 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 97.7829 | 43.6962 | 92.3742 | 20.7041 |
| corner_tile | 0.1 | 5.596746 | 269.548 | 4.3653 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 84.3082 | 84.3082 | 70.9567 | 70.9567 |

Differences below are corner tile minus elementwise; positive loss change means worse quality.

| kappa | Loss delta | Perplexity change (%) | H empty tiles delta (pp) | Z empty tiles delta (pp) |
|---:|---:|---:|---:|---:|
| 0 | +0.426737 | +53.225 | +54.4614 | +44.9108 |
| 0.1 | +0.444400 | +55.955 | +40.6120 | +50.2527 |

The tile operator changes which signed values survive and rejects equality. This compares complete operators, not geometry alone. One seed and a fixed 712-update budget; no runtime measurement.
