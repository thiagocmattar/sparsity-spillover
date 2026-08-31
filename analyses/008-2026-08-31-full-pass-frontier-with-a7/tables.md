# A7 and A7-OL1 full-validation results

| kappa | Variant | Validation loss | R_model (%) | a zero (%) | m zero (%) | h zero (%) | q_post zero (%) | k_post zero (%) | v zero (%) | z zero (%) | attention_output zero (%) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A7 | 5.468401 | 7.217651 | 48.589665 | 49.789948 | 68.689276 | 0.000002 | 0.000000 | 0.000004 | 55.220760 | 0.000024 |
| 0 | A7-OL1 | 5.480184 | 7.054229 | 48.582652 | 49.502531 | 68.276640 | 0.000002 | 0.000002 | 0.000007 | 42.760534 | 0.000024 |
| 0.01 | A7 | 5.458822 | 7.617649 | 49.063737 | 50.329729 | 71.275100 | 0.398545 | 0.406645 | 1.654445 | 59.041219 | 0.000024 |
| 0.01 | A7-OL1 | 5.475797 | 7.723393 | 49.196145 | 50.103297 | 71.049532 | 1.372630 | 1.495615 | 2.337596 | 48.932977 | 0.000027 |
| 0.05 | A7 | 5.437888 | 9.126924 | 51.035704 | 52.681117 | 81.297894 | 1.839654 | 1.716139 | 7.312497 | 78.018438 | 0.000030 |
| 0.05 | A7-OL1 | 5.462811 | 9.863032 | 51.035985 | 52.240957 | 81.052322 | 5.427475 | 5.817149 | 10.525714 | 72.067291 | 0.000027 |
| 0.1 | A7 | 5.428681 | 10.425018 | 53.052260 | 55.074809 | 91.352964 | 3.150124 | 3.227091 | 11.355038 | 90.051548 | 0.000045 |
| 0.1 | A7-OL1 | 5.429497 | 11.796760 | 53.104115 | 54.873615 | 91.607391 | 8.227452 | 8.544486 | 18.819221 | 88.896452 | 0.000045 |
| 0.5 | A7 | 5.702923 | 15.386813 | 62.306143 | 65.721056 | 99.858786 | 17.434735 | 18.133822 | 31.255166 | 99.865473 | 0.000018 |
| 0.5 | A7-OL1 | 5.829390 | 27.482684 | 75.732456 | 68.697737 | 99.876421 | 93.544983 | 94.541277 | 98.713252 | 99.918232 | 0.000010 |

All percentages use pooled integer counts over all six layers and the complete 338-block validation pass (692,224 input tokens; 1,444-token excluded tail).
Per-site zero mass is exact-zero activation mass. `attention_output` is the post-`W_o` diagnostic and is not a gated A7 site.
`R_model` is measured exact-zero logical-product opportunity including the dense LM-head denominator, not runtime speedup.
The ten rows share seed 1234, 712 optimizer steps, and A7-Z-POST mixed one-sided/symmetric gates. A7 has no pressure; A7-OL1 applies seven-site orthogonal L1 with lambda 1 and trust budget 1.
