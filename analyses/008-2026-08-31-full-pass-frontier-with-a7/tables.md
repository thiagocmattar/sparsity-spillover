# A7 full-validation results

| kappa | Validation loss | R_model (%) | a zero (%) | m zero (%) | h zero (%) | q_post zero (%) | k_post zero (%) | v zero (%) | z zero (%) | attention_output zero (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.468401 | 7.217651 | 48.589665 | 49.789948 | 68.689276 | 0.000002 | 0.000000 | 0.000004 | 55.220760 | 0.000024 |
| 0.01 | 5.458822 | 7.617649 | 49.063737 | 50.329729 | 71.275100 | 0.398545 | 0.406645 | 1.654445 | 59.041219 | 0.000024 |
| 0.05 | 5.437888 | 9.126924 | 51.035704 | 52.681117 | 81.297894 | 1.839654 | 1.716139 | 7.312497 | 78.018438 | 0.000030 |
| 0.1 | 5.428681 | 10.425018 | 53.052260 | 55.074809 | 91.352964 | 3.150124 | 3.227091 | 11.355038 | 90.051548 | 0.000045 |
| 0.5 | 5.702923 | 15.386813 | 62.306143 | 65.721056 | 99.858786 | 17.434735 | 18.133822 | 31.255166 | 99.865473 | 0.000018 |

All percentages use pooled integer counts over all six layers and the complete 338-block validation pass (692,224 input tokens; 1,444-token excluded tail).
Per-site zero mass is exact-zero activation mass. `attention_output` is the post-`W_o` diagnostic and is not a gated A7 site.
`R_model` is measured exact-zero logical-product opportunity including the dense LM-head denominator, not runtime speedup.
The five rows share seed 1234, 712 optimizer steps, A7-Z-POST topology, mixed one-sided/symmetric gates, and no pressure objective.
