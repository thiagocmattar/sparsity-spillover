# A4-OL1 full-validation results

| kappa | Validation loss | R_model (%) | a zero (%) | m zero (%) | h zero (%) | z zero (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.215749 | 8.409 | 49.413 | 49.816 | 93.543 | 64.671 |
| 0.01 | 5.204504 | 8.586 | 49.942 | 50.390 | 94.871 | 71.971 |
| 0.05 | 5.195590 | 9.036 | 51.791 | 52.391 | 97.915 | 88.307 |
| 0.1 | 5.228687 | 9.343 | 53.809 | 54.747 | 99.125 | 96.774 |
| 0.5 | 5.722666 | 10.227 | 63.622 | 66.441 | 99.944 | 99.934 |

All percentages are count-pooled over the complete 338-block validation pass (692,224 input tokens; 1,444-token excluded tail).
Per-site zero mass is exact-zero activation mass. `R_model` is measured exact-zero logical-product opportunity, not runtime speedup.
The five endpoints share seed 1, 712 optimizer steps, A4-Z topology, one-sided threshold gates, and four-site OL1 pressure with lambda=1.
