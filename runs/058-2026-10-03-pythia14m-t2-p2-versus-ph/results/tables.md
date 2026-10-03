# T2/P2 versus T2/Ph: 14M, seed1234

Complete500-document validation:338 complete2048-token blocks; excluded tail1444 tokens. All gates are unilateral at h/z only. Zeros and tiles are count-pooled percentages.

| Pressure | kappa | Validation loss | R_model (%) | A zeros (%) | A empty8x16 (%) | M zeros (%) | M empty8x16 (%) | H zeros (%) | H empty8x16 (%) | Z zeros (%) | Z empty8x16 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PH | 0 | 5.127033 | 4.6301 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 93.122695 | 3.044216 | 60.457109 | 0.015337 |
| P2 | 0 | 5.280096 | 4.9558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 88.521395 | 0.209578 | 93.698246 | 37.946411 |
| PH | 0.1 | 5.152346 | 5.1708 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 97.782855 | 43.696188 | 92.374169 | 20.704064 |
| P2 | 0.1 | 5.330544 | 5.1970 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 96.742329 | 34.653426 | 98.983271 | 74.828018 |

Paired differences are P2 minus Ph. Negative loss delta favors P2.

| kappa | Loss difference | R_model difference (pp) | H zeros difference (pp) | H tiles difference (pp) | Z zeros difference (pp) | Z tiles difference (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.153062 | +0.3257 | -4.6013 | -2.8346 | +33.2411 | +37.9311 |
| 0.1 | +0.178198 | +0.0262 | -1.0405 | -9.0428 | +6.6091 | +54.1240 |

Empty tiles require all128 values to equal zero. R_model is logical-product opportunity, not runtime speedup. Equal lambda uses a mean over6 tensors in Ph versus12 in P2; this is a recipe comparison with one seed, not matched gradient intensity.
