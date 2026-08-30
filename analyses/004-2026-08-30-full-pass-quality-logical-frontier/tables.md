# Analysis 004 tables

Fractions were recalculated from stored integer counts before display.

## Combined endpoints

| Series | Dose | Final validation loss | `R_model` (%) |
| --- | ---: | ---: | ---: |
| GeLU control | control | 5.208583 | 0.000001 |
| A1-H ReLU control | control | 5.269646 | 2.714130 |
| A1-H naive L1 | lambda=0.05 | 5.206149 | 3.141595 |
| A1-H naive L1 | lambda=0.1 | 5.165474 | 3.336306 |
| A1-H naive L1 | lambda=0.5 | 5.112688 | 3.787578 |
| A1-H naive L1 | lambda=1 | 5.102276 | 3.949334 |
| A1-H OL1 | lambda=0.05 | 5.198062 | 3.145227 |
| A1-H OL1 | lambda=0.1 | 5.159391 | 3.338579 |
| A1-H OL1 | lambda=0.5 | 5.110235 | 3.767998 |
| A1-H OL1 | lambda=1 | 5.121030 | 3.938363 |
| A4-Z threshold | kappa=0 | 5.470497 | 7.212019 |
| A4-Z threshold | kappa=0.01 | 5.466500 | 7.413704 |
| A4-Z threshold | kappa=0.05 | 5.434110 | 8.205893 |
| A4-Z threshold | kappa=0.1 | 5.419642 | 8.953005 |
| A4-Z threshold | kappa=0.5 | 5.659680 | 10.215537 |

## Run 011 threshold effects

Deltas are relative to the within-A4-Z `kappa=0` reference.

| `kappa` | Final validation loss | Delta loss | `R_model` (%) | Delta `R_model` (pp) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | +0.000000 | 7.212019 | +0.000000 |
| 0.01 | 5.466500 | -0.003998 | 7.413704 | +0.201686 |
| 0.05 | 5.434110 | -0.036387 | 8.205893 | +0.993874 |
| 0.1 | 5.419642 | -0.050855 | 8.953005 | +1.740986 |
| 0.5 | 5.659680 | +0.189182 | 10.215537 | +3.003518 |

## Run 011 selected-site exact-zero mass

| `kappa` | `a` (%) | `m` (%) | `h` (%) | `z` (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 48.559601 | 49.779081 | 68.764996 | 54.525292 |
| 0.01 | 49.095297 | 50.396965 | 71.180523 | 59.643281 |
| 0.05 | 51.155428 | 52.711752 | 81.340082 | 77.640876 |
| 0.1 | 53.244102 | 55.199547 | 91.853055 | 89.233831 |
| 0.5 | 63.433959 | 66.386250 | 99.876273 | 99.880513 |

## Run 011 untargeted-site exact-zero mass

| `kappa` | `q_post` (%) | `k_post` (%) | `v` (%) | Attention output (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000564 | 0.000000564 | 0.000004891 | 0.000024265 |
| 0.01 | 0.000001129 | 0.000000188 | 0.000003010 | 0.000023889 |
| 0.05 | 0.000000564 | 0.000000564 | 0.000001693 | 0.000027839 |
| 0.1 | 0.000000752 | 0.000000941 | 0.000001505 | 0.000045332 |
| 0.5 | 0.000000188 | 0.000000941 | 0.000000752 | 0.000005079 |

`attention_output` is post-`W_o`; it is distinct from the selected pre-`W_o` site `z`.
