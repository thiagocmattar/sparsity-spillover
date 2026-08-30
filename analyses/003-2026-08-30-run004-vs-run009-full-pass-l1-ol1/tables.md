# Analysis 003 tables

All percentages are count-first reductions from the reloaded final checkpoints.
Activation statistics pool all six layers and all 338 complete validation
blocks (692,224 input tokens); the 1,444-token tail is excluded. Run 009
reuses Run 004's controls rather than rerunning them.

## Exact-zero mass, measured R_model, and final validation loss

| Run / condition | lambda | `h` zero (%) | `m` zero (%) | `q_post` zero (%) | `k_post` zero (%) | `v` zero (%) | attention output zero (%) | `R_model` (%) | final validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 004 GeLU control | - | 9.828e-06 | 0 | 1.317e-06 | 1.881e-06 | 2.257e-06 | 2.182e-05 | 1.142e-06 | 5.208583 |
| Run 004 ReLU control | - | 63.448126 | 0 | 2.069e-06 | 1.129e-06 | 5.079e-06 | 1.486e-05 | 2.714130 | 5.269646 |
| Run 004 Naive L1 | 0.05 | 73.440901 | 0 | 1.129e-06 | 2.822e-06 | 3.950e-06 | 2.220e-05 | 3.141595 | 5.206149 |
| Run 004 Naive L1 | 0.10 | 77.992711 | 0 | 3.762e-07 | 1.129e-06 | 3.198e-06 | 2.558e-05 | 3.336306 | 5.165474 |
| Run 004 Naive L1 | 0.50 | 88.542050 | 0 | 1.505e-06 | 1.505e-06 | 3.386e-06 | 2.859e-05 | 3.787578 | 5.112688 |
| Run 004 Naive L1 | 1.00 | 92.323398 | 0 | 2.069e-06 | 9.405e-07 | 5.267e-06 | 2.088e-05 | 3.949334 | 5.102276 |
| Run 009 OL1 | 0.05 | 73.525849 | 0 | 2.257e-06 | 9.405e-07 | 5.831e-06 | 2.182e-05 | 3.145227 | 5.198062 |
| Run 009 OL1 | 0.10 | 78.045801 | 0 | 1.129e-06 | 7.524e-07 | 2.445e-06 | 2.671e-05 | 3.338579 | 5.159391 |
| Run 009 OL1 | 0.50 | 88.084312 | 0 | 1.317e-06 | 7.524e-07 | 1.448e-05 | 1.975e-05 | 3.767998 | 5.110235 |
| Run 009 OL1 | 1.00 | 92.067074 | 0 | 9.405e-07 | 1.881e-07 | 6.207e-06 | 1.636e-05 | 3.938363 | 5.121030 |

## Near-zero mass at |x| <= 1e-3

| Run / condition | lambda | `h` (%) | `m` (%) | `q_post` (%) | `k_post` (%) | `v` (%) | attention output (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 004 GeLU control | - | 0.237736 | 0.070630 | 0.065518 | 0.059550 | 0.129597 | 0.701354 |
| Run 004 ReLU control | - | 63.508542 | 0.067001 | 0.050349 | 0.045815 | 0.132739 | 0.480568 |
| Run 004 Naive L1 | 0.05 | 73.501792 | 0.070568 | 0.056967 | 0.045499 | 0.149564 | 0.697766 |
| Run 004 Naive L1 | 0.10 | 78.047723 | 0.067589 | 0.055263 | 0.055462 | 0.155159 | 0.801612 |
| Run 004 Naive L1 | 0.50 | 88.579031 | 0.071537 | 0.057830 | 0.049814 | 0.180704 | 0.855134 |
| Run 004 Naive L1 | 1.00 | 92.351206 | 0.063650 | 0.051757 | 0.050614 | 0.185325 | 0.724827 |
| Run 009 OL1 | 0.05 | 73.585716 | 0.067713 | 0.052586 | 0.046340 | 0.149782 | 0.709594 |
| Run 009 OL1 | 0.10 | 78.103702 | 0.075298 | 0.057245 | 0.051244 | 0.165584 | 0.799813 |
| Run 009 OL1 | 0.50 | 88.126710 | 0.073156 | 0.047091 | 0.038698 | 0.187612 | 0.764404 |
| Run 009 OL1 | 1.00 | 92.102116 | 0.063182 | 0.031245 | 0.024312 | 0.192956 | 0.525481 |

## Matched OL1 minus naive-L1 endpoint changes

Positive zero-mass and `R_model` values mean OL1 is higher; negative loss
values mean OL1 has lower validation loss.

| lambda | `h` exact-zero change (pp) | `h` near-zero change (pp) | `R_model` change (pp) | final validation-loss change |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | +0.084948 | +0.083925 | +0.003632 | -0.008087 |
| 0.10 | +0.053090 | +0.055979 | +0.002272 | -0.006083 |
| 0.50 | -0.457738 | -0.452321 | -0.019581 | -0.002453 |
| 1.00 | -0.256324 | -0.249090 | -0.010971 | +0.018753 |

`attention_output` is the output of `attention.dense` (`W_o`) before
residual addition; it is not the pre-`W_o` context site `z`. `R_model`
is an exact-zero logical-product opportunity, not measured speedup.
