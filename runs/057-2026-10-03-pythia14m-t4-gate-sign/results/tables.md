# T4 gate-sign results (14M, seed 1234)

All 500 MiniPile validation documents; 338 complete 2,048-token blocks; excluded tail: 1,444 tokens.
Bilateral means a/m only. h/z remain unilateral. Values are final reloaded-checkpoint diagnostics.

| Pressure | a/m gate | kappa | Validation loss | R_model (%) | a zeros (%) | m zeros (%) | h zeros (%) | z zeros (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PH | unilateral | 0 | 5.217240 | 8.4167 | 49.3966 | 49.7673 | 93.5527 | 65.5603 |
| PH | bilateral | 0 | 5.132202 | 4.6316 | 0.0000 | 0.0000 | 93.1082 | 60.6549 |
| PH | unilateral | 0.1 | 5.238433 | 9.3410 | 53.8470 | 54.7600 | 99.1063 | 96.4461 |
| PH | bilateral | 0.1 | 5.183998 | 5.7376 | 7.0525 | 7.5365 | 97.8782 | 93.6946 |
| PALL | unilateral | 0 | 5.458790 | 7.7812 | 68.4805 | 39.0400 | 74.0325 | 69.8708 |
| PALL | bilateral | 0 | 5.477359 | 3.9325 | 0.0000 | 0.0000 | 72.6685 | 77.0495 |
| PALL | unilateral | 0.1 | 5.555985 | 11.4053 | 91.6314 | 75.9876 | 97.0790 | 99.3220 |
| PALL | bilateral | 0.1 | 5.682653 | 10.0464 | 79.8030 | 59.4295 | 91.1509 | 97.6863 |

Paired deltas are bilateral minus unilateral; negative loss delta favors bilateral.

| Pressure | kappa | Loss delta | R_model delta (pp) | a zeros delta (pp) | m zeros delta (pp) |
|---|---:|---:|---:|---:|---:|
| PH | 0 | -0.085038 | -3.7852 | -49.3966 | -49.7673 |
| PH | 0.1 | -0.054435 | -3.6034 | -46.7944 | -47.2234 |
| PALL | 0 | +0.018569 | -3.8487 | -68.4805 | -39.0400 |
| PALL | 0.1 | +0.126668 | -1.3589 | -11.8284 | -16.5581 |

R_model is a logical scalar-product opportunity, not measured speedup.

| Pressure | Gate | kappa | Site | Pre negative (%) | Pre near zero <=.01 (%) | Removed large-negative energy (%) | Pre mean | Pre RMS | Post RMS | Empty 8x16 tiles (%) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| PH | unilateral | 0 | a | 49.3966 | 0.8779 | 50.2139 | 0.002929 | 1.011788 | 0.713910 | 0.0000 |
| PH | unilateral | 0 | m | 49.7673 | 0.8514 | 50.7828 | -0.004744 | 1.004000 | 0.704356 | 0.0000 |
| PH | bilateral | 0 | a | 49.7695 | 0.6644 | 0.0000 | 0.000654 | 1.006507 | 1.006507 | 0.0000 |
| PH | bilateral | 0 | m | 49.7906 | 0.6877 | 0.0000 | -0.000711 | 0.984181 | 0.984181 | 0.0000 |
| PH | unilateral | 0.1 | a | 50.1019 | 0.7431 | 48.5605 | 0.010988 | 1.023951 | 0.734221 | 0.0000 |
| PH | unilateral | 0.1 | m | 50.8847 | 0.7462 | 49.4468 | -0.004841 | 1.018139 | 0.723728 | 0.0000 |
| PH | bilateral | 0.1 | a | 49.7857 | 0.7350 | 0.0000 | 0.000823 | 1.014637 | 1.014521 | 0.0000 |
| PH | bilateral | 0.1 | m | 49.7757 | 0.7055 | 0.0000 | 0.000002 | 0.991485 | 0.991349 | 0.0000 |
| PALL | unilateral | 0 | a | 68.4805 | 4.2720 | 94.0791 | -0.177751 | 0.986923 | 0.240148 | 0.0006 |
| PALL | unilateral | 0 | m | 39.0400 | 4.7756 | 93.4028 | -0.070764 | 0.979412 | 0.251563 | 0.0000 |
| PALL | bilateral | 0 | a | 50.9180 | 7.3938 | 0.0000 | -0.000402 | 0.657343 | 0.657343 | 0.0000 |
| PALL | bilateral | 0 | m | 50.0107 | 6.9127 | 0.0000 | 0.000757 | 0.800071 | 0.800071 | 0.0000 |
| PALL | unilateral | 0.1 | a | 89.0512 | 0.5464 | 94.7067 | -0.417390 | 1.021240 | 0.234467 | 58.1223 |
| PALL | unilateral | 0.1 | m | 62.1089 | 3.0599 | 94.1950 | -0.189252 | 0.968779 | 0.231284 | 0.0104 |
| PALL | bilateral | 0.1 | a | 50.9494 | 15.9197 | 0.0000 | 0.004145 | 0.438119 | 0.436422 | 8.3387 |
| PALL | bilateral | 0.1 | m | 52.5279 | 6.8470 | 0.0000 | 0.006264 | 0.722883 | 0.721631 | 0.0000 |

Boundary summaries below use the final 64 updates (649--712), with equal weight per update. Cosine compares clipped task gradients to unclipped pressure gradients; OL1 projection is measured in the Adam-preconditioned direction space.

| Pressure | Gate | kappa | Mean pressure loss | Mean gradient cosine | Conflict updates / 64 | OL1 projections / 64 | Median trust scale | Median final pressure/task direction ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PH | unilateral | 0 | 0.010052 | -0.063691 | 43 | 64 | 1.000000 | 0.334731 |
| PH | bilateral | 0 | 0.016673 | -0.124254 | 57 | 64 | 1.000000 | 0.474714 |
| PH | unilateral | 0.1 | 0.002621 | -0.027716 | 37 | 64 | 1.000000 | 0.042457 |
| PH | bilateral | 0.1 | 0.007008 | -0.075238 | 44 | 64 | 1.000000 | 0.215877 |
| PALL | unilateral | 0 | 0.059548 | -0.044929 | 38 | 64 | 1.000000 | 0.692651 |
| PALL | bilateral | 0 | 0.149839 | -0.008070 | 37 | 0 | 0.555688 | 1.000000 |
| PALL | unilateral | 0.1 | 0.038111 | -0.088135 | 50 | 64 | 1.000000 | 0.201954 |
| PALL | bilateral | 0.1 | 0.092046 | -0.039478 | 32 | 0 | 1.000000 | 0.403728 |
