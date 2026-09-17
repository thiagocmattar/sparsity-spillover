# 14M table — uniform ordinary final validation

Explicit alternative: all six columns use ordinary final-checkpoint validation loss.

Each cell is (S_model %, validation loss in nats).

| kappa | A4 | A4+OL1(all) | A4+OL1(h) | A7 | A7+OL1(all) | A7+OL1(h) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | (7.2120, 5.470497) | (7.8100, 5.458170) | (8.4094, 5.215749) | (7.2177, 5.468401) | (7.0542, 5.480184) | (8.3994, 5.219827) |
| 0.01 | (7.4137, 5.466500) | (8.5867, 5.458277) | (8.5858, 5.204504) | (7.6176, 5.458822) | (7.7234, 5.475797) | (8.8561, 5.198102) |
| 0.05 | (8.2059, 5.434110) | (10.4564, 5.489803) | (9.0356, 5.195590) | (9.1269, 5.437888) | (9.8630, 5.462811) | (10.1261, 5.194959) |
| 0.1 | (8.9530, 5.419642) | (11.3943, 5.548323) | (9.3435, 5.228687) | (10.4250, 5.428681) | (11.7968, 5.429497) | (10.9036, 5.237395) |
| 0.5 | (10.2155, 5.659680) | (12.7134, 6.037987) | (10.2274, 5.722666) | (15.3868, 5.702923) | (27.4827, 5.829390) | (16.6636, 5.732049) |

All 30 points: seed 1234, 712 updates, 1,493,172,224 training tokens;
500 validation documents, 338 complete blocks, 1,444-token excluded tail.
A4+OL1(h) is the audited realized Run 012 intervention. See README.md and results.json.
