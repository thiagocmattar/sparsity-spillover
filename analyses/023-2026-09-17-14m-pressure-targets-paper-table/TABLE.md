# Approved 14M table — exact delivered values

Archival table: A4/A7 without pressure and with all-site OL1 use eager logical-pass loss; h-only columns use ordinary final validation loss. All evaluate the final checkpoint.

Each cell is (S_model %, validation loss in nats).

| kappa | A4 | A4+OL1(all) | A4+OL1(h) | A7 | A7+OL1(all) | A7+OL1(h) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | (7.2120, 5.470516) | (7.8100, 5.458276) | (8.4094, 5.215749) | (7.2177, 5.468393) | (7.0542, 5.480181) | (8.3994, 5.219827) |
| 0.01 | (7.4137, 5.466488) | (8.5867, 5.458309) | (8.5858, 5.204504) | (7.6176, 5.458820) | (7.7234, 5.475801) | (8.8561, 5.198102) |
| 0.05 | (8.2059, 5.434115) | (10.4564, 5.489760) | (9.0356, 5.195590) | (9.1269, 5.437875) | (9.8630, 5.462800) | (10.1261, 5.194959) |
| 0.1 | (8.9530, 5.419673) | (11.3943, 5.548254) | (9.3435, 5.228687) | (10.4250, 5.428668) | (11.7968, 5.429488) | (10.9036, 5.237395) |
| 0.5 | (10.2155, 5.659678) | (12.7134, 6.037982) | (10.2274, 5.722666) | (15.3868, 5.702895) | (27.4827, 5.829407) | (16.6636, 5.732049) |

All 30 points: seed 1234, 712 updates, 1,493,172,224 training tokens;
500 validation documents, 338 complete blocks, 1,444-token excluded tail.
A4+OL1(h) is the audited realized Run 012 intervention. See README.md and results.json.
