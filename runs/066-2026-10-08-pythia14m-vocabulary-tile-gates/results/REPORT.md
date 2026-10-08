# Run066 — vocabulary-input tiles and pressure

All six models use existing h/z 8x16 corner tiles. Only the four new conditions add head-input tiles after final affine LayerNorm. The projection weights and logits are dense.

All 500 validation documents, 338 complete T=2048 blocks, excluded 1,444-token tail; seed 1234. Each new condition trains 712 matched updates. Counts are pooled before fractions.

| kappa | Head execution | Pressure | Loss | Delta loss vs parent | Head exact zeros | Empty 8x16 head tiles | Removed head energy | Zero head rows |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Ungated dense matmul | Ph | 5.553770 | +0.000000 | 0.000% | 0.000% | 0.000% | 0.000% |
| 0 | Tiled gate + dense matmul | Ph | 5.529001 | -0.024769 | 25.328% | 25.328% | 6.269% | 0.000% |
| 0 | Tiled gate + dense matmul | Ph + head | 5.497717 | -0.056053 | 27.421% | 27.421% | 6.515% | 0.000% |
| 0.1 | Ungated dense matmul | Ph | 5.596746 | +0.000000 | 0.000% | 0.000% | 0.000% | 0.000% |
| 0.1 | Tiled gate + dense matmul | Ph | 5.547977 | -0.048769 | 17.340% | 17.340% | 5.223% | 0.000% |
| 0.1 | Tiled gate + dense matmul | Ph + head | 5.507129 | -0.089616 | 19.808% | 19.808% | 5.409% | 0.000% |

All four new conditions have lower observed loss than their same-kappa parents while producing 17.340-27.421% empty 8x16 head tiles. In these measurements, the native exact-zero element fractions equal the empty 8x16 tile fractions.

At both kappa values, Ph+head has lower loss and more empty head tiles than Ph. The lowest-loss endpoint is Head tile / Ph + head at kappa=0: loss 5.497717, 27.421% empty tiles. Its rejected head tiles contain 6.515% of the measured pre-gate squared activation energy. This is evidence of structured sparsity with no observed loss penalty against these parents in this matched test.

Figures: [quality/tiles](../figures/01-head-tiles-and-loss.pdf), [zeros/occupancy](../figures/02-head-zero-elements-and-tiles.pdf), [trajectories](../figures/03-head-training-trajectories.pdf).

The JSON/CSV retain all tile shapes, head RMS and h/z spillover. Actual head dtypes, per-row source paths and full validation coverage are in the JSON.

Ph+head averages seven tensor means; Ph averages six, changing each h coefficient from 1/6 to 1/7. This is part of the intervention, so the comparison does not isolate head pressure from h-pressure reweighting. One seed, architecture and training budget limit generalization. No acceptable-loss bound was declared. Inference matmuls remain dense and latency is unmeasured in this experiment.
