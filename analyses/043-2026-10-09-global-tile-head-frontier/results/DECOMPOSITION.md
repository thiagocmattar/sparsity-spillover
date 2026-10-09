# Measured full-model decomposition

Run064 only: complete CUPTI profiles for two retained 14M checkpoints. Categories
below are milliseconds of arithmetic mean CUDA activity durations across 12 traces
per condition/mode. They keep Analysis042's categories, colors, and mapping.
The total is a sum of activities, not the independently timed host geometric mean.
The CSV also preserves the GPU activity union, span, and gaps. Small differences
between summed activities and the union reflect the original timestamp reductions.

Both checkpoints have trained 8×16 corner thresholds at h/z.
Every vocabulary head here has its threshold OFF and dense arithmetic. The H/Z
output block groups gate, consumer/preparation, down projections and residual work;
fused 14M internals have no independent substep timing.

| Model | κ | Mode | Input setup | Block normalization | QKV projection | RoPE / context layout | Attention | MLP up projection | H/Z output block | Final normalization | Vocabulary projection | Activity sum | Host geometric mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | 0 | native | 0.017629 | 0.041986 | 0.024154 | 0.125930 | 0.122602 | 0.024691 | 0.192002 | 0.003451 | 0.187447 | 0.739892 | 0.837750 |
| 14M | 0 | tile_sparse | 0.015877 | 0.010485 | 0.024349 | 0.024149 | 0.121322 | 0.027536 | 0.063397 | 0.003475 | 0.180482 | 0.471072 | 0.509939 |
| 14M | 0.1 | native | 0.017523 | 0.041931 | 0.024208 | 0.125652 | 0.122458 | 0.024626 | 0.191473 | 0.003451 | 0.186687 | 0.738009 | 0.834927 |
| 14M | 0.1 | tile_sparse | 0.015816 | 0.010442 | 0.024288 | 0.024075 | 0.121250 | 0.027389 | 0.040568 | 0.003483 | 0.179700 | 0.447010 | 0.486441 |

Full decompositions for the newer head interventions: **N/A**. Runs067/068 retain
projection-only, gate+cast+projection, and complete-forward timings, measured as
separate overlapping boundaries. Their independent geometric means are never
subtracted or stacked to estimate unmeasured stages.

[All 10 execution decompositions](semantic-decomposition.csv) ·
[Original stage mapping and fusion limits](../../042-2026-10-08-model-latency-decomposition/LABELS.md) ·
[Original detailed stage table](../../042-2026-10-08-model-latency-decomposition/results/forward-stages.csv).
