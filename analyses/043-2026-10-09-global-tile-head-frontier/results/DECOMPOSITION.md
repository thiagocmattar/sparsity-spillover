# Measured full-model decomposition

Runs064/065 only: complete CUPTI profiles for six retained checkpoints. Categories
below are milliseconds of arithmetic mean CUDA activity durations across 12 traces
per condition/mode. They keep Analysis042's categories, colors, and mapping.
The total is a sum of activities, not the independently timed host geometric mean.
The CSV also preserves the GPU activity union, span, and gaps. Small differences
between summed activities and the union reflect the original timestamp reductions.

Only 14M has trained 8×16 corner thresholds at h/z. 31M/70M use scalar thresholds;
their consumers' matrix tiles or compaction do not make the training gate tiled.
Every vocabulary head here has its threshold OFF and dense arithmetic. The H/Z
output block groups gate, consumer/preparation, down projections and residual work;
fused 14M/31M internals have no independent substep timing.

| Model | κ | Mode | Input setup | Block normalization | QKV projection | RoPE / context layout | Attention | MLP up projection | H/Z output block | Final normalization | Vocabulary projection | Activity sum | Host geometric mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | 0 | native | 0.017629 | 0.041986 | 0.024154 | 0.125930 | 0.122602 | 0.024691 | 0.192002 | 0.003451 | 0.187447 | 0.739892 | 0.837750 |
| 14M | 0 | tile_sparse | 0.015877 | 0.010485 | 0.024349 | 0.024149 | 0.121322 | 0.027536 | 0.063397 | 0.003475 | 0.180482 | 0.471072 | 0.509939 |
| 14M | 0.1 | native | 0.017523 | 0.041931 | 0.024208 | 0.125652 | 0.122458 | 0.024626 | 0.191473 | 0.003451 | 0.186687 | 0.738009 | 0.834927 |
| 14M | 0.1 | tile_sparse | 0.015816 | 0.010442 | 0.024288 | 0.024075 | 0.121250 | 0.027389 | 0.040568 | 0.003483 | 0.179700 | 0.447010 | 0.486441 |
| 31M | 0 | native | 0.018397 | 0.043419 | 0.049160 | 0.147215 | 0.188462 | 0.079710 | 0.172881 | 0.003637 | 0.281373 | 0.984254 | 1.055761 |
| 31M | 0 | sparse | 0.016039 | 0.014952 | 0.048451 | 0.022488 | 0.122885 | 0.080381 | 0.431555 | 0.003640 | 0.230982 | 0.971373 | 1.006855 |
| 31M | 0.1 | native | 0.018453 | 0.043480 | 0.048821 | 0.147080 | 0.188378 | 0.079586 | 0.172851 | 0.003675 | 0.279354 | 0.981678 | 1.055637 |
| 31M | 0.1 | sparse | 0.016088 | 0.014795 | 0.048501 | 0.022541 | 0.123321 | 0.080508 | 0.315837 | 0.003645 | 0.230926 | 0.856164 | 0.890827 |
| 70M | 0 | native | 0.017009 | 0.045403 | 0.116390 | 0.182267 | 0.341840 | 0.146096 | 0.304310 | 0.003768 | 0.497358 | 1.654440 | 1.736296 |
| 70M | 0 | sparse | 0.015962 | 0.019875 | 0.116227 | 0.034963 | 0.218123 | 0.147072 | 0.205560 | 0.003896 | 0.438691 | 1.200370 | 1.246257 |
| 70M | 0.1 | native | 0.016963 | 0.045448 | 0.116234 | 0.182035 | 0.341966 | 0.146120 | 0.304391 | 0.003776 | 0.497786 | 1.654718 | 1.731771 |
| 70M | 0.1 | sparse | 0.015934 | 0.019878 | 0.116239 | 0.034981 | 0.218413 | 0.146725 | 0.159674 | 0.003880 | 0.438799 | 1.154523 | 1.194367 |

Full decompositions for the newer head interventions: **N/A**. Runs067/068 retain
projection-only, gate+cast+projection, and complete-forward timings, measured as
separate overlapping boundaries. Their independent geometric means are never
subtracted or stacked to estimate unmeasured stages.

[All 22 execution decompositions](semantic-decomposition.csv) ·
[Original stage mapping and fusion limits](../../042-2026-10-08-model-latency-decomposition/LABELS.md) ·
[Original detailed stage table](../../042-2026-10-08-model-latency-decomposition/results/forward-stages.csv).
