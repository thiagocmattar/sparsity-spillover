# Vocabulary kernel efficiency: matched final measurements

This diagnostic keeps Run068 weights, gates and quality unchanged. Native means native PyTorch/cuBLAS
on the **same gated checkpoint**. It is a kernel-equivalence control. The original ungated model and
existing optimized 14M frontier remain separate practical references; the strongest body pipeline is
not integrated here. A kernel improvement alone does not establish a useful quality–latency gain.

All accepted modes qualify over all338 validation blocks in three fresh processes on the same RTX5090.
Times below are geometric means of equal-size process geometric means. Independent host boundaries
overlap and must not be summed. Positive full-forward change means slower than the matched native control.

| Head κ | Canonical loss | Mode | Projection µs | Gate/cast/projection µs | Full ms | Full change vs matched native |
|---:|---:|---|---:|---:|---:|---:|
| 0.1 | 5.560725 | legacy_n256/no_skip | 318.065 | 336.710 | 1.126695 | +12.82% |
| 0.1 | 5.560725 | legacy_n256/sparse | 199.857 | 218.875 | 1.009376 | +1.07% |
| 0.1 | 5.560725 | native | 192.579 | 209.533 | 0.998643 | +0.00% |
| 0.1 | 5.560725 | reuse2_n128/no_skip | 246.760 | 265.120 | 1.052548 | +5.40% |
| 0.1 | 5.560725 | reuse2_n128/sparse | 173.949 | 192.724 | 0.984591 | -1.41% |
| 0.1 | 5.560725 | reuse2_n256/no_skip | 287.730 | 306.043 | 1.097346 | +9.88% |
| 0.1 | 5.560725 | reuse2_n256/sparse | 169.459 | 188.005 | 0.980100 | -1.86% |
| 0.5 | 5.882600 | legacy_n256/no_skip | 318.188 | 337.423 | 1.127391 | +12.87% |
| 0.5 | 5.882600 | legacy_n256/sparse | 160.785 | 179.970 | 0.971115 | -2.78% |
| 0.5 | 5.882600 | native | 192.982 | 210.219 | 0.998882 | +0.00% |
| 0.5 | 5.882600 | prefetch_n128/no_skip | 361.794 | 381.749 | 1.169332 | +17.06% |
| 0.5 | 5.882600 | prefetch_n128/sparse | 157.091 | 176.143 | 0.968456 | -3.05% |
| 0.5 | 5.882600 | reuse2_n128/no_skip | 245.462 | 265.128 | 1.051916 | +5.31% |
| 0.5 | 5.882600 | reuse2_n128/sparse | 158.950 | 177.915 | 0.971130 | -2.78% |

Profiler status: **unavailable**. Counter denial or incomplete collection leaves
the hardware bottleneck unresolved; it must not be replaced by a confident source-level explanation.

Exploratory screens retain all variants and failures. Only modes qualified in every final process
appear above. Process ranges in comparison.json are descriptive ranges, not confidence intervals.
One GPU, two checkpoints, one training seed and one uncached full-sequence workload limit generalization.

Sources: [verification](verification.json), [process reduction](comparison.json),
[raw endpoints](../artifacts/endpoints/), [all exploratory candidates](../artifacts/screen/),
[profiler reports](../artifacts/profiles/). Generator: [11_reduce.py](../11_reduce.py).
