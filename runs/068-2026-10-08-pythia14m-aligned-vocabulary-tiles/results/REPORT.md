# Run068 — aligned sparse vocabulary head

Two fresh Pythia-14M models, head κ={0.1,0.5}, head16×16 gate, h/z8×16 κ=0.
Ph+head is OL1 pressure at six h tensors plus one head tensor, each weighted1/7; z is gated but has no direct pressure.
Initialization, seed1234, training order,712 updates and optimizer are matched.

The head gate masks the FP32 output of the learned affine final LayerNorm, before the vocabulary matrix multiplication.
A top-left value≤κ zeros its entire16-token×16-feature tile; live tiles retain signed interiors.
Weights and all50304 output logits remain dense. Training uses native dense matmul.
At inference, all h/z projections remain native dense. Only the custom sparse head bypasses MMA.

## Quality and occupancy

Both evaluations cover all500 MiniPile validation documents:338 complete2048-token blocks,692224 input tokens,
691886 predictions and1444 excluded tail tokens. Counts are pooled before division.
B32 native training evaluation and B1 inference counts are separate measurements.

| Head κ | Native loss (B32) | Perplexity | Head exact zeros (B32) | Empty16×16 tiles (B32) | Consumer/MMA bypass (B1) |
|---:|---:|---:|---:|---:|---:|
|0.1|5.560725|260.011|39.878%|39.878%|39.881%|
|0.5|5.882600|358.741|70.763%|70.763%|70.763%|

Raising only head κ from.1 to.5 increases final native validation loss by0.321875
and changes perplexity by a factor of1.3797.
No acceptable-quality cutoff was declared.

The M16/K16 consumer now matches the16×16 gate. In B1 inference, empty gate tiles, empty consumer tiles
and skipped/potential MMA fractions agree exactly. Real-operand device counters verify the branches over
all338 blocks in replicate1 for both custom modes. This removes the M8-to-M16 union mismatch of Run067.
Source-level work counters do not measure HBM traffic.

## Latency on one RTX5090

Synchronized host geometric means, milliseconds. Three fresh processes/model;64 fixed inputs×7 randomized
paired passes/process. Static weight packing, input staging, loss, softmax and token selection are outside timing.
CUDA-event measurements use separate replays. The three boundaries overlap and are not additive.

| Head κ | Boundary | Native dense | Custom no-skip | Custom sparse | Sparse reduction vs dense | Reduction vs no-skip |
|---:|---|---:|---:|---:|---:|---:|
|0.1|Projection only|0.196628|0.318962|0.203997|-3.75%|+36.04%|
|0.1|Gate + cast + projection|0.214132|0.339314|0.223645|-4.44%|+34.09%|
|0.1|Complete forward|0.983987|1.109741|0.994169|-1.03%|+10.41%|
|0.5|Projection only|0.197138|0.321092|0.162489|+17.58%|+49.39%|
|0.5|Gate + cast + projection|0.213253|0.339114|0.180095|+15.55%|+46.89%|
|0.5|Complete forward|1.021812|1.148920|0.990146|+3.10%|+13.82%|

Positive reductions mean faster sparse execution; negative values mean slower.
The no-skip control isolates bypass inside the custom package. The dense comparison also includes layout/preparation costs.

## Numerical qualification and conclusion

| Head κ | B1 native loss | Maximum absolute logit error, all modes/processes | Maximum pooled loss delta | Full-forward reduction range |
|---:|---:|---:|---:|---:|
|0.1|5.560689779|0|0|-1.10% to-1.00%|
|0.5|5.882653701|0|0|+2.97% to+3.18%|

All six processes passed full338-block native-reference qualification and standalone head-graph checks
under the unchanged numerical bounds. B1 native losses are the kernel reference; B32 training losses are quality measurements.

For head κ=0.1, custom sparse execution increases head-path latency by4.44% and increases complete-forward latency by1.03% versus native dense.
For head κ=0.5, custom sparse execution reduces head-path latency by15.55% and reduces complete-forward latency by3.10% versus native dense.

The two-model training comparison isolates head threshold at fixed alignment and pressure. Comparisons with Run066
are historical context and do not isolate alignment alone. Each within-model runtime comparison keeps its checkpoint,
gate and complete vocabulary contract matched. One training seed/model size and one GPU/workload limit generalization;
these measurements do not establish cached-decoding performance or benefits from integrating earlier block kernels.

Sources: [native training reduction](training-comparison.json), [endpoint reduction](../inference/results/comparison.json),
[training transfer receipt](../prelaunch/retrieval-training-001.json), [endpoint transfer receipt](../inference/transfer/receipt-001.json).
Figures: [quality and aligned occupancy](../figures/01-quality-and-aligned-occupancy.pdf),
[dense/no-skip/sparse latency](../figures/02-dense-and-sparse-endpoint-latency.pdf).
Methods/captions: [observations](../observations/INDEX.md). Generating script: [21_summary.py](../21_summary.py).
