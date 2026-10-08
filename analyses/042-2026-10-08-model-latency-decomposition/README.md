# Latency decomposition by model: tiled versus scalar thresholds

**Only the 14M checkpoints in this analysis have a tiled threshold. The 31M and 70M checkpoints use elementwise scalar thresholds.** Kernel tiles, token groups, feature unions and sparse matrix arithmetic do not imply a tiled threshold.

This analysis packages the completed Run064 and Run065 measurements requested by the user. It performs local reduction and plotting of saved evidence. The source runs, checkpoints, gate rules and execution policies remain unchanged. There is no new GPU benchmark, training or manuscript edit.

## Gate identity and execution identity

T2 means the gate topology **h and z**; it is not a tile-size label. Ph means the retained training intervention used OL1 pressure at h. At h, the gate replaces GELU; z is the concatenated attention context immediately before the attention output projection. All six checkpoints were retained after 712 updates; the present work measures inference.

| Model | Checkpoint source | Gate rule at h and z | Tiled threshold? | Selected sparse consumer |
| --- | --- | --- | --- | --- |
| 14M | Run060 | Keep an entire 8-token x 16-feature tile iff its top-left value > kappa; otherwise zero the whole tile | **YES** | Run063 corner-specialized joint h/z consumer |
| 31M | Run054 | Keep each element x iff x >= kappa; otherwise zero that element | **NO — scalar** | Frozen opt073-31m-v1 joint h/z consumer |
| 70M | Run043 | Keep each element x iff x >= kappa; otherwise zero that element | **NO — scalar** | Frozen Run051 sparse_c; gathered h in layers 1–5 |


The 14M tile rule rejects equality at the corner and preserves every signed interior value in a surviving tile, including negative/subthreshold values. The scalar rule keeps equality and applies independently to every element. Comparisons use the activation comparison dtype. At kappa = 0 the scalar rule is elementwise ReLU, while the tile rule still retains signed interiors when the corner survives.

**Native is dense arithmetic on the same gated checkpoint, not an ungated A0/dense-trained model.** A no-skip control still applies the trained threshold and produces its zeros; it disables arithmetic skipping in the chosen consumer. At 70M the control retains compaction and small-group padding, so it is not a best-dense-kernel reference.

| Model / execution variant | Tiled threshold? | What consumes the gated activations? |
| --- | --- | --- |
| 14M native | YES | Dense projections with the same corner gate |
| 14M generic_sparse | YES | Generic joint consumer inspects activation tiles and skips eligible work |
| 14M tile_sparse | YES | Specialized joint consumer reads tile corners directly and skips eligible h/z work |
| 14M generic_no_skip / tile_no_skip | YES | Same corresponding consumer with h/z skipping disabled |
| 31M native | NO | Dense projections with scalar h/z gates |
| 31M sparse | NO | Short-row prepass plus joint h/z fallback; eligible rows/tiles skip work |
| 31M no_skip | NO | Joint fallback with skipping disabled; prepass omitted |
| 70M native | NO | Dense projections with scalar h/z gates |
| 70M sparse | NO | h0 is dense; h1–5 compact feature unions and use gathered products; all z products stay dense |
| 70M no_skip | NO | Same scalar gates and conversion; h consumers compute the full K range |


## Complete-forward latency

BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes = 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, 338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. 14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.

Primary numbers below are pooled geometric means of synchronized host latency in milliseconds. The selected sparse mode is the 14M corner-specialized consumer and the existing scalar-threshold consumer at 31M/70M. Reductions use each point’s own reference.

| Model | Threshold rule | kappa | Native ms | Selected sparse ms | Selected no-skip ms | Reduction vs native | Reduction vs no-skip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | TILED 8x16 | 0 | 0.8378 | 0.5099 | 0.6596 | 39.13% | 22.69% |
| 14M | TILED 8x16 | 0.1 | 0.8349 | 0.4864 | 0.6578 | 41.74% | 26.05% |
| 31M | SCALAR (not tiled) | 0 | 1.0558 | 1.0069 | 1.0827 | 4.63% | 7.01% |
| 31M | SCALAR (not tiled) | 0.1 | 1.0556 | 0.8908 | 1.0826 | 15.61% | 17.71% |
| 70M | SCALAR (not tiled) | 0 | 1.7363 | 1.2463 | 1.8154 | 28.22% | 31.35% |
| 70M | SCALAR (not tiled) | 0.1 | 1.7318 | 1.1944 | 1.8090 | 31.03% | 33.98% |


[Figure 1 — full-forward latency, distributions and process repeats](figures/01-full-forward-latency.pdf)

Bars show pooled host geometric means; black whiskers are pooled sample p10-p90; three black diamonds show fresh-process geometric means. These are descriptive distributions, not confidence intervals. Green bars identify the selected sparse implementation. Hatching means arithmetic skipping is disabled; the trained gate stays active. All panels share the millisecond axis. 14M has a tiled gate in every mode, including native and generic. 31M/70M have scalar gates in every mode. Different gates/consumers and physical GPU sessions make the cross-size comparison descriptive.

**The 14M specialization gain is a different comparison:** against the generic consumer on the same tiled checkpoint, the corner consumer reduces host latency by 10.34%/9.00% at kappa 0/0.1. The larger native-to-sparse reductions also include frozen norm, layout, attention, head and dispatch optimizations. They cannot all be attributed to sparsity.

## Complete forward map

Position indices → token embedding → RoPE tables → six transformer blocks → final normalization → 50,304-way vocabulary projection. Each block has parallel attention/MLP residual topology, but its measured GPU stages execute serially on one stream.

| Step | Native dense execution | 14M corner / 31M scalar optimized execution | 70M scalar optimized execution |
| --- | --- | --- | --- |
| 1 | Separate attention and MLP layer norms | Paired norms | Paired norms |
| 2 | QKV projection | QKV projection | QKV projection |
| 3 | RoPE and QKV layout | Fused RoPE/layout | Fused RoPE/layout |
| 4 | Causal attention (QK, softmax, PV) | Fused attention; split/combine activities mapped | Fused attention; split/combine activities mapped |
| 5 | Context layout, z gate, z down projection | Context layout; raw z reaches the joint consumer | Context layout; z scalar gate fused with dense z down projection |
| 6 | MLP up projection, h gate, h down projection | MLP up projection; raw h reaches the joint consumer | MLP up projection; h0 gate + dense down, h1–5 compaction + gathered down |
| 7 | Attention + MLP + residual sum | Joint gate/down/residual consumer (31M includes prepass + fallback) | BF16 residual combination |


The joint stage and fused attention remain indivisible where the evidence has no independent wall-time boundary. Their ordered CUDA activities are available in the kernel inventory; internal SM-cycle probes from Run064 are not additive device latencies. [Every original forward stage](results/forward-stages.csv) and [every ordered CUDA activity](results/cuda-kernel-nodes.csv) retain finer detail.

## Per-model decomposition

Tables report arithmetic means of actual CUPTI CUDA activity durations, summed over all six layers where applicable. Shares divide by that mode’s GPU-active interval union; they are not percentages of uninstrumented host latency. **N/A** means no separate stage in that execution mode (often because it is fused); an observed 0.000 for context layout means a view with no CUDA activity.

### 14M: TILED threshold, corner-specialized sparse consumer

[Figure 2 — 14M complete GPU budget and per-layer branch costs](figures/02-14m-decomposition.pdf)

14M at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards; color and hatching identify execution categories. Absent/fused categories are not independent measurements; zero CUDA-activity context layout is a view. Lower panels sum the H/Z gates, down projections, feature compaction and residual activities within each layer, making native separate stages comparable with an indivisible fused stage. The layer bundle is an accounting group, not an additional timed kernel. All three model figures share the upper microsecond axis and lower vertical axis. Gaps and host overhead are excluded from these active budgets. No event or CUPTI number is rescaled to the primary uninstrumented host total. CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.

#### kappa = 0

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.888 | 0.837 | 0.869 | 0.18% |
| Token embedding | 2.197 | 2.173 | 2.216 | 0.46% |
| RoPE tables | 14.544 | 12.867 | 12.931 | 2.73% |
| Block layer norms | 41.986 | 10.485 | 10.453 | 2.23% |
| QKV projection | 24.154 | 24.349 | 24.341 | 5.17% |
| RoPE / QKV layout | 125.930 | 13.917 | 14.053 | 2.95% |
| Attention | 122.602 | 121.322 | 121.967 | 25.75% |
| Context layout | 0.000 | 10.232 | 10.253 | 2.17% |
| MLP up projection | 24.691 | 27.536 | 26.941 | 5.85% |
| H gate | 67.042 | N/A | N/A | N/A |
| Z gate | 69.333 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 14.955 | N/A | N/A | N/A |
| H down projection | 28.696 | N/A | N/A | N/A |
| Joint H/Z gates / down / residual | N/A | 63.397 | 210.299 | 13.46% |
| Residual sum | 11.976 | N/A | N/A | N/A |
| Final norm | 3.451 | 3.475 | 3.461 | 0.74% |
| Vocabulary projection | 187.447 | 180.482 | 183.385 | 38.31% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Corner sparse us | Corner, no skip us |
| --- | --- | --- | --- |
| 0 | 32.026 | 11.904 | 35.914 |
| 1 | 32.011 | 9.925 | 37.434 |
| 2 | 32.034 | 10.957 | 34.909 |
| 3 | 31.909 | 10.525 | 34.048 |
| 4 | 31.947 | 10.539 | 34.074 |
| 5 | 32.075 | 9.547 | 33.919 |


#### kappa = 0.1

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.885 | 0.840 | 0.888 | 0.19% |
| Token embedding | 2.192 | 2.211 | 2.237 | 0.49% |
| RoPE tables | 14.446 | 12.765 | 12.872 | 2.86% |
| Block layer norms | 41.931 | 10.442 | 10.413 | 2.34% |
| QKV projection | 24.208 | 24.288 | 24.312 | 5.43% |
| RoPE / QKV layout | 125.652 | 13.856 | 14.008 | 3.10% |
| Attention | 122.458 | 121.250 | 121.532 | 27.12% |
| Context layout | 0.000 | 10.219 | 10.227 | 2.29% |
| MLP up projection | 24.627 | 27.389 | 26.946 | 6.13% |
| H gate | 66.906 | N/A | N/A | N/A |
| Z gate | 69.234 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 14.861 | N/A | N/A | N/A |
| H down projection | 28.546 | N/A | N/A | N/A |
| Joint H/Z gates / down / residual | N/A | 40.568 | 208.276 | 9.08% |
| Residual sum | 11.925 | N/A | N/A | N/A |
| Final norm | 3.451 | 3.483 | 3.480 | 0.78% |
| Vocabulary projection | 186.687 | 179.700 | 183.180 | 40.20% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Corner sparse us | Corner, no skip us |
| --- | --- | --- | --- |
| 0 | 31.947 | 8.221 | 35.344 |
| 1 | 31.946 | 6.621 | 36.997 |
| 2 | 31.890 | 6.989 | 34.653 |
| 3 | 31.909 | 6.723 | 34.040 |
| 4 | 31.869 | 6.107 | 33.634 |
| 5 | 31.912 | 5.907 | 33.608 |


The specialized joint stage is cheaper than the generic joint stage on the same tiled activations. The remaining selected-sparse GPU-active budget is led by the vocabulary projection (38.31%/40.20%) and attention (25.75%/27.12%). The checkpoints’ previously measured quality penalty from training with tile gates is unchanged; faster inference does not repair it.

| kappa | Generic sparse host ms | Corner sparse host ms | Generic joint us | Corner joint us |
| --- | --- | --- | --- | --- |
| 0 | 0.5687 | 0.5099 | 121.025 | 63.397 |
| 0.1 | 0.5346 | 0.4864 | 89.324 | 40.568 |


### 31M: SCALAR threshold, not tiled

[Figure 3 — 31M complete GPU budget and per-layer branch costs](figures/03-31m-decomposition.pdf)

31M at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards; color and hatching identify execution categories. Absent/fused categories are not independent measurements; zero CUDA-activity context layout is a view. Lower panels sum the H/Z gates, down projections, feature compaction and residual activities within each layer, making native separate stages comparable with an indivisible fused stage. The layer bundle is an accounting group, not an additional timed kernel. All three model figures share the upper microsecond axis and lower vertical axis. Gaps and host overhead are excluded from these active budgets. No event or CUPTI number is rescaled to the primary uninstrumented host total. CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.

#### kappa = 0

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.968 | 0.901 | 0.923 | 0.09% |
| Token embedding | 2.595 | 2.445 | 2.515 | 0.25% |
| RoPE tables | 14.835 | 12.693 | 12.784 | 1.31% |
| Block layer norms | 43.419 | 14.952 | 14.763 | 1.54% |
| QKV projection | 49.160 | 48.451 | 49.941 | 4.99% |
| RoPE / QKV layout | 147.215 | 22.488 | 22.421 | 2.32% |
| Attention | 188.462 | 122.885 | 122.909 | 12.65% |
| Context layout | 0.000 | 0.000 | 0.000 | 0.00% |
| MLP up projection | 79.710 | 80.381 | 80.035 | 8.28% |
| H gate | 35.931 | N/A | N/A | N/A |
| Z gate | 18.621 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 27.165 | N/A | N/A | N/A |
| H down projection | 76.830 | N/A | N/A | N/A |
| Joint H/Z gates / down / residual | N/A | 431.555 | 506.451 | 44.43% |
| Residual sum | 14.333 | N/A | N/A | N/A |
| Final norm | 3.637 | 3.640 | 3.669 | 0.37% |
| Vocabulary projection | 281.373 | 230.982 | 232.512 | 23.78% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Scalar sparse us | Scalar, no skip us |
| --- | --- | --- | --- |
| 0 | 29.349 | 78.493 | 85.712 |
| 1 | 29.245 | 67.485 | 83.613 |
| 2 | 28.539 | 71.280 | 85.621 |
| 3 | 27.998 | 71.240 | 84.358 |
| 4 | 29.179 | 70.080 | 80.672 |
| 5 | 28.570 | 72.976 | 86.475 |


#### kappa = 0.1

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.965 | 0.896 | 0.915 | 0.10% |
| Token embedding | 2.597 | 2.493 | 2.488 | 0.29% |
| RoPE tables | 14.891 | 12.699 | 12.763 | 1.48% |
| Block layer norms | 43.480 | 14.795 | 14.690 | 1.73% |
| QKV projection | 48.821 | 48.501 | 49.474 | 5.66% |
| RoPE / QKV layout | 147.080 | 22.541 | 22.410 | 2.63% |
| Attention | 188.378 | 123.321 | 122.973 | 14.40% |
| Context layout | 0.000 | 0.000 | 0.000 | 0.00% |
| MLP up projection | 79.586 | 80.508 | 79.984 | 9.40% |
| H gate | 35.939 | N/A | N/A | N/A |
| Z gate | 18.604 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 27.099 | N/A | N/A | N/A |
| H down projection | 76.840 | N/A | N/A | N/A |
| Joint H/Z gates / down / residual | N/A | 315.837 | 511.135 | 36.89% |
| Residual sum | 14.368 | N/A | N/A | N/A |
| Final norm | 3.675 | 3.645 | 3.659 | 0.43% |
| Vocabulary projection | 279.354 | 230.926 | 232.176 | 26.97% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Scalar sparse us | Scalar, no skip us |
| --- | --- | --- | --- |
| 0 | 29.349 | 76.538 | 88.050 |
| 1 | 29.136 | 38.264 | 84.005 |
| 2 | 28.534 | 38.864 | 85.595 |
| 3 | 27.920 | 44.152 | 85.120 |
| 4 | 29.168 | 52.528 | 84.378 |
| 5 | 28.744 | 65.492 | 83.986 |


The joint stage takes 44.43%/36.89% of sparse GPU-active time, and costs 431.56/315.84 us versus 172.88/172.85 us for the complete native gate/down/residual bundle. Savings in other optimized stages and dispatch gaps compensate for this slower consumer. The kappa 0 short-row prepass completes no token row over the entire 338-block validation workload; kappa 0.1 helps mainly layers 1–3.

### 70M: SCALAR threshold, not tiled

[Figure 4 — 70M complete GPU budget and per-layer branch costs](figures/04-70m-decomposition.pdf)

70M at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards; color and hatching identify execution categories. Absent/fused categories are not independent measurements; zero CUDA-activity context layout is a view. Lower panels sum the H/Z gates, down projections, feature compaction and residual activities within each layer, making native separate stages comparable with an indivisible fused stage. The layer bundle is an accounting group, not an additional timed kernel. All three model figures share the upper microsecond axis and lower vertical axis. Gaps and host overhead are excluded from these active budgets. No event or CUPTI number is rescaled to the primary uninstrumented host total. CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.

#### kappa = 0

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.864 | 0.877 | 0.880 | 0.07% |
| Token embedding | 2.797 | 2.784 | 2.741 | 0.23% |
| RoPE tables | 13.347 | 12.301 | 12.315 | 1.02% |
| Block layer norms | 45.403 | 19.875 | 19.646 | 1.66% |
| QKV projection | 116.390 | 116.227 | 115.942 | 9.68% |
| RoPE / QKV layout | 182.267 | 34.963 | 34.813 | 2.91% |
| Attention | 341.840 | 218.123 | 217.669 | 18.17% |
| Context layout | 0.000 | 0.000 | 0.000 | 0.00% |
| MLP up projection | 146.096 | 147.072 | 146.298 | 12.25% |
| H gate | 60.185 | 3.587 | 3.600 | 0.30% |
| Z gate | 23.936 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 49.810 | 56.939 | 56.776 | 4.74% |
| H down projection | 151.718 | 115.171 | 690.146 | 9.59% |
| H feature compaction | N/A | 19.147 | 18.931 | 1.60% |
| Residual sum | 18.662 | 10.717 | 10.707 | 0.89% |
| Final norm | 3.768 | 3.896 | 3.944 | 0.32% |
| Vocabulary projection | 497.358 | 438.691 | 439.204 | 36.55% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Scalar sparse us | Scalar, no skip us |
| --- | --- | --- | --- |
| 0 | 50.731 | 40.440 | 40.328 |
| 1 | 50.707 | 25.192 | 125.305 |
| 2 | 50.765 | 29.010 | 185.272 |
| 3 | 50.701 | 30.029 | 184.982 |
| 4 | 50.630 | 38.587 | 184.885 |
| 5 | 50.776 | 42.301 | 59.387 |


#### kappa = 0.1

| Forward category | Native us | Selected sparse us | No-skip us | Selected sparse active share |
| --- | --- | --- | --- | --- |
| Position indices | 0.861 | 0.877 | 0.883 | 0.08% |
| Token embedding | 2.784 | 2.757 | 2.752 | 0.24% |
| RoPE tables | 13.317 | 12.299 | 12.309 | 1.07% |
| Block layer norms | 45.448 | 19.878 | 19.683 | 1.72% |
| QKV projection | 116.234 | 116.239 | 116.083 | 10.07% |
| RoPE / QKV layout | 182.035 | 34.981 | 34.824 | 3.03% |
| Attention | 341.966 | 218.413 | 218.024 | 18.92% |
| Context layout | 0.000 | 0.000 | 0.000 | 0.00% |
| MLP up projection | 146.120 | 146.725 | 146.426 | 12.71% |
| H gate | 60.269 | 3.579 | 3.608 | 0.31% |
| Z gate | 23.925 | N/A | N/A | N/A |
| Z down projection (gate fused if applicable) | 49.776 | 56.534 | 56.773 | 4.90% |
| H down projection | 151.757 | 69.638 | 689.573 | 6.03% |
| H feature compaction | N/A | 19.190 | 19.000 | 1.66% |
| Residual sum | 18.664 | 10.734 | 10.714 | 0.93% |
| Final norm | 3.776 | 3.880 | 3.936 | 0.34% |
| Vocabulary projection | 497.786 | 438.799 | 439.034 | 38.01% |


Comparable h/z branch bundle (gates + down projections + compaction + residual), by layer:

| Layer | Native (dense) us | Scalar sparse us | Scalar, no skip us |
| --- | --- | --- | --- |
| 0 | 50.784 | 40.416 | 40.344 |
| 1 | 50.712 | 19.473 | 125.256 |
| 2 | 50.762 | 20.822 | 185.141 |
| 3 | 50.725 | 23.240 | 184.821 |
| 4 | 50.634 | 25.974 | 184.872 |
| 5 | 50.773 | 29.749 | 59.235 |


Vocabulary projection is the largest remaining sparse component (36.55%/38.01%); attention takes 18.17%/18.92%. Actual h feature-union counters in the five gathered layers retain 7.815%/3.052% of the no-skip padded tensor-core logical products. Those counters are not measured speedups. All z products and h0 stay dense.

## Timing boundaries and qualification

| Model | kappa | Sparse active us | Trace gaps us | Trace span us | Unprofiled host geomean us | Event slowdown |
| --- | --- | --- | --- | --- | --- | --- |
| 14M | 0 | 471.072 | 26.909 | 497.982 | 509.939 | 1.239x |
| 14M | 0.1 | 447.011 | 27.119 | 474.130 | 486.441 | 1.249x |
| 31M | 0 | 971.373 | 24.291 | 995.664 | 1006.855 | 1.098x |
| 31M | 0.1 | 856.164 | 24.218 | 880.382 | 890.827 | 1.112x |
| 70M | 0 | 1200.369 | 29.215 | 1229.584 | 1246.257 | 1.120x |
| 70M | 0.1 | 1154.523 | 29.455 | 1183.978 | 1194.367 | 1.131x |


Active time plus gaps closes to the **profiled** first-to-last activity span. The span and unprofiled host timing use different measurement boundaries and profiling overhead, so their difference is not assigned to an invented stage. CUDA-event partition intervals are additive within their own instrumented graphs; events slow execution and are not rescaled into the baseline.

| Model | kappa | Native validation loss | Selected sparse loss | Sparse minus native | Complete-output comparisons |
| --- | --- | --- | --- | --- | --- |
| 14M | 0 | 5.55624849 | 5.55624849 | 0 | 10140 |
| 14M | 0.1 | 5.59878475 | 5.59878475 | 0 | 10140 |
| 31M | 0 | 4.69125580 | 4.69125607 | 2.72e-07 | 6084 |
| 31M | 0.1 | 4.75987041 | 4.75987064 | 2.33e-07 | 6084 |
| 70M | 0 | 4.13636840 | 4.13633542 | -3.3e-05 | 6084 |
| 70M | 0.1 | 4.19006060 | 4.19005415 | -6.45e-06 | 6084 |


All 44,616 complete-output comparisons passed the original criteria across 18 fresh profiling processes. Run064 is exact in logits and loss. Run065 passes per-logit atol 0.25 + rtol 0.02, relative L2 0.02 and full-loss atol 0.001; its largest absolute difference is 0.75 and relative L2 is about 0.0010043. A fixed absolute-only 0.25 claim would be incorrect. All 88 input/mode trace mappings succeeded in their source reductions.

The rows are different trained checkpoints, gates and consumers, with one training seed. They describe the observed workload; they do not isolate architecture-only scaling, establish a scaling law, test the corner-specialized kernel at 31M/70M, or cover autoregressive decoding, other batching/sequence lengths or hardware. Native-to-sparse includes all frozen component changes. Sparse-to-no-skip measures the net skipping effect within each implementation, including its overhead and padding.

### All execution modes and timing spread

Microseconds; p10/p90 describe the pooled 1,344 samples and are not confidence bounds.

| Model | kappa | Mode | Host geomean us | p10 us | p90 us | Process 1/2/3 geomean us |
| --- | --- | --- | --- | --- | --- | --- |
| 14M | 0 | native | 837.750 | 835.471 | 840.001 | 838.522/838.020/836.710 |
| 14M | 0 | generic_sparse | 568.726 | 564.951 | 572.661 | 569.473/568.579/568.126 |
| 14M | 0 | generic_no_skip | 635.799 | 633.360 | 638.661 | 635.681/636.174/635.543 |
| 14M | 0 | tile_sparse | 509.939 | 507.541 | 512.320 | 510.464/509.766/509.587 |
| 14M | 0 | tile_no_skip | 659.606 | 656.731 | 662.450 | 660.152/659.496/659.169 |
| 14M | 0.1 | native | 834.927 | 831.381 | 838.691 | 832.477/834.338/837.976 |
| 14M | 0.1 | generic_sparse | 534.562 | 530.231 | 538.971 | 532.976/534.268/536.447 |
| 14M | 0.1 | generic_no_skip | 633.721 | 630.300 | 637.431 | 631.609/633.633/635.928 |
| 14M | 0.1 | tile_sparse | 486.441 | 483.500 | 489.550 | 485.255/486.182/487.890 |
| 14M | 0.1 | tile_no_skip | 657.777 | 654.421 | 661.201 | 656.270/657.434/659.631 |
| 31M | 0 | native | 1055.761 | 1052.820 | 1058.689 | 1056.062/1055.708/1055.513 |
| 31M | 0 | sparse | 1006.855 | 1001.489 | 1012.189 | 1006.832/1007.274/1006.459 |
| 31M | 0 | no_skip | 1082.719 | 1072.230 | 1093.569 | 1083.080/1082.660/1082.418 |
| 31M | 0.1 | native | 1055.637 | 1052.389 | 1058.699 | 1053.717/1057.233/1055.964 |
| 31M | 0.1 | sparse | 890.827 | 866.360 | 912.550 | 889.654/892.220/890.610 |
| 31M | 0.1 | no_skip | 1082.609 | 1071.369 | 1092.739 | 1081.915/1083.248/1082.663 |
| 70M | 0 | native | 1736.296 | 1732.019 | 1740.019 | 1735.768/1736.243/1736.878 |
| 70M | 0 | sparse | 1246.257 | 1238.170 | 1253.960 | 1245.112/1245.604/1248.059 |
| 70M | 0 | no_skip | 1815.435 | 1809.289 | 1819.929 | 1814.441/1815.234/1816.631 |
| 70M | 0.1 | native | 1731.771 | 1724.639 | 1738.349 | 1731.217/1728.831/1735.272 |
| 70M | 0.1 | sparse | 1194.367 | 1186.760 | 1201.729 | 1195.242/1192.051/1195.810 |
| 70M | 0.1 | no_skip | 1809.038 | 1802.229 | 1815.369 | 1810.059/1806.618/1810.439 |


## Evidence and regeneration

[Run064 observation](../../runs/064-2026-10-07-pythia14m-forward-latency-decomposition/observations/01-forward-decomposition.md) · [Run065 observation](../../runs/065-2026-10-07-larger-model-forward-decomposition/observations/01-larger-forward-decomposition.md). The checkpoint sources are Run060 (14M), Run054 (31M) and Run043 (70M). Exact source/checkpoint identities and SHA256 values are in [source provenance](provenance/sources.json) and [combined evidence](results/decomposition.json). Figures have individual [observations](observations/INDEX.md).

[Full-forward timings](results/full-forward-latency.csv) · [Forward categories](results/forward-categories.csv) · [Every forward stage](results/forward-stages.csv) · [Per-layer branch bundle](results/branch-bundle-by-layer.csv) · [Every CUDA activity](results/cuda-kernel-nodes.csv).

Run locally from the repository root:

```powershell
python analyses/042-2026-10-08-model-latency-decomposition/01_collect.py
python analyses/042-2026-10-08-model-latency-decomposition/02_report.py
python analyses/042-2026-10-08-model-latency-decomposition/03_verify.py
```

These scripts read existing completed evidence and generate this analysis only. Publication figures are PDF only; links above open the figures. No source run is overwritten, no cloud resource is created, and no manuscript/finding promotion is implied.
