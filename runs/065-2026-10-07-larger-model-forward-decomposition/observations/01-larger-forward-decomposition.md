# Existing 31M/70M full-forward latency decomposition

Question: does the latency distribution and sparse benefit persist in the retained larger T2/Ph models?

Method/coverage: one RTX5090, BF16 batch1,2048 tokens, uncached full50304-vocabulary output. Four final712-update checkpoints; three fresh processes each;64 matched random inputs ×7 randomized paired passes. Every baseline and mapped graph qualified against its own native eager model on all338 complete validation blocks from500 documents;1444 tail tokens excluded. Exact checkpoint/cache/source hashes are retained. No training, weights, threshold, gate or kernel policy was changed.

The31M frozen consumer is Run054 opt073-31m-v1;70M is Run051 sparse_c. Both consume elementwise one-sided h/z thresholds. Run06414M consumes corner-selected8x16 tiles, so the cross-size comparison is descriptive and includes differences in gate and implementation. It cannot establish architecture-only scaling of the new14M corner kernel.

Uninstrumented complete-forward synchronized host latency, geometric means in milliseconds. Controls retain the same gates and non-sparse components. The70M no-skip control also retains compaction, then computes the full K range.

| Model | κ | Native ms | Sparse ms | No-skip ms | Native / sparse | Sparse reduction vs no-skip |
| --- | --- | --- | --- | --- | --- | --- |
| 31m | 0 | 1.0558 | 1.0069 | 1.0827 | 1.049x | 7.01% |
| 31m | 0.1 | 1.0556 | 0.8908 | 1.0826 | 1.185x | 17.71% |
| 70m | 0 | 1.7363 | 1.2463 | 1.8154 | 1.393x | 31.35% |
| 70m | 0.1 | 1.7318 | 1.1944 | 1.8090 | 1.450x | 33.98% |


Result: the distribution changes with size and implementation.31M full latency falls4.63%/15.61% versus native, but its fused joint stage takes431.56/315.84µs versus172.88/172.85µs for native gates/down/residual. Its other component and dispatch savings compensate for that slower joint consumer. At κ0 the short-row prepass completes no token row in the entire338-block workload; at κ.1 it helps mainly layers1–3.70M full latency falls28.22%/31.03%; its five gathered h consumers benefit strongly from skipping, and vocabulary projection becomes its largest component at36.55%/38.01% of sparse GPU-active time. This does not reproduce a uniform cross-size bottleneck or demonstrate the new14M corner consumer at larger sizes.

| Model / κ | Native gates/down/residual µs | Sparse µs | No-skip µs |
| --- | --- | --- | --- |
| 31m-k0 | 172.881 | 431.555 | 506.451 |
| 31m-k0p1 | 172.851 | 315.837 | 511.135 |
| 70m-k0 | 304.310 | 205.560 | 780.159 |
| 70m-k0p1 | 304.391 | 159.674 | 779.668 |


Descriptive context against Run06414M (a different physical RTX5090 session and a different gate/consumer). These rows compare observed bottlenecks, not a matched architecture-only effect:

| Model / gate | κ | Sparse host ms | Vocabulary active share | Attention active share |
| --- | --- | --- | --- | --- |
| 14M corner tiles | 0 | 0.5099 | 38.31% | 25.75% |
| 14M corner tiles | 0.1 | 0.4864 | 40.20% | 27.12% |
| 31M scalar | 0 | 1.0069 | 23.78% | 12.65% |
| 31M scalar | 0.1 | 0.8908 | 26.97% | 14.40% |
| 70M scalar | 0 | 1.2463 | 36.55% | 18.17% |
| 70M scalar | 0.1 | 1.1944 | 38.01% | 18.92% |


Forward map: position indices → token embedding → RoPE tables → six transformer blocks with parallel attention/MLP residual topology → final norm → vocabulary projection. Each block computes attention and MLP from the same residual input; these measurements execute the GPU stages serially on one stream. Native: attention norm → QKV → rotary/layout → attention → context layout → z gate → z down; MLP norm → MLP up → h gate → h down; then both branches + residual. Optimized paired norm computes both norm outputs once.31M combines scalar inspection/short-row handling, h/z down projections, BF16 branch rounding and residual into the short-row prepass and joint fallback.70M exposes h0 gate + dense linear, h1..5 feature compaction + gathered linear, z fused gate + dense dot, and BF16 residual combination. Context layout may have zero CUDA activities when it is a view.

Primary component numbers below are summed actual CUPTI activity durations. Four timing inputs/process1 ×3 graph replays per mode; all computational kernel names and launch order must match the annotated eager map. The narrowly declared native graph-only memcpy mapping retains the original strict failure and attributes only enclosed gate copy nodes. GPU active time is an interval union; gaps are measured separately. CUDA-event partitions are additive but perturb execution; their overhead is reported and never rescaled into the baseline. Timing distributions, every layer/stage and every ordered CUDA activity are in the CSVs.

## 31m-k0

| Step category | Native µs | Sparse µs | No-skip µs | Sparse active share |
| --- | --- | --- | --- | --- |
| Positions | 0.968 | 0.901 | 0.923 | 0.09% |
| Embedding | 2.595 | 2.445 | 2.515 | 0.25% |
| RoPE tables | 14.835 | 12.693 | 12.784 | 1.31% |
| Layer norms | 43.419 | 14.952 | 14.763 | 1.54% |
| QKV projection | 49.160 | 48.451 | 49.941 | 4.99% |
| RoPE / QKV layout | 147.215 | 22.488 | 22.421 | 2.32% |
| Attention | 188.462 | 122.885 | 122.909 | 12.65% |
| MLP up projection | 79.710 | 80.381 | 80.035 | 8.28% |
| Scalar gates | 54.552 | N/A | N/A | N/A |
| Z down projection (+ fused gate) | 27.165 | N/A | N/A | N/A |
| H down projection | 76.830 | N/A | N/A | N/A |
| Residual sum | 14.333 | N/A | N/A | N/A |
| Joint gates / down / residual | N/A | 431.555 | 506.451 | 44.43% |
| Final norm | 3.637 | 3.640 | 3.669 | 0.37% |
| Vocabulary projection | 281.373 | 230.982 | 232.512 | 23.78% |


| Accounting | Native | Sparse | No-skip |
| --- | --- | --- | --- |
| GPU active µs | 984.254 | 971.373 | 1048.924 |
| First-to-last activity span µs | 1061.799 | 995.664 | 1070.097 |
| Inter-activity gaps µs | 77.546 | 24.291 | 21.173 |
| Host minus outer CUDA mean µs | 10.649 | 10.569 | 10.580 |
| Equal input staging mean µs (excluded) | 17.621 | 17.321 | 16.997 |
| Event instrumentation CUDA slowdown | 1.168x | 1.098x | 1.098x |
| Activities per graph | 182 | 56 | 50 |


Fresh-process host geometric means and pooled sample distributions (µs):

| Mode | Process1 | Process2 | Process3 | Pooled median | Pooled p10 | Pooled p90 |
| --- | --- | --- | --- | --- | --- | --- |
| native | 1056.062 | 1055.708 | 1055.513 | 1055.514 | 1052.820 | 1058.689 |
| sparse | 1006.832 | 1007.274 | 1006.459 | 1006.840 | 1001.489 | 1012.189 |
| no_skip | 1083.080 | 1082.660 | 1082.418 | 1083.174 | 1072.230 | 1093.569 |


Per-layer sparse CUDA-active stage measurements (µs):

| Layer | norm_pair | qkv | rope_qkv_layout | attention | context_pack | mlp_up | joint_gate_down_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.493 | 8.245 | 3.792 | 20.661 | 0.000 | 13.384 | 78.493 |
| 1 | 2.501 | 7.989 | 3.752 | 20.472 | 0.000 | 13.325 | 67.485 |
| 2 | 2.499 | 7.955 | 3.723 | 20.437 | 0.000 | 13.509 | 71.280 |
| 3 | 2.483 | 8.264 | 3.739 | 20.424 | 0.000 | 13.363 | 71.240 |
| 4 | 2.517 | 7.965 | 3.733 | 20.413 | 0.000 | 13.469 | 70.080 |
| 5 | 2.459 | 8.032 | 3.749 | 20.477 | 0.000 | 13.331 | 72.976 |


The fused31M stage contains these separately measured CUDA activities. Their summed durations expose the short-row prepass cost and fallback cost, but do not separate the internal h/z inspection and overlapping MMA phases:

| Mode | Activity role | Summed over6 layers µs |
| --- | --- | --- |
| sparse | short-row prepass | 19.149 |
| sparse | joint fallback | 412.406 |
| no_skip | short-row prepass | 0.000 |
| no_skip | joint fallback | 506.451 |


Actual sparse-consumer gated operands, integer counts pooled over338 blocks:

| Site.layer | Exact zeros | Rows ≤8 nonzeros | Empty8×16 tiles | RMS |
| --- | --- | --- | --- | --- |
| h.0 | 84.69% | 0.00% | 0.01% | 0.147732 |
| z.0 | 66.52% | 0.00% | 0.00% | 0.109706 |
| h.1 | 98.38% | 19.57% | 18.48% | 0.032841 |
| z.1 | 78.21% | 0.00% | 0.88% | 0.220317 |
| h.2 | 97.73% | 3.94% | 10.76% | 0.038802 |
| z.2 | 81.96% | 0.00% | 0.07% | 0.175662 |
| h.3 | 97.32% | 0.54% | 8.03% | 0.048415 |
| z.3 | 78.24% | 0.00% | 1.86% | 0.214835 |
| h.4 | 96.21% | 0.02% | 2.98% | 0.069434 |
| z.4 | 79.66% | 0.00% | 7.25% | 0.219114 |
| h.5 | 93.21% | 0.00% | 0.74% | 0.130305 |
| z.5 | 63.71% | 0.00% | 0.01% | 0.348210 |


Actual31M prepass decisions, pooled over338blocks. A state9 includes >8 entries or unsafe values. Both branches short for every row of an8-token group permits the fallback CTA to return early:

| Layer | Rows both branches short | 8-token blocks all short |
| --- | --- | --- |
| joint.0 | 0.00% | 0.00% |
| joint.1 | 0.00% | 0.00% |
| joint.2 | 0.00% | 0.00% |
| joint.3 | 0.00% | 0.00% |
| joint.4 | 0.00% | 0.00% |
| joint.5 | 0.00% | 0.00% |


## 31m-k0.1

| Step category | Native µs | Sparse µs | No-skip µs | Sparse active share |
| --- | --- | --- | --- | --- |
| Positions | 0.965 | 0.896 | 0.915 | 0.10% |
| Embedding | 2.597 | 2.493 | 2.488 | 0.29% |
| RoPE tables | 14.891 | 12.699 | 12.763 | 1.48% |
| Layer norms | 43.480 | 14.795 | 14.690 | 1.73% |
| QKV projection | 48.821 | 48.501 | 49.474 | 5.66% |
| RoPE / QKV layout | 147.080 | 22.541 | 22.410 | 2.63% |
| Attention | 188.378 | 123.321 | 122.973 | 14.40% |
| MLP up projection | 79.586 | 80.508 | 79.984 | 9.40% |
| Scalar gates | 54.543 | N/A | N/A | N/A |
| Z down projection (+ fused gate) | 27.099 | N/A | N/A | N/A |
| H down projection | 76.840 | N/A | N/A | N/A |
| Residual sum | 14.368 | N/A | N/A | N/A |
| Joint gates / down / residual | N/A | 315.837 | 511.135 | 36.89% |
| Final norm | 3.675 | 3.645 | 3.659 | 0.43% |
| Vocabulary projection | 279.354 | 230.926 | 232.176 | 26.97% |


| Accounting | Native | Sparse | No-skip |
| --- | --- | --- | --- |
| GPU active µs | 981.678 | 856.164 | 1052.666 |
| First-to-last activity span µs | 1059.300 | 880.382 | 1073.782 |
| Inter-activity gaps µs | 77.622 | 24.218 | 21.116 |
| Host minus outer CUDA mean µs | 10.515 | 10.469 | 10.519 |
| Equal input staging mean µs (excluded) | 17.269 | 16.733 | 16.993 |
| Event instrumentation CUDA slowdown | 1.168x | 1.112x | 1.098x |
| Activities per graph | 182 | 56 | 50 |


Fresh-process host geometric means and pooled sample distributions (µs):

| Mode | Process1 | Process2 | Process3 | Pooled median | Pooled p10 | Pooled p90 |
| --- | --- | --- | --- | --- | --- | --- |
| native | 1053.717 | 1057.233 | 1055.964 | 1055.690 | 1052.389 | 1058.699 |
| sparse | 889.654 | 892.220 | 890.610 | 892.590 | 866.360 | 912.550 |
| no_skip | 1081.915 | 1083.248 | 1082.663 | 1083.200 | 1071.369 | 1092.739 |


Per-layer sparse CUDA-active stage measurements (µs):

| Layer | norm_pair | qkv | rope_qkv_layout | attention | context_pack | mlp_up | joint_gate_down_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.515 | 8.216 | 3.789 | 20.738 | 0.000 | 13.437 | 76.538 |
| 1 | 2.499 | 7.979 | 3.760 | 20.530 | 0.000 | 13.360 | 38.264 |
| 2 | 2.493 | 8.056 | 3.747 | 20.490 | 0.000 | 13.467 | 38.864 |
| 3 | 2.400 | 8.371 | 3.747 | 20.517 | 0.000 | 13.368 | 44.152 |
| 4 | 2.419 | 7.960 | 3.747 | 20.517 | 0.000 | 13.475 | 52.528 |
| 5 | 2.469 | 7.920 | 3.752 | 20.528 | 0.000 | 13.402 | 65.492 |


The fused31M stage contains these separately measured CUDA activities. Their summed durations expose the short-row prepass cost and fallback cost, but do not separate the internal h/z inspection and overlapping MMA phases:

| Mode | Activity role | Summed over6 layers µs |
| --- | --- | --- |
| sparse | short-row prepass | 35.877 |
| sparse | joint fallback | 279.960 |
| no_skip | short-row prepass | 0.000 |
| no_skip | joint fallback | 511.135 |


Actual sparse-consumer gated operands, integer counts pooled over338 blocks:

| Site.layer | Exact zeros | Rows ≤8 nonzeros | Empty8×16 tiles | RMS |
| --- | --- | --- | --- | --- |
| h.0 | 92.83% | 0.00% | 0.14% | 0.105359 |
| z.0 | 97.61% | 83.63% | 54.81% | 0.057928 |
| h.1 | 99.81% | 97.95% | 79.79% | 0.019180 |
| z.1 | 96.50% | 46.98% | 47.96% | 0.112016 |
| h.2 | 99.70% | 96.38% | 71.30% | 0.022346 |
| z.2 | 97.58% | 76.96% | 54.09% | 0.077226 |
| h.3 | 99.58% | 89.10% | 64.23% | 0.027117 |
| z.3 | 97.12% | 70.66% | 45.26% | 0.101665 |
| h.4 | 99.35% | 74.25% | 51.35% | 0.037936 |
| z.4 | 93.49% | 1.06% | 19.95% | 0.123957 |
| h.5 | 98.08% | 3.51% | 20.26% | 0.089299 |
| z.5 | 91.10% | 0.24% | 8.49% | 0.217765 |


Actual31M prepass decisions, pooled over338blocks. A state9 includes >8 entries or unsafe values. Both branches short for every row of an8-token group permits the fallback CTA to return early:

| Layer | Rows both branches short | 8-token blocks all short |
| --- | --- | --- |
| joint.0 | 0.00% | 0.00% |
| joint.1 | 46.33% | 5.44% |
| joint.2 | 75.03% | 52.26% |
| joint.3 | 67.43% | 39.90% |
| joint.4 | 0.70% | 0.00% |
| joint.5 | 0.00% | 0.00% |


## 70m-k0

| Step category | Native µs | Sparse µs | No-skip µs | Sparse active share |
| --- | --- | --- | --- | --- |
| Positions | 0.864 | 0.877 | 0.880 | 0.07% |
| Embedding | 2.797 | 2.784 | 2.741 | 0.23% |
| RoPE tables | 13.347 | 12.301 | 12.315 | 1.02% |
| Layer norms | 45.403 | 19.875 | 19.646 | 1.66% |
| QKV projection | 116.390 | 116.227 | 115.942 | 9.68% |
| RoPE / QKV layout | 182.267 | 34.963 | 34.813 | 2.91% |
| Attention | 341.840 | 218.123 | 217.669 | 18.17% |
| MLP up projection | 146.096 | 147.072 | 146.298 | 12.25% |
| Scalar gates | 84.121 | 3.587 | 3.600 | 0.30% |
| Z down projection (+ fused gate) | 49.810 | 56.939 | 56.776 | 4.74% |
| H down projection | 151.718 | 115.171 | 690.146 | 9.59% |
| H feature compaction | N/A | 19.147 | 18.931 | 1.60% |
| Residual sum | 18.662 | 10.717 | 10.707 | 0.89% |
| Final norm | 3.768 | 3.896 | 3.944 | 0.32% |
| Vocabulary projection | 497.358 | 438.691 | 439.204 | 36.55% |


| Accounting | Native | Sparse | No-skip |
| --- | --- | --- | --- |
| GPU active µs | 1654.441 | 1200.369 | 1773.611 |
| First-to-last activity span µs | 1732.081 | 1229.584 | 1802.481 |
| Inter-activity gaps µs | 77.641 | 29.215 | 28.870 |
| Host minus outer CUDA mean µs | 10.692 | 10.622 | 10.658 |
| Equal input staging mean µs (excluded) | 17.855 | 17.935 | 18.162 |
| Event instrumentation CUDA slowdown | 1.107x | 1.120x | 1.085x |
| Activities per graph | 182 | 68 | 68 |


Fresh-process host geometric means and pooled sample distributions (µs):

| Mode | Process1 | Process2 | Process3 | Pooled median | Pooled p10 | Pooled p90 |
| --- | --- | --- | --- | --- | --- | --- |
| native | 1735.768 | 1736.243 | 1736.878 | 1736.579 | 1732.019 | 1740.019 |
| sparse | 1245.112 | 1245.604 | 1248.059 | 1246.465 | 1238.170 | 1253.960 |
| no_skip | 1814.441 | 1815.234 | 1816.631 | 1815.939 | 1809.289 | 1819.929 |


Per-layer sparse CUDA-active stage measurements (µs):

| Layer | norm_pair | qkv | rope_qkv_layout | attention | context_pack | mlp_up | h_gate | h_dense_down | z_gate_dense_down | residual | h_feature_compaction | h_gathered_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.984 | 19.187 | 5.792 | 36.373 | 0.000 | 24.485 | 3.587 | 25.510 | 9.488 | 1.856 | N/A | N/A |
| 1 | 3.384 | 19.392 | 5.864 | 36.525 | 0.000 | 24.627 | N/A | N/A | 9.488 | 1.797 | 3.381 | 10.525 |
| 2 | 3.384 | 19.424 | 5.808 | 36.334 | 0.000 | 24.469 | N/A | N/A | 9.440 | 1.786 | 4.221 | 13.563 |
| 3 | 3.419 | 19.469 | 5.846 | 36.413 | 0.000 | 24.464 | N/A | N/A | 9.480 | 1.765 | 4.219 | 14.565 |
| 4 | 3.371 | 19.411 | 5.835 | 36.261 | 0.000 | 24.432 | N/A | N/A | 9.544 | 1.765 | 4.224 | 23.053 |
| 5 | 3.333 | 19.344 | 5.819 | 36.216 | 0.000 | 24.595 | N/A | N/A | 9.499 | 1.747 | 3.101 | 27.955 |


Actual sparse-consumer gated operands, integer counts pooled over338 blocks:

| Site.layer | Exact zeros | Rows ≤8 nonzeros | Empty8×16 tiles | RMS |
| --- | --- | --- | --- | --- |
| h.0 | 93.25% | 0.00% | 0.22% | 0.111935 |
| z.0 | 76.16% | 0.00% | 0.34% | 0.120698 |
| h.1 | 99.33% | 26.08% | 50.51% | 0.030939 |
| z.1 | 89.67% | 0.00% | 9.72% | 0.163007 |
| h.2 | 98.91% | 3.31% | 33.40% | 0.039507 |
| z.2 | 91.59% | 0.00% | 21.36% | 0.123906 |
| h.3 | 98.71% | 0.55% | 27.35% | 0.042927 |
| z.3 | 85.92% | 0.00% | 4.70% | 0.223929 |
| h.4 | 97.68% | 0.00% | 11.28% | 0.072587 |
| z.4 | 82.58% | 0.00% | 9.43% | 0.276615 |
| h.5 | 94.13% | 0.00% | 1.65% | 0.152211 |
| z.5 | 88.03% | 0.00% | 4.44% | 0.245446 |


Actual70M feature unions read directly from the frozen index kernel after each forward. Empty8×16tiles above are a geometric diagnostic; this consumer uses the declared GM groups and32-wide K loops. Rounded K includes MMA padding:

| Site.layer | GM | Mean union K /2048 | Mean rounded K /2048 | Rounded K / full K |
| --- | --- | --- | --- | --- |
| h.1 | 8 | 87.383 | 102.907 | 5.02% |
| h.2 | 4 | 78.786 | 94.227 | 4.60% |
| h.3 | 4 | 91.800 | 107.241 | 5.24% |
| h.4 | 4 | 156.360 | 171.844 | 8.39% |
| h.5 | 16 | 686.312 | 701.806 | 34.27% |


Pooled padded tensorcore product count in these five h consumers is 850,912,608,256 versus 10,887,742,095,360 for their no-skip paths (7.815% retained). Integer counts are pooled before division. This is a logical executed-operand counter, not a runtime speedup;16-row MMA padding means small GM no-skip consumers are not the best dense kernels.

## 70m-k0.1

| Step category | Native µs | Sparse µs | No-skip µs | Sparse active share |
| --- | --- | --- | --- | --- |
| Positions | 0.861 | 0.877 | 0.883 | 0.08% |
| Embedding | 2.784 | 2.757 | 2.752 | 0.24% |
| RoPE tables | 13.317 | 12.299 | 12.309 | 1.07% |
| Layer norms | 45.448 | 19.878 | 19.683 | 1.72% |
| QKV projection | 116.234 | 116.239 | 116.083 | 10.07% |
| RoPE / QKV layout | 182.035 | 34.981 | 34.824 | 3.03% |
| Attention | 341.966 | 218.413 | 218.024 | 18.92% |
| MLP up projection | 146.120 | 146.725 | 146.426 | 12.71% |
| Scalar gates | 84.194 | 3.579 | 3.608 | 0.31% |
| Z down projection (+ fused gate) | 49.776 | 56.534 | 56.773 | 4.90% |
| H down projection | 151.757 | 69.638 | 689.573 | 6.03% |
| H feature compaction | N/A | 19.190 | 19.000 | 1.66% |
| Residual sum | 18.664 | 10.734 | 10.714 | 0.93% |
| Final norm | 3.776 | 3.880 | 3.936 | 0.34% |
| Vocabulary projection | 497.786 | 438.799 | 439.034 | 38.01% |


| Accounting | Native | Sparse | No-skip |
| --- | --- | --- | --- |
| GPU active µs | 1654.719 | 1154.523 | 1773.622 |
| First-to-last activity span µs | 1732.596 | 1183.978 | 1802.499 |
| Inter-activity gaps µs | 77.877 | 29.455 | 28.878 |
| Host minus outer CUDA mean µs | 11.211 | 11.184 | 11.220 |
| Equal input staging mean µs (excluded) | 19.479 | 19.779 | 19.231 |
| Event instrumentation CUDA slowdown | 1.108x | 1.131x | 1.088x |
| Activities per graph | 182 | 68 | 68 |


Fresh-process host geometric means and pooled sample distributions (µs):

| Mode | Process1 | Process2 | Process3 | Pooled median | Pooled p10 | Pooled p90 |
| --- | --- | --- | --- | --- | --- | --- |
| native | 1731.217 | 1728.831 | 1735.272 | 1731.414 | 1724.639 | 1738.349 |
| sparse | 1195.242 | 1192.051 | 1195.810 | 1194.209 | 1186.760 | 1201.729 |
| no_skip | 1810.059 | 1806.618 | 1810.439 | 1808.879 | 1802.229 | 1815.369 |


Per-layer sparse CUDA-active stage measurements (µs):

| Layer | norm_pair | qkv | rope_qkv_layout | attention | context_pack | mlp_up | h_gate | h_dense_down | z_gate_dense_down | residual | h_feature_compaction | h_gathered_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.966 | 19.184 | 5.797 | 36.371 | 0.000 | 24.403 | 3.579 | 25.525 | 9.459 | 1.853 | N/A | N/A |
| 1 | 3.373 | 19.347 | 5.864 | 36.517 | 0.000 | 24.611 | N/A | N/A | 9.408 | 1.806 | 3.502 | 4.757 |
| 2 | 3.365 | 19.472 | 5.813 | 36.448 | 0.000 | 24.534 | N/A | N/A | 9.363 | 1.781 | 4.235 | 5.443 |
| 3 | 3.413 | 19.461 | 5.840 | 36.480 | 0.000 | 24.434 | N/A | N/A | 9.363 | 1.760 | 4.219 | 7.899 |
| 4 | 3.379 | 19.440 | 5.843 | 36.331 | 0.000 | 24.344 | N/A | N/A | 9.427 | 1.776 | 4.229 | 10.541 |
| 5 | 3.382 | 19.336 | 5.824 | 36.266 | 0.000 | 24.400 | N/A | N/A | 9.515 | 1.757 | 3.005 | 15.472 |


Actual sparse-consumer gated operands, integer counts pooled over338 blocks:

| Site.layer | Exact zeros | Rows ≤8 nonzeros | Empty8×16 tiles | RMS |
| --- | --- | --- | --- | --- |
| h.0 | 96.87% | 0.01% | 3.93% | 0.079216 |
| z.0 | 96.90% | 17.46% | 47.31% | 0.064088 |
| h.1 | 99.85% | 95.49% | 84.78% | 0.020255 |
| z.1 | 98.09% | 45.26% | 53.88% | 0.098772 |
| h.2 | 99.82% | 92.55% | 81.33% | 0.022746 |
| z.2 | 96.28% | 4.53% | 33.08% | 0.114598 |
| h.3 | 99.61% | 61.43% | 65.11% | 0.034364 |
| z.3 | 94.84% | 6.30% | 38.79% | 0.154892 |
| h.4 | 99.27% | 21.01% | 48.45% | 0.050320 |
| z.4 | 96.92% | 7.60% | 40.73% | 0.105468 |
| h.5 | 98.10% | 0.07% | 23.06% | 0.112144 |
| z.5 | 98.03% | 37.32% | 58.71% | 0.112319 |


Actual70M feature unions read directly from the frozen index kernel after each forward. Empty8×16tiles above are a geometric diagnostic; this consumer uses the declared GM groups and32-wide K loops. Rounded K includes MMA padding:

| Site.layer | GM | Mean union K /2048 | Mean rounded K /2048 | Rounded K / full K |
| --- | --- | --- | --- | --- |
| h.1 | 8 | 21.211 | 34.835 | 1.70% |
| h.2 | 4 | 14.243 | 33.098 | 1.62% |
| h.3 | 4 | 29.701 | 43.251 | 2.11% |
| h.4 | 4 | 52.445 | 68.980 | 3.37% |
| h.5 | 16 | 271.143 | 286.638 | 14.00% |


Pooled padded tensorcore product count in these five h consumers is 332,311,298,048 versus 10,887,742,095,360 for their no-skip paths (3.052% retained). Integer counts are pooled before division. This is a logical executed-operand counter, not a runtime speedup;16-row MMA padding means small GM no-skip consumers are not the best dense kernels.

## Numerical and interpretation limits

All12 fresh processes pass the unchanged bounds: 24,336 complete-output comparisons. Maximum absolute logit error across the cohort is 0.75; exact equality is reported only when observed. Loss values and per-block numerical evidence remain in results/decomposition.json and quality.json.

Sparse versus native includes all frozen component optimizations (paired norms, attention/layout and head), while sparse versus no-skip isolates the net skipping effect in each existing implementation.70M uses sparse h only in five layers; z remains dense. A no-skip control that retains conversion is not the best possible dense kernel. Changing the gate structure or porting the14M corner-specialized consumer would require a separate experiment. Panels share the complete-forward horizontal scale; per-condition and per-layer detail panels use their own vertical scales. Single-seed, full-sequence batch1 results do not cover autoregressive decode, other sequence lengths, batching, training, or other GPUs. Raw CUPTI profiling and events have their own overhead and boundary definitions; no phase is assigned a fraction of uninstrumented wall time by rescaling.

Source scripts:20_analyze.py and21_report.py. Caption/legend: colors/patterns identify forward stages; bars sum actual GPU activities, lines show sparse per-layer means; bands show p10–p90 across12 profiled replays (a descriptive spread, not a confidence interval). Zero-valued layout stages are views. All figures are PDF only.

[Publication PDF](../figures/01-larger-forward-decomposition.pdf) · [All stage timings](../results/forward-stages.csv) · [Every CUDA activity](../results/cuda-kernel-nodes.csv) · [Verified result](../results/decomposition.json)
