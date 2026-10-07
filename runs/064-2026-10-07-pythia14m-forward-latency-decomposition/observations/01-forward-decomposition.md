# Full forward inference latency distribution

Question: where is full-sequence inference time spent, and which stages explain the generic-to-specialized difference?

## Measured result

Accepted evidence is the six complete fresh processes in infrastructure attempt 003. The two earlier infrastructure failures are preserved but excluded. All 20,280 full-block logit comparisons have zero absolute and relative-L2 error; losses are exactly 5.556248486078721 and 5.5987847543876 at kappa 0/.1. The generic and specialized consumers also have identical full-validation activation, tile and MMA/SIMT counters.

At kappa 0, synchronized host latency falls 568.726 to 509.939 us (10.337%). The six joint gate/down-projection/residual kernels fall 121.025 to 63.397 us in the CUPTI sample, a 57.628-us saving. Total kernel-active time falls 58.104 us: the changed stage accounts for essentially all of that observed difference, with small offsetting variation elsewhere. The remaining specialized active budget is 38.31% vocabulary projection, 25.75% attention, and 13.46% joint kernels. With skipping disabled, specialization is 3.744% slower.

At kappa 0.1, synchronized host latency falls 534.562 to 486.441 us (9.002%). The six joint gate/down-projection/residual kernels fall 89.324 to 40.568 us in the CUPTI sample, a 48.757-us saving. Total kernel-active time falls 48.347 us: the changed stage accounts for essentially all of that observed difference, with small offsetting variation elsewhere. The remaining specialized active budget is 40.20% vocabulary projection, 27.12% attention, and 9.08% joint kernels. With skipping disabled, specialization is 3.796% slower.

The event-rich mapped graphs add 18.0% to 24.9% to mean CUDA latency. Their detailed partitions are useful for the execution map, but the less intrusive CUPTI kernel durations are the primary stage-localization evidence. The report keeps baseline wall time, profiled active time/gaps, instrumented intervals and SM cycles as separate budgets.

The specialized source reads one corner per 8x16 tile, removes the generic full inspection/shared-memory/barrier path, and launches 128 rather than 256 threads while retaining four projection warps. Timestamp probes show much shorter support/preparation and accumulation intervals, consistent with removing support work and changing warp scheduling. The preparation boundaries differ and clocks include stalls; these measurements do not identify a causal percentage for mask inspection, occupancy, memory traffic or arithmetic. Identical MMA counters rule out additional product skipping as the generic-to-specialized explanation.

Method: one physical RTX 5090; BF16 B=1, T=2048, uncached full 50,304-way logits. Same retained Run060 T2/Ph checkpoints and corner 8x16 gate at kappa 0/.1. Three fresh processes per threshold, 64 fixed validation inputs x 7 randomized paired passes. Full 500-document/338-block qualification covers every baseline and mapped graph; 1,444-token excluded tail. No weights, thresholds, arithmetic acceptance bounds or scientific kernel settings changed.

The full step means a forward for all 2,048 tokens, not one decoding token. Input copies are measured separately; dataset reads, model loading, weight preparation, compilation, allocation and capture are setup. Cross-entropy/loss and diagnostics are qualification work outside the inference timer.

## Detailed execution map

| Order | Mathematical work / shapes | Actual optimized execution |
|---|---|---|
| 1 | Position indices [1,2048] | arange kernel; view is metadata |
| 2 | Token lookup [1,2048,128] | embedding lookup; eval dropout identity |
| 3 | Shared partial RoPE cos/sin tables | recurring table generation for this forward |
| 4a, each of 6 layers | Two LayerNorms of the same residual, distinct affine weights | one K050 Welford normalization kernel; MLP retrieves the second output later |
| 4b | QKV: [2048,128] x [128,384], including bias | K042 CUTLASS projection; no a gate |
| 4c | Q/K partial RoPE and V packing; [1,4,2048,32] | K019 fused RoPE/layout; identity Q/K/V taps |
| 4d | Causal QK, scaled softmax, PV | frozen K035 split-KV attention and split recombination kernels; QK/softmax/PV are fused, with no independently measured wall times |
| 4e | Heads -> [1,2048,128] context | transpose view plus contiguous packing copy; original z gate is deferred to joint |
| 4f | MLP up: [2048,128] x [128,512], bias | K042 CUTLASS projection; no m gate and h gate deferred |
| 4g | h/z tile gate, W2 [512,128], Wo [128,128], bias, rounded branch sum and residual add | one joint CUDA kernel; generic or corner-specialized; SAME mathematical outputs |
| 5 | Final LayerNorm [1,2048,128] | native affine normalization |
| 6 | Vocabulary head [2048,128] x [128,50304] | dense cuBLAS projection, [1,2048,50304] logits |

GPT-NeoX has parallel mathematical branches; this single-stream implementation executes the attention work then MLP work before joining them. LayerNorm pairs and joint projections are fusions, not concurrent whole-model branches. Metadata views, identity taps and eval dropout have no independent CUDA kernel. Their host work appears in annotated eager CPU traces.

## Full forward, overhead and measurement coverage

| kappa | Mode | Host geom mean (us) | Host arithmetic mean (us) | CUDA arithmetic mean (us) | Host minus CUDA mean (us) | Timed-event graph / baseline CUDA |
|---|---|---:|---:|---:|---:|---:|
| 0 | Native | 837.750 | 837.753 | 822.720 | 15.033 | 1.2310x |
| 0 | Generic sparse | 568.726 | 568.733 | 553.704 | 15.030 | 1.2109x |
| 0 | Specialized sparse | 509.939 | 509.943 | 494.903 | 15.040 | 1.2393x |
| 0 | Generic no skip | 635.799 | 635.803 | 620.774 | 15.029 | 1.1864x |
| 0 | Specialized no skip | 659.606 | 659.610 | 644.602 | 15.008 | 1.1806x |
| 0.1 | Native | 834.927 | 834.932 | 820.379 | 14.553 | 1.2278x |
| 0.1 | Generic sparse | 534.562 | 534.573 | 520.002 | 14.572 | 1.2221x |
| 0.1 | Specialized sparse | 486.441 | 486.447 | 471.869 | 14.578 | 1.2493x |
| 0.1 | Generic no skip | 633.721 | 633.728 | 619.142 | 14.586 | 1.1878x |
| 0.1 | Specialized no skip | 657.777 | 657.783 | 643.162 | 14.621 | 1.1800x |

Host minus CUDA is a timer-boundary residual including launch/synchronization/timing overhead, not a precisely isolated CPU compute stage. Event instrumentation includes extra graph event nodes and may alter scheduling/cache state. Event stage budgets close to their OWN instrumented totals and are never rescaled to uninstrumented latency. Arithmetic means, not geometric means or summed medians, define additive budgets.

| kappa | Process replicate | Generic sparse host geom (us) | Specialized sparse host geom (us) | Reduction (%) |
|---|---:|---:|---:|---:|
| 0 | 1 | 569.473 | 510.464 | 10.362 |
| 0 | 2 | 568.579 | 509.766 | 10.344 |
| 0 | 3 | 568.126 | 509.587 | 10.304 |
| 0.1 | 1 | 532.976 | 485.255 | 8.954 |
| 0.1 | 2 | 534.268 | 486.182 | 9.000 |
| 0.1 | 3 | 536.447 | 487.890 | 9.052 |

Equal staging is a device-to-device copy into the graph input buffer, synchronization and outer event construction. It is timed separately from resident-input inference; disk reads and initial CPU-to-GPU dataset staging are setup. Annotated eager CPU stages are retained in raw traces, but steady graph replay dispatches one graph rather than executing those Python stages.

| kappa | Mode | Staging host arithmetic mean (us), excluded |
|---|---|---:|
| 0 | Native | 28.388 |
| 0 | Generic sparse | 29.076 |
| 0 | Specialized sparse | 29.057 |
| 0 | Generic no skip | 28.835 |
| 0 | Specialized no skip | 28.814 |
| 0.1 | Native | 28.229 |
| 0.1 | Generic sparse | 28.841 |
| 0.1 | Specialized sparse | 28.874 |
| 0.1 | Generic no skip | 28.768 |
| 0.1 | Specialized no skip | 28.450 |

## kappa 0: per-stage distribution

Baseline graph CUPTI activity time, without internal timing-event nodes; arithmetic mean over 4 inputs x 3 replays. CUPTI profiling remains active. Sum covers measured GPU activities; inter-kernel gaps and host overhead are separate. Computational kernels require exact ordered kernel-name agreement with the annotated equivalent eager map. Native replay also has 12 graph copy activities, one in each h/z gate: eager records these as device-to-device memcpy, while graph replay records memcpy32_post/memcpy128 kernels. The local mapper accepts them only when enclosed by kernels of the same gate, with exactly one copy per gate/layer and identical signatures in every replay. The original strict-mapping failures remain in the raw evidence. This is a 12-forward profiling sample, separate from the 1,344 baseline samples per mode.

| Stage, pooled over 6 layers where applicable | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |
|---|---:|---:|---:|---:|---:|
| Position indices | 0.888 | 0.872 | 0.837 | 0.885 | 0.869 |
| Embedding | 2.197 | 2.173 | 2.173 | 2.237 | 2.216 |
| Rotary tables | 14.544 | 12.837 | 12.867 | 12.824 | 12.931 |
| Paired normalization | 41.986 | 10.491 | 10.485 | 10.453 | 10.453 |
| QKV projection | 24.154 | 24.475 | 24.349 | 24.677 | 24.341 |
| RoPE / QKV layout | 125.930 | 13.991 | 13.917 | 13.989 | 14.053 |
| Causal attention | 122.602 | 121.538 | 121.322 | 121.906 | 121.967 |
| Context packing | 0.000 | 10.280 | 10.232 | 10.259 | 10.253 |
| MLP up projection | 24.691 | 26.962 | 27.536 | 27.024 | 26.941 |
| Gates / down projections / residual | 192.002 | 121.025 | 63.397 | 185.604 | 210.299 |
| Final normalization | 3.451 | 3.475 | 3.475 | 3.453 | 3.461 |
| Vocabulary projection | 187.447 | 181.057 | 180.482 | 181.986 | 183.385 |
| Total kernel-active time | 739.893 | 529.177 | 471.072 | 595.298 | 621.171 |

Specialized sparse distribution, using its own profiled GPU-active denominator:

| Stage | Mean (us) | Active share (%) | Generic minus specialized (us) |
|---|---:|---:|---:|
| Position indices | 0.837 | 0.18 | 0.035 |
| Embedding | 2.173 | 0.46 | 0.000 |
| Rotary tables | 12.867 | 2.73 | -0.029 |
| Paired normalization | 10.485 | 2.23 | 0.005 |
| QKV projection | 24.349 | 5.17 | 0.125 |
| RoPE / QKV layout | 13.917 | 2.95 | 0.074 |
| Causal attention | 121.322 | 25.75 | 0.216 |
| Context packing | 10.232 | 2.17 | 0.048 |
| MLP up projection | 27.536 | 5.85 | -0.573 |
| Gates / down projections / residual | 63.397 | 13.46 | 57.628 |
| Final normalization | 3.475 | 0.74 | 0.000 |
| Vocabulary projection | 180.482 | 38.31 | 0.575 |

| Mode | Mapped inputs | Activities/forward | Active union (us) | First-to-last activity span (us) | Gaps in span (us) | Profile span / matched unprofiled CUDA bracket |
|---|---:|---:|---:|---:|---:|---:|
| Native | 4/4 | 248 | 739.893 | 844.921 | 105.028 | 1.0263x |
| Generic sparse | 4/4 | 62 | 529.177 | 555.859 | 26.682 | 1.0033x |
| Specialized sparse | 4/4 | 62 | 471.072 | 497.982 | 26.909 | 1.0048x |
| Generic no skip | 4/4 | 62 | 595.298 | 621.874 | 26.576 | 1.0011x |
| Specialized no skip | 4/4 | 62 | 621.171 | 648.018 | 26.847 | 1.0057x |

The final ratio compares the profiled first-to-last activity span with unprofiled outer-CUDA-event timings of the same 4 inputs in the same process. Their timer boundaries differ, so the ratio is a consistency check, not an exact estimate of CUPTI overhead. No latency budget mixes these samples.

Inside attention, the actually separate CUDA kernels can be timed. The fused split-attention kernel includes QK, online softmax and PV; the second kernel combines splits. Native attention also includes a layout copy in this stage.

| Attention component, pooled over 6 layers | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |
|---|---:|---:|---:|---:|---:|
| Fused QK / softmax / PV | 97.010 | 105.607 | 105.376 | 105.917 | 106.007 |
| Split recombination | 15.309 | 15.930 | 15.947 | 15.989 | 15.960 |
| Layout copy | 10.283 | 0.000 | 0.000 | 0.000 | 0.000 |

Rotary-table generation also comprises separate small CUDA operations. The specialized sparse graph calls, in execution order:

| Rotary table substep | Mean (us) |
|---|---:|
| Inverse-frequency dtype copy | 1.547 |
| Position dtype copy | 2.005 |
| Frequency-position product | 1.147 |
| Frequency concatenation | 1.504 |
| Cosine | 1.597 |
| Cosine scaling | 0.941 |
| Sine | 1.552 |
| Sine scaling | 0.877 |
| Cosine BF16 cast | 0.888 |
| Sine BF16 cast | 0.808 |

[Complete kernel-node inventory](../results/kernel-nodes.csv) lists every ordered CUDA activity, full kernel name, mapped stage and duration distribution for all five modes and both thresholds. Raw timelines retain their timestamps.

Every optimized execution stage in the CUPTI trace, separately for all six layers:

| Stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |
|---|---:|---:|---:|
| positions | 0.872 | 0.837 | 0.035 |
| embedding | 2.173 | 2.173 | 0.000 |
| rotary_tables | 12.837 | 12.867 | -0.029 |
| layer0.norm_pair | 1.861 | 1.832 | 0.029 |
| layer0.qkv | 4.384 | 4.312 | 0.072 |
| layer0.rope_qkv_layout | 2.477 | 2.453 | 0.024 |
| layer0.attention | 22.691 | 22.581 | 0.109 |
| layer0.context_pack | 1.731 | 1.741 | -0.011 |
| layer0.mlp_up | 4.491 | 4.504 | -0.013 |
| layer0.joint_gate_down_residual | 22.456 | 11.904 | 10.552 |
| layer1.norm_pair | 1.707 | 1.755 | -0.048 |
| layer1.qkv | 3.976 | 4.059 | -0.083 |
| layer1.rope_qkv_layout | 2.312 | 2.285 | 0.027 |
| layer1.attention | 19.784 | 19.709 | 0.074 |
| layer1.context_pack | 1.701 | 1.707 | -0.005 |
| layer1.mlp_up | 4.573 | 4.557 | 0.016 |
| layer1.joint_gate_down_residual | 19.890 | 9.925 | 9.965 |
| layer2.norm_pair | 1.757 | 1.733 | 0.024 |
| layer2.qkv | 4.024 | 3.909 | 0.115 |
| layer2.rope_qkv_layout | 2.290 | 2.296 | -0.006 |
| layer2.attention | 19.747 | 19.792 | -0.045 |
| layer2.context_pack | 1.717 | 1.699 | 0.019 |
| layer2.mlp_up | 4.509 | 4.483 | 0.026 |
| layer2.joint_gate_down_residual | 20.880 | 10.957 | 9.923 |
| layer3.norm_pair | 1.741 | 1.717 | 0.024 |
| layer3.qkv | 4.032 | 3.981 | 0.051 |
| layer3.rope_qkv_layout | 2.312 | 2.301 | 0.011 |
| layer3.attention | 19.768 | 19.715 | 0.053 |
| layer3.context_pack | 1.712 | 1.699 | 0.013 |
| layer3.mlp_up | 4.523 | 4.656 | -0.133 |
| layer3.joint_gate_down_residual | 19.432 | 10.525 | 8.906 |
| layer4.norm_pair | 1.709 | 1.720 | -0.011 |
| layer4.qkv | 4.016 | 4.029 | -0.013 |
| layer4.rope_qkv_layout | 2.299 | 2.280 | 0.019 |
| layer4.attention | 19.768 | 19.738 | 0.029 |
| layer4.context_pack | 1.725 | 1.688 | 0.037 |
| layer4.mlp_up | 4.405 | 4.661 | -0.256 |
| layer4.joint_gate_down_residual | 18.509 | 10.539 | 7.971 |
| layer5.norm_pair | 1.715 | 1.728 | -0.013 |
| layer5.qkv | 4.043 | 4.059 | -0.016 |
| layer5.rope_qkv_layout | 2.301 | 2.301 | -0.000 |
| layer5.attention | 19.781 | 19.787 | -0.005 |
| layer5.context_pack | 1.693 | 1.699 | -0.005 |
| layer5.mlp_up | 4.461 | 4.675 | -0.213 |
| layer5.joint_gate_down_residual | 19.858 | 9.547 | 10.312 |
| final_norm | 3.475 | 3.475 | 0.000 |
| lm_head | 181.057 | 180.482 | 0.575 |

Native graph operations before fusion, pooled over six layers:

| Native operation | Mean GPU activity (us) |
|---|---:|
| positions | 0.888 |
| embedding | 2.197 |
| rotary_tables | 14.544 |
| norm_attention | 21.179 |
| qkv | 24.154 |
| rope_qkv_layout | 125.930 |
| attention | 122.602 |
| context_pack | 0.000 |
| z_gate | 69.333 |
| attention_out | 14.955 |
| norm_mlp | 20.808 |
| mlp_up | 24.691 |
| h_gate | 67.042 |
| mlp_down | 28.696 |
| residual | 11.976 |
| final_norm | 3.451 |
| lm_head | 187.447 |

Full-validation projection instruction counters are identical between consumers. These count declared MMA instructions, not hardware-measured FLOPs or memory traffic.

| Counter | Pooled integer count, both consumers |
|---|---:|
| h_mma_issued | 112,930,320 |
| h_mma_bypassed | 152,883,696 |
| z_mma_issued | 36,599,216 |
| z_mma_bypassed | 29,854,288 |
| h_simt_products | 0 |
| z_simt_products | 0 |

Instrumented graph event budget: all 64 inputs x 7 passes x 3 processes. These intervals include event-node/scheduling costs and close to the instrumented forward.

| Execution stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |
|---|---:|---:|---:|
| Position indices | 3.173 | 3.247 | -0.074 |
| Embedding | 5.398 | 5.363 | 0.035 |
| Rotary tables | 18.477 | 18.445 | 0.032 |
| Paired normalization | 29.139 | 28.768 | 0.371 |
| QKV projection | 40.573 | 39.665 | 0.907 |
| RoPE / QKV layout | 28.237 | 27.962 | 0.274 |
| Causal attention | 140.416 | 140.350 | 0.066 |
| Context packing | 28.489 | 28.471 | 0.017 |
| MLP up projection | 42.677 | 46.125 | -3.448 |
| Gates / down projections / residual | 138.027 | 79.520 | 58.507 |
| Final normalization | 6.144 | 6.144 | 0.000 |
| Vocabulary projection | 182.510 | 182.058 | 0.452 |

Complete per-layer event measurements:

| Stage | Generic sparse mean (us) | Specialized sparse mean (us) |
|---|---:|---:|
| positions | 3.173 | 3.247 |
| embedding | 5.398 | 5.363 |
| rotary_tables | 18.477 | 18.445 |
| layer0.norm_pair | 6.707 | 6.768 |
| layer0.qkv | 6.152 | 6.170 |
| layer0.rope_qkv_layout | 5.411 | 5.458 |
| layer0.attention | 26.825 | 26.829 |
| layer0.context_pack | 6.515 | 6.317 |
| layer0.mlp_up | 9.670 | 10.032 |
| layer0.joint_gate_down_residual | 27.040 | 16.220 |
| layer1.norm_pair | 6.040 | 5.467 |
| layer1.qkv | 9.861 | 8.927 |
| layer1.rope_qkv_layout | 6.443 | 6.138 |
| layer1.attention | 23.479 | 23.408 |
| layer1.context_pack | 5.590 | 5.771 |
| layer1.mlp_up | 8.393 | 9.781 |
| layer1.joint_gate_down_residual | 22.936 | 12.640 |
| layer2.norm_pair | 4.104 | 4.239 |
| layer2.qkv | 6.128 | 6.137 |
| layer2.rope_qkv_layout | 4.095 | 4.079 |
| layer2.attention | 22.528 | 22.528 |
| layer2.context_pack | 4.096 | 4.095 |
| layer2.mlp_up | 6.146 | 6.148 |
| layer2.joint_gate_down_residual | 23.195 | 13.298 |
| layer3.norm_pair | 4.096 | 4.098 |
| layer3.qkv | 6.144 | 6.144 |
| layer3.rope_qkv_layout | 4.096 | 4.096 |
| layer3.attention | 22.528 | 22.528 |
| layer3.context_pack | 4.096 | 4.096 |
| layer3.mlp_up | 6.156 | 6.273 |
| layer3.joint_gate_down_residual | 21.451 | 12.556 |
| layer4.norm_pair | 4.096 | 4.099 |
| layer4.qkv | 6.144 | 6.144 |
| layer4.rope_qkv_layout | 4.096 | 4.096 |
| layer4.attention | 22.528 | 22.528 |
| layer4.context_pack | 4.096 | 4.096 |
| layer4.mlp_up | 6.144 | 7.168 |
| layer4.joint_gate_down_residual | 21.257 | 12.521 |
| layer5.norm_pair | 4.096 | 4.097 |
| layer5.qkv | 6.144 | 6.144 |
| layer5.rope_qkv_layout | 4.096 | 4.096 |
| layer5.attention | 22.528 | 22.528 |
| layer5.context_pack | 4.096 | 4.096 |
| layer5.mlp_up | 6.168 | 6.723 |
| layer5.joint_gate_down_residual | 22.147 | 12.285 |
| final_norm | 6.144 | 6.144 |
| lm_head | 182.510 | 182.058 |

Internal timestamp diagnostic (mean SM cycles per projection warp; every 8-token block, 8 inputs x 5 repeats). Support/preparation, h, z and output phases are serial within a warp but overlap across warps/CTAs. They cannot be added to full-device latency. The generic preparation phase includes inspection, a block barrier and short/zero-row bias/output handling; the specialized phase reads corners. For a wholly dead generic block, its bias/residual output happens in preparation, and later phases are zero. These phase labels therefore do not isolate pure mask cost or provide clean causal percentages. Probe slowdown is independently measured and retained.

| Mode / layer | Support cycles | h accumulation cycles | z accumulation cycles | Output cycles | Probe / production event time |
|---|---:|---:|---:|---:|---:|
| Generic sparse / 0 | 13861.9 | 26539.2 | 11997.7 | 2698.3 | 1.059x |
| Generic sparse / 1 | 13664.9 | 21997.0 | 9092.0 | 2652.7 | 1.075x |
| Generic sparse / 2 | 13757.9 | 22283.2 | 10873.1 | 2752.4 | 1.069x |
| Generic sparse / 3 | 13481.5 | 20291.9 | 8074.2 | 2704.8 | 1.060x |
| Generic sparse / 4 | 13634.3 | 20008.6 | 6410.4 | 2662.2 | 1.071x |
| Generic sparse / 5 | 13408.5 | 21747.7 | 7569.2 | 2672.3 | 1.066x |
| Specialized sparse / 0 | 695.9 | 15771.8 | 5396.9 | 2057.3 | 0.969x |
| Specialized sparse / 1 | 630.1 | 13145.5 | 3914.9 | 2466.9 | 1.021x |
| Specialized sparse / 2 | 629.8 | 16419.9 | 4767.9 | 2180.0 | 1.063x |
| Specialized sparse / 3 | 629.9 | 15130.8 | 3803.7 | 2052.7 | 1.036x |
| Specialized sparse / 4 | 630.2 | 14794.8 | 2983.6 | 2031.8 | 1.004x |
| Specialized sparse / 5 | 630.6 | 12987.9 | 3058.5 | 2449.2 | 1.010x |
| Generic no skip / 0 | 77.3 | 46254.0 | 11755.9 | 2201.4 | 1.002x |
| Generic no skip / 1 | 77.8 | 46081.0 | 11474.2 | 2231.1 | 1.021x |
| Generic no skip / 2 | 77.8 | 46100.0 | 11643.5 | 2198.7 | 1.013x |
| Generic no skip / 3 | 77.8 | 45602.4 | 11421.3 | 2150.7 | 1.005x |
| Generic no skip / 4 | 77.8 | 45557.9 | 11326.5 | 2222.9 | 1.013x |
| Generic no skip / 5 | 77.8 | 45748.7 | 11362.7 | 2167.2 | 1.008x |
| Specialized no skip / 0 | 711.2 | 29919.1 | 7934.2 | 2200.6 | 1.055x |
| Specialized no skip / 1 | 629.1 | 30467.4 | 8501.8 | 2063.9 | 1.031x |
| Specialized no skip / 2 | 628.9 | 24489.1 | 8532.3 | 2031.6 | 1.025x |
| Specialized no skip / 3 | 628.7 | 24227.7 | 6708.5 | 2389.7 | 0.984x |
| Specialized no skip / 4 | 628.8 | 24216.9 | 6823.7 | 2388.9 | 1.018x |
| Specialized no skip / 5 | 628.9 | 24260.2 | 6792.2 | 2264.4 | 1.020x |

## kappa 0.1: per-stage distribution

Baseline graph CUPTI activity time, without internal timing-event nodes; arithmetic mean over 4 inputs x 3 replays. CUPTI profiling remains active. Sum covers measured GPU activities; inter-kernel gaps and host overhead are separate. Computational kernels require exact ordered kernel-name agreement with the annotated equivalent eager map. Native replay also has 12 graph copy activities, one in each h/z gate: eager records these as device-to-device memcpy, while graph replay records memcpy32_post/memcpy128 kernels. The local mapper accepts them only when enclosed by kernels of the same gate, with exactly one copy per gate/layer and identical signatures in every replay. The original strict-mapping failures remain in the raw evidence. This is a 12-forward profiling sample, separate from the 1,344 baseline samples per mode.

| Stage, pooled over 6 layers where applicable | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |
|---|---:|---:|---:|---:|---:|
| Position indices | 0.885 | 0.856 | 0.840 | 0.891 | 0.888 |
| Embedding | 2.192 | 2.157 | 2.211 | 2.229 | 2.237 |
| Rotary tables | 14.446 | 12.707 | 12.765 | 12.811 | 12.872 |
| Paired normalization | 41.931 | 10.507 | 10.442 | 10.437 | 10.413 |
| QKV projection | 24.208 | 24.360 | 24.288 | 24.538 | 24.312 |
| RoPE / QKV layout | 125.652 | 13.963 | 13.856 | 13.973 | 14.008 |
| Causal attention | 122.458 | 120.907 | 121.250 | 121.485 | 121.532 |
| Context packing | 0.000 | 10.240 | 10.219 | 10.251 | 10.227 |
| MLP up projection | 24.627 | 26.779 | 27.389 | 27.045 | 26.946 |
| Gates / down projections / residual | 191.473 | 89.324 | 40.568 | 185.412 | 208.276 |
| Final normalization | 3.451 | 3.459 | 3.483 | 3.435 | 3.480 |
| Vocabulary projection | 186.687 | 180.100 | 179.700 | 181.524 | 183.180 |
| Total kernel-active time | 738.010 | 495.358 | 447.011 | 594.031 | 618.372 |

Specialized sparse distribution, using its own profiled GPU-active denominator:

| Stage | Mean (us) | Active share (%) | Generic minus specialized (us) |
|---|---:|---:|---:|
| Position indices | 0.840 | 0.19 | 0.016 |
| Embedding | 2.211 | 0.49 | -0.053 |
| Rotary tables | 12.765 | 2.86 | -0.059 |
| Paired normalization | 10.442 | 2.34 | 0.064 |
| QKV projection | 24.288 | 5.43 | 0.072 |
| RoPE / QKV layout | 13.856 | 3.10 | 0.107 |
| Causal attention | 121.250 | 27.12 | -0.344 |
| Context packing | 10.219 | 2.29 | 0.021 |
| MLP up projection | 27.389 | 6.13 | -0.610 |
| Gates / down projections / residual | 40.568 | 9.08 | 48.757 |
| Final normalization | 3.483 | 0.78 | -0.024 |
| Vocabulary projection | 179.700 | 40.20 | 0.400 |

| Mode | Mapped inputs | Activities/forward | Active union (us) | First-to-last activity span (us) | Gaps in span (us) | Profile span / matched unprofiled CUDA bracket |
|---|---:|---:|---:|---:|---:|---:|
| Native | 4/4 | 248 | 738.010 | 843.000 | 104.991 | 1.0300x |
| Generic sparse | 4/4 | 62 | 495.358 | 522.238 | 26.880 | 1.0055x |
| Specialized sparse | 4/4 | 62 | 447.011 | 474.130 | 27.119 | 1.0071x |
| Generic no skip | 4/4 | 62 | 594.031 | 620.737 | 26.706 | 1.0064x |
| Specialized no skip | 4/4 | 62 | 618.372 | 645.323 | 26.952 | 1.0066x |

The final ratio compares the profiled first-to-last activity span with unprofiled outer-CUDA-event timings of the same 4 inputs in the same process. Their timer boundaries differ, so the ratio is a consistency check, not an exact estimate of CUPTI overhead. No latency budget mixes these samples.

Inside attention, the actually separate CUDA kernels can be timed. The fused split-attention kernel includes QK, online softmax and PV; the second kernel combines splits. Native attention also includes a layout copy in this stage.

| Attention component, pooled over 6 layers | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |
|---|---:|---:|---:|---:|---:|
| Fused QK / softmax / PV | 96.861 | 105.032 | 105.355 | 105.541 | 105.623 |
| Split recombination | 15.322 | 15.874 | 15.896 | 15.944 | 15.909 |
| Layout copy | 10.275 | 0.000 | 0.000 | 0.000 | 0.000 |

Rotary-table generation also comprises separate small CUDA operations. The specialized sparse graph calls, in execution order:

| Rotary table substep | Mean (us) |
|---|---:|
| Inverse-frequency dtype copy | 1.525 |
| Position dtype copy | 2.000 |
| Frequency-position product | 1.139 |
| Frequency concatenation | 1.509 |
| Cosine | 1.579 |
| Cosine scaling | 0.933 |
| Sine | 1.533 |
| Sine scaling | 0.867 |
| Cosine BF16 cast | 0.875 |
| Sine BF16 cast | 0.805 |

[Complete kernel-node inventory](../results/kernel-nodes.csv) lists every ordered CUDA activity, full kernel name, mapped stage and duration distribution for all five modes and both thresholds. Raw timelines retain their timestamps.

Every optimized execution stage in the CUPTI trace, separately for all six layers:

| Stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |
|---|---:|---:|---:|
| positions | 0.856 | 0.840 | 0.016 |
| embedding | 2.157 | 2.211 | -0.053 |
| rotary_tables | 12.707 | 12.765 | -0.059 |
| layer0.norm_pair | 1.875 | 1.803 | 0.072 |
| layer0.qkv | 4.296 | 4.296 | 0.000 |
| layer0.rope_qkv_layout | 2.464 | 2.437 | 0.027 |
| layer0.attention | 22.496 | 22.675 | -0.179 |
| layer0.context_pack | 1.723 | 1.733 | -0.011 |
| layer0.mlp_up | 4.427 | 4.395 | 0.032 |
| layer0.joint_gate_down_residual | 17.925 | 8.221 | 9.704 |
| layer1.norm_pair | 1.707 | 1.744 | -0.037 |
| layer1.qkv | 3.965 | 4.013 | -0.048 |
| layer1.rope_qkv_layout | 2.312 | 2.267 | 0.045 |
| layer1.attention | 19.699 | 19.715 | -0.016 |
| layer1.context_pack | 1.696 | 1.707 | -0.011 |
| layer1.mlp_up | 4.531 | 4.560 | -0.029 |
| layer1.joint_gate_down_residual | 14.176 | 6.621 | 7.555 |
| layer2.norm_pair | 1.744 | 1.723 | 0.021 |
| layer2.qkv | 4.008 | 3.973 | 0.035 |
| layer2.rope_qkv_layout | 2.291 | 2.296 | -0.005 |
| layer2.attention | 19.645 | 19.728 | -0.083 |
| layer2.context_pack | 1.715 | 1.693 | 0.021 |
| layer2.mlp_up | 4.496 | 4.427 | 0.069 |
| layer2.joint_gate_down_residual | 15.328 | 6.989 | 8.339 |
| layer3.norm_pair | 1.752 | 1.725 | 0.027 |
| layer3.qkv | 4.019 | 3.965 | 0.053 |
| layer3.rope_qkv_layout | 2.309 | 2.285 | 0.024 |
| layer3.attention | 19.699 | 19.696 | 0.003 |
| layer3.context_pack | 1.701 | 1.693 | 0.008 |
| layer3.mlp_up | 4.485 | 4.637 | -0.152 |
| layer3.joint_gate_down_residual | 14.211 | 6.723 | 7.488 |
| layer4.norm_pair | 1.717 | 1.723 | -0.005 |
| layer4.qkv | 4.008 | 4.035 | -0.027 |
| layer4.rope_qkv_layout | 2.291 | 2.280 | 0.011 |
| layer4.attention | 19.669 | 19.704 | -0.034 |
| layer4.context_pack | 1.720 | 1.688 | 0.032 |
| layer4.mlp_up | 4.392 | 4.664 | -0.272 |
| layer4.joint_gate_down_residual | 13.963 | 6.107 | 7.856 |
| layer5.norm_pair | 1.712 | 1.725 | -0.013 |
| layer5.qkv | 4.064 | 4.005 | 0.059 |
| layer5.rope_qkv_layout | 2.296 | 2.291 | 0.005 |
| layer5.attention | 19.699 | 19.733 | -0.035 |
| layer5.context_pack | 1.685 | 1.704 | -0.019 |
| layer5.mlp_up | 4.448 | 4.707 | -0.259 |
| layer5.joint_gate_down_residual | 13.723 | 5.907 | 7.816 |
| final_norm | 3.459 | 3.483 | -0.024 |
| lm_head | 180.100 | 179.700 | 0.400 |

Native graph operations before fusion, pooled over six layers:

| Native operation | Mean GPU activity (us) |
|---|---:|
| positions | 0.885 |
| embedding | 2.192 |
| rotary_tables | 14.446 |
| norm_attention | 21.139 |
| qkv | 24.208 |
| rope_qkv_layout | 125.652 |
| attention | 122.458 |
| context_pack | 0.000 |
| z_gate | 69.234 |
| attention_out | 14.861 |
| norm_mlp | 20.792 |
| mlp_up | 24.627 |
| h_gate | 66.906 |
| mlp_down | 28.546 |
| residual | 11.925 |
| final_norm | 3.451 |
| lm_head | 186.687 |

Full-validation projection instruction counters are identical between consumers. These count declared MMA instructions, not hardware-measured FLOPs or memory traffic.

| Counter | Pooled integer count, both consumers |
|---|---:|
| h_mma_issued | 41,568,800 |
| h_mma_bypassed | 224,245,216 |
| z_mma_issued | 19,289,056 |
| z_mma_bypassed | 47,164,448 |
| h_simt_products | 0 |
| z_simt_products | 0 |

Instrumented graph event budget: all 64 inputs x 7 passes x 3 processes. These intervals include event-node/scheduling costs and close to the instrumented forward.

| Execution stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |
|---|---:|---:|---:|
| Position indices | 3.279 | 3.377 | -0.098 |
| Embedding | 5.293 | 5.173 | 0.120 |
| Rotary tables | 18.441 | 18.451 | -0.010 |
| Paired normalization | 28.740 | 28.536 | 0.204 |
| QKV projection | 40.047 | 39.602 | 0.445 |
| RoPE / QKV layout | 27.938 | 27.728 | 0.210 |
| Causal attention | 140.188 | 140.285 | -0.098 |
| Context packing | 28.314 | 28.334 | -0.020 |
| MLP up projection | 43.378 | 45.770 | -2.392 |
| Gates / down projections / residual | 104.492 | 57.352 | 47.140 |
| Final normalization | 6.144 | 6.144 | 0.000 |
| Vocabulary projection | 181.951 | 181.457 | 0.494 |

Complete per-layer event measurements:

| Stage | Generic sparse mean (us) | Specialized sparse mean (us) |
|---|---:|---:|
| positions | 3.279 | 3.377 |
| embedding | 5.293 | 5.173 |
| rotary_tables | 18.441 | 18.451 |
| layer0.norm_pair | 6.571 | 6.660 |
| layer0.qkv | 6.237 | 6.206 |
| layer0.rope_qkv_layout | 5.324 | 5.406 |
| layer0.attention | 26.713 | 26.759 |
| layer0.context_pack | 6.392 | 6.144 |
| layer0.mlp_up | 9.614 | 10.014 |
| layer0.joint_gate_down_residual | 21.868 | 12.465 |
| layer1.norm_pair | 5.678 | 5.165 |
| layer1.qkv | 9.292 | 8.593 |
| layer1.rope_qkv_layout | 6.240 | 5.913 |
| layer1.attention | 23.362 | 23.422 |
| layer1.context_pack | 5.537 | 5.810 |
| layer1.mlp_up | 9.156 | 9.691 |
| layer1.joint_gate_down_residual | 17.389 | 9.913 |
| layer2.norm_pair | 4.203 | 4.418 |
| layer2.qkv | 6.087 | 6.372 |
| layer2.rope_qkv_layout | 4.086 | 4.121 |
| layer2.attention | 22.528 | 22.520 |
| layer2.context_pack | 4.096 | 4.092 |
| layer2.mlp_up | 6.147 | 6.147 |
| layer2.joint_gate_down_residual | 17.466 | 9.527 |
| layer3.norm_pair | 4.096 | 4.098 |
| layer3.qkv | 6.144 | 6.143 |
| layer3.rope_qkv_layout | 4.096 | 4.096 |
| layer3.attention | 22.528 | 22.528 |
| layer3.context_pack | 4.096 | 4.096 |
| layer3.mlp_up | 6.158 | 6.225 |
| layer3.joint_gate_down_residual | 16.116 | 8.776 |
| layer4.norm_pair | 4.096 | 4.098 |
| layer4.qkv | 6.144 | 6.144 |
| layer4.rope_qkv_layout | 4.096 | 4.096 |
| layer4.attention | 22.528 | 22.528 |
| layer4.context_pack | 4.096 | 4.096 |
| layer4.mlp_up | 6.144 | 7.064 |
| layer4.joint_gate_down_residual | 15.848 | 8.354 |
| layer5.norm_pair | 4.096 | 4.097 |
| layer5.qkv | 6.144 | 6.144 |
| layer5.rope_qkv_layout | 4.096 | 4.095 |
| layer5.attention | 22.528 | 22.529 |
| layer5.context_pack | 4.096 | 4.096 |
| layer5.mlp_up | 6.160 | 6.629 |
| layer5.joint_gate_down_residual | 15.806 | 8.317 |
| final_norm | 6.144 | 6.144 |
| lm_head | 181.951 | 181.457 |

Internal timestamp diagnostic (mean SM cycles per projection warp; every 8-token block, 8 inputs x 5 repeats). Support/preparation, h, z and output phases are serial within a warp but overlap across warps/CTAs. They cannot be added to full-device latency. The generic preparation phase includes inspection, a block barrier and short/zero-row bias/output handling; the specialized phase reads corners. For a wholly dead generic block, its bias/residual output happens in preparation, and later phases are zero. These phase labels therefore do not isolate pure mask cost or provide clean causal percentages. Probe slowdown is independently measured and retained.

| Mode / layer | Support cycles | h accumulation cycles | z accumulation cycles | Output cycles | Probe / production event time |
|---|---:|---:|---:|---:|---:|
| Generic sparse / 0 | 13641.4 | 16629.8 | 4184.7 | 2627.6 | 1.054x |
| Generic sparse / 1 | 13307.6 | 7154.9 | 4833.2 | 2626.5 | 1.102x |
| Generic sparse / 2 | 13505.5 | 5417.1 | 8636.1 | 2654.4 | 1.060x |
| Generic sparse / 3 | 13529.6 | 5102.2 | 4936.6 | 2644.1 | 1.000x |
| Generic sparse / 4 | 13346.8 | 6238.3 | 2628.9 | 2645.9 | 1.034x |
| Generic sparse / 5 | 13392.7 | 5999.8 | 2391.4 | 2592.4 | 1.029x |
| Specialized sparse / 0 | 677.6 | 9884.0 | 1566.0 | 2106.8 | 1.022x |
| Specialized sparse / 1 | 630.1 | 4720.1 | 2186.3 | 2585.2 | 1.018x |
| Specialized sparse / 2 | 629.1 | 4338.5 | 4512.4 | 2268.2 | 1.038x |
| Specialized sparse / 3 | 629.3 | 4212.2 | 2613.1 | 2125.9 | 1.019x |
| Specialized sparse / 4 | 630.0 | 4645.2 | 1249.0 | 2123.3 | 1.029x |
| Specialized sparse / 5 | 630.0 | 3916.9 | 1035.8 | 2650.5 | 1.024x |
| Generic no skip / 0 | 77.3 | 45532.2 | 11335.8 | 2213.7 | 1.014x |
| Generic no skip / 1 | 77.8 | 44821.9 | 11192.3 | 2225.8 | 1.009x |
| Generic no skip / 2 | 77.8 | 44680.2 | 11578.9 | 2195.4 | 1.004x |
| Generic no skip / 3 | 77.8 | 44290.2 | 11282.7 | 2167.1 | 1.004x |
| Generic no skip / 4 | 77.8 | 44346.2 | 11102.2 | 2244.8 | 1.002x |
| Generic no skip / 5 | 77.8 | 44337.8 | 11093.9 | 2187.9 | 1.024x |
| Specialized no skip / 0 | 670.7 | 29521.8 | 7850.3 | 2206.5 | 1.040x |
| Specialized no skip / 1 | 627.9 | 29983.8 | 8441.5 | 2062.8 | 1.015x |
| Specialized no skip / 2 | 627.7 | 23922.6 | 8619.2 | 2033.3 | 1.001x |
| Specialized no skip / 3 | 627.9 | 23713.3 | 6717.8 | 2435.9 | 1.029x |
| Specialized no skip / 4 | 627.7 | 23757.0 | 6822.6 | 2416.0 | 1.003x |
| Specialized no skip / 5 | 628.3 | 23743.2 | 6744.3 | 2301.9 | 1.016x |

## Interpretation limits and provenance

The no-skip probes illustrate the boundary problem: specialized projection warps have shorter clock intervals while the full specialized no-skip kernel/forward is slower. Warp-local elapsed time omits the whole-device distribution of resident and queued blocks. The cycle partitions must not be used as device latency or as a hardware occupancy measurement.

One trained seed, two thresholds, one physical GPU and one uncached full-sequence shape. CUPTI sample size and measurement perturbation are explicit. No individual full-device timings are invented for operations fused into a kernel. Internal SM clocks include warp scheduling stalls and instrumentation, and do not measure DRAM bytes, hardware FLOPs, pure arithmetic cost or causal percentages of the speedup. No kernel or model tuning was performed. The original trained tile-model quality penalty remains.

Every raw kernel timeline, full numerical comparison, paired timing sample, per-layer stage interval, internal cycle/density sample, activation/weight diagnostic, setup/compiler log and transfer inventory is retained under artifacts/. The reducer rejects incomplete timing/full validation coverage and different GPU UUIDs.

Source scripts: [20_analyze.py](../20_analyze.py), [21_report.py](../21_report.py), [stages.py](../stages.py), [profiling.py](../profiling.py), [internal.py](../internal.py), [native copy mapper](../trace_copy_mapping.py).
Evidence: [decomposition.json](../results/decomposition.json) and [complete kernel-node inventory](../results/kernel-nodes.csv). Figure caption: full forward distributions and per-layer joint timings, followed by diagnostic warp-cycle partitions; units and sampling follow the tables above.
Figure: [four-page PDF](../figures/01-forward-latency-decomposition.pdf).
