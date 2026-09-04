# Sparse-kernel promotion plan after Run 022

This is a staged research plan, not approval to create or launch later numbered runs.

## Target universe

The trained endpoint set is fixed at 24 checkpoints: for each of Pythia-14M and Pythia-70M, A0, A1-H, A4-OL1 at `kappa in {0, 0.01, 0.05, 0.1, 0.5}`, and A7-OL1 at the same five kappas. The four historical Run-004 `l1_naive` conditions are not part of this sweep.

The approved source families are Run 004 for 14M A0/A1-H, Run 015 for corrected 14M A4-OL1, Run 014 for 14M A7-OL1, and Run 018 for the canonical 70M ladder. Every future run must resolve and hash-lock its exact attempt/checkpoint rather than relying on these family-level labels alone.

If the stored ten-point post-hoc TEAL frontiers for A0 and A1-H are later included, the complete runtime matrix contains 60 configurations: per scale, ten A0 thresholds, ten A1-H thresholds, five A4 checkpoints, and five A7 checkpoints. TEAL operating points are configurations of a common checkpoint, not additional trained models.

All reported comparisons will keep GPU SKU, CUDA/kernel commit, PyTorch stack, sequence length, batch size, timing method, warmup, repetitions, and validation block selection matched. `R_model` remains the canonical logical-product opportunity. Measured latency/throughput is reported separately. A run-local `R_covered` may describe the integer-count fraction of the canonical denominator covered by kernels actually executed.

## Compatibility boundary

The published high-level non-gated TwELL module is not a drop-in Pythia module: the pinned source specializes the published SparseLM MLP shapes (including output width 2,048 and feature widths 5,632/8,192), whereas the selected Pythia models use different widths. Re-implementing the full fused module before measuring would mix kernel-porting risk with the scientific question. The first ladder therefore uses the upstream generic `ell_spmm_raw` operation with exact run-local packing. Its `M x K` sparse-left by `K x N` dense-right contract covers the relevant Pythia projection widths because they are multiples of its eight-value output vector, but it does not fuse surrounding operations.

The pinned build targets Hopper `sm_90a`; RTX/Ada/Blackwell and A100 are not interchangeable cheap substitutes for this exact implementation. H100 SXM is the primary measurement device, with H100 NVL or H200 usable for engineering fallbacks. Final runtime comparisons must use one exact SKU.

## Stage gates

1. **Run 022 — A0 14M:** reproduce the official SparseLM positive control; measure dense instrumentation overhead; establish exact ELL correctness and the dense A0 negative control. Stop and report.
2. **A1-H 14M:** run the same `h -> W2` primitive on naturally exact-zero ReLU activations at the un-clipped endpoint. This is the first positive sparsity test. Stop and report.
3. **A0/A1-H TEAL 14M:** apply the already stored ten-point post-hoc clipping definitions to both checkpoints. Start with one high-sparsity sentinel per checkpoint; if validation identity, correctness, and timing dynamic range hold, fill all stored points. This is the first test of monotonic `R_covered` versus kernel speedup. Stop and report.
4. **A4-OL1 14M:** benchmark only `kappa=0` and `0.5` sentinels first. If numerical correctness and useful timing dynamic range hold, fill `0.01, 0.05, 0.1`. Cover only downstream linear operations for which exact sparse operands and correct bias/gate ordering are implemented.
5. **A7-OL1 14M projection coverage:** repeat the projection-compatible portion with `kappa=0` and `0.5`. Report canonical total `R_model`, its kernel-covered numerator, and the uncovered QK/PV attention component separately.
6. **A7 attention feasibility:** prototype one forward-only signed sparse attention composition on a single layer/head/shape using the generic raw ELL operator. Compare against Flash SDPA, including pack/index construction, per-head dispatch, required transposes, masking, scaling, and softmax. Proceed to a custom fused/batched kernel only if measured break-even—not logical opportunity alone—justifies it. The released TwELL MLP converter does not implement attention and its ReLU-positive high-level packing rule is not valid for A7's signed symmetric-threshold `q`, `k`, and `v` tensors; the low-level ELL values themselves are signed BF16 and can be packed by our exact wrapper.
7. **Pythia-70M:** only after the 14M gates pass, repeat A0 then A1-H/TEAL, A4 sentinels/ladder, and A7 projection/attention decisions. Do not infer 70M compatibility from 14M because the shapes and launch regimes change.
8. **Final fixed-hardware sweep:** rerun the selected configurations on one exact GPU SKU in randomized/interleaved order and fit descriptive relationships between `R_model`, `R_covered`, occupancy/capacity distributions, and measured speedup. Do not claim a universal conversion from `R_model` to speedup.

## Per-test playbooks

### Projection or MLP test

Validate the original checkpoint and complete-validation loss; capture and integer-pool exact/near-zero counts; inspect row/tile NNZ and TwELL capacity overflow; pack every nonzero without truncation; compare sparse output with dense BF16 and FP32 references; time dense kernel, sparse kernel, pack, and pack-plus-sparse; then measure end-to-end model latency only after a correct model integration exists. Preserve bias addition and any subsequent gate in their original order.

### TEAL operating-point test

Load one source checkpoint, reproduce its unclipped loss, apply only the archived post-hoc clipping definition and threshold at its declared sites, re-evaluate all 338 complete blocks, and benchmark with the same workload. Do not label a TEAL operating point as a separately trained model, and do not use validation loss to tune a new threshold during this benchmark.

### Attention test

Separate QK and PV shapes and sparsity. For QK, exact-pack either each `q_post[T, d_h]` matrix and compute `Q K^T`, or pack `k_post` and transpose `K Q^T`; each option exploits only the selected operand's zeros and must not claim the union of Q and K logical opportunities. For PV, test the identity `P V = (V^T P^T)^T`, packing signed `V^T`; include all transpose and layout costs. The generic operator has no batch/head dimension, so head dispatch must also be timed.

Keep scaling, causal masking, softmax, and the repository's exact A7-Z-POST hook placement unchanged, and compare with eager/Flash reference semantics. If the raw composition loses after packing and dispatch, retain the result as an attention non-break-even bound rather than building a custom kernel. A projection-only result must never be presented as full A7 acceleration, and a one-sparse-operand QK result must never be credited with both Q and K zeros.

## Evidence needed for the paper

For each retained point: checkpoint and cache identity, validation loss and coverage, canonical `R_model` integer numerator/denominator, kernel-covered integer counts, per-site/layer occupancy, capacity/overflow statistics, raw timing blocks, environment/GPU identity, numerical error, and both kernel-only and end-to-end latency. Report negative controls and break-even failures; they delimit the meaning of logical opportunity.
