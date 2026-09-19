# Direct h/z strategy port to a/m: correct, slower

## Question and method

Would moving the final h/z load-avoiding hybrid to a/m reduce latency for
14M T7/Pall at kappa 0.5? This tests the implementation alternative raised
while reviewing the manuscript's explanation of per-operation speedups.

Use the approved [design](../../../analyses/024-2026-09-17-h-only-kernel-latency/AM-LOAD-AVOIDANCE-DESIGN.md),
the retained Run029 c30 checkpoint, and the same Run037 protocol. Checkpoint
SHA256: `f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
The original training was random initialization, seed 1234, step 712 and
1,493,172,224 input tokens. This experiment performs no training. Gates,
thresholds, weights and all other kernel paths remain fixed.

The port uses eight real rows per group, K16 support masks inspected before
weight requests, scalar arithmetic for safe rows with at most two nonzeros,
and padded M16 matrix instructions otherwise. Standalone a/m outputs have
widths 384/512. Compare frozen K050, a-only, m-only, both-site port, and a
both-site control with the same new layout but sparse logic disabled.
The last control preserves the gated model and all other optimized paths;
it is not the base model or a fully dense kernel.

One physical RTX 5090; BF16, batch 1, sequence length 2,048, uncached causal
inference with all 50,304 logits and CUDA graphs. Use the same 64 validation
timing identities, seven passes and three fresh processes per mode, in fixed
randomized order (seed 2504). Runtime seed 2801. Geometric means of paired,
synchronized full-model host timings are the latency estimand.

## Coverage and correctness

Each of 15 processes checked all 338 complete validation blocks from 500
MiniPile documents: 692,224 input tokens, 691,886 prediction tokens and the
unchanged 1,444-token excluded tail. All original numerical bounds passed.
Maximum measured logit error and pooled-loss difference are both **zero**.
BF16 validation loss is 5.83130066301001 for all paths and processes. This
runtime check does not replace the canonical paper quality evaluation.

All 224 direct cases (32 synthetic, 192 captured training operands) matched
native and frozen projection outputs with zero error and passed independent
counter checks. All five preliminary smokes passed; smoke timings are excluded
from the scientific results. Untimed diagnostics cover all 338 blocks once per
mode, with exact/near-zero counts, RMS, row histograms, tile occupancy, weight
norms, logical opportunities and verified instrumented work counters.

## Results

Table caption: full-model latency for the same checkpoint and workload, changing
only the selected a/m implementation. Ranges span the three process geometric
means; they are descriptive ranges, not confidence intervals.

| Implementation | Latency (ms) | Process range (ms) | Increase over frozen |
|---|---:|---|---:|
| Frozen K050 | 0.494995 | 0.492736--0.497663 | reference |
| Port at a | 0.580284 | 0.580026--0.580607 | 17.23% |
| Port at m | 0.630549 | 0.629739--0.631037 | 27.38% |
| Port at a and m | 0.715227 | 0.713795--0.716800 | 44.49% |
| Both-port layout, sparse logic off | 0.708533 | 0.706923--0.709426 | 43.14% |

All port process means exceed every frozen process mean. The both-site sparse
port is also 6.694 microseconds slower than its matching dense-layout control;
the descriptive difference span is 4.369--9.877 microseconds. Most regression
therefore persists in the new execution layout even without the sparse logic.
This control does not isolate padding from every other layout/scheduling change.

## Why the direct port does not help here

Integer counts are pooled over the complete validation pass before division.
The following sparsities refer to actual BF16 operands at each site, not the
canonical model-wide sparsity statistic.

| Site | Element sparsity (%) | Rows with at most two nonzeros (%) | Mean nonzeros per row |
|---|---:|---:|---:|
| a | 75.6292 | 15.3706 | 31.1947 |
| m | 68.6494 | 0.0000 | 40.1288 |
| h | 99.8755 | 92.7546 | 0.6376 |
| z | 99.9170 | 99.7254 | 0.1062 |

Reducing a/m tile height from 16 to 8 changes the empty-tile fraction from
8.4385% to 11.1218% at a, and from 0% to 0% at m. Most a/m groups therefore
still need the matrix fallback. The port issues **172,966,272** a instructions
versus **91,268,736** in the frozen path (1.895x), and **265,814,016** m
instructions versus **132,907,008** (2x). Its padded M16 fallback performs work
for eight unused rows; the finer grouping does not offset that cost here.

Within the new padded layout, sparse logic avoids 12.9886% of source-level
weight-element requests at a and 0% at m. These are requests implied by the
implemented instructions and scalar products, not measured DRAM traffic and
not a direct comparison with CUTLASS weight traffic. Cache reuse is not removed.
At h/z, by comparison, most rows have at most two nonzeros, making the scalar
replacement useful far more often. Their paths and work counts remain unchanged
in every port mode. Full-model logical opportunity counts also match frozen.

## Verdict and limits

**Do not replace the frozen a/m paths with this direct port.** The port extends
correctly but increases latency in all tested placements. The result supports
the specific implementation choice on this retained checkpoint and workload.
It does not show that a/m load avoidance is intrinsically unhelpful, that an
unpadded design would fail, or that the selected kernel is globally optimal.
No new shape sweep, checkpoint selection or optimization search was performed.

## Provenance and recovery

Sources: [CUDA port](../candidate/projection.cu), [direct checks](../06_cuda_checks.py),
[benchmark](../02_benchmark.py), [diagnostics](../diagnostics.py),
[latency reduction](../07_reduce.py), [mechanism reduction](../12_mechanism.py),
and [local verification](../13_verify_local_reduction.py).
Results: [latency JSON](../results/am-load-avoidance.json),
[mechanism JSON](../results/mechanism.json),
[transfer verification](../artifacts/verification.json), and
[raw-data verification](../artifacts/local-reduction-verification.json).
No figure was required for this short test.

Run038's original 16-row fixture was rejected by the frozen comparison before
candidate qualification. That failure is retained and recovered; Run039 uses
32 synthetic rows, with identical candidate CUDA and full-model protocol.
All 190 Run039 output files and 38 Run038 output files were hash verified locally.
The 13,440 raw timing samples were rechecked locally, including all 50 source
file identities. The shared Pod was deleted after recovery; no Pods remain.
GPU cost is approximately USD 0.616 plus temporary storage. See
[closeout](../artifacts/closeout/teardown.json).
