# Source review and measured optimization budget

Question: what can external sparse-kernel implementations teach us about
accelerating retained70M T2/Ph checkpoints at kappa=.05/.1?

Method: inspect pinned official source trees and primary documentation; inspect
two kernel archive cards and a few example rows without executing their code;
compare with Run023/025/029 history; reduce existing Run049 profiler traces.
No inference or new GPU timing was performed. Source identities and hashes are
in `../data/source-inventory.json`; reduction source is `../01_profile_audit.py`.

## TEAL: useful load policy, different workload

The [released GEMV implementation](https://github.com/FasterDecoding/TEAL/blob/fb7373c93ac3594817c9ee64d4e08b47430a1822/kernels/sparse_gemv.py)
masks weight loads using nonzero activation decisions, divides the reduction
across K blocks, uses FP32 local sums and atomic output accumulation, and tunes
tile sizes. Its wrapper uses the sparse route only when sequence length is1;
longer sequences call dense matmul. The released output buffer is FP16 and its
gate is magnitude-based with strict greater-than. Our BF16, one-sided,
equality-preserving gates and 2048-token workload require adaptation.

Transferable idea: avoid a mandatory global packing pass and skip weight rows
at load time. The proposed parallel-K family adds a deterministic FP32
workspace reduction and includes its initialization/traffic in timing. It is
not an unchanged TEAL benchmark. Independent row execution could lose the
weight reuse that makes a large dense GEMM efficient.

## Sakana: compact streaming and producer fusion, with constraints

The [TwELL consumer](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/custom_models/twell_modules/matmul_t2d.cu)
stores one count plus 31 packed index/BF16-value payloads per 256 features.
A warp loads the record collectively, broadcasts entries and vector-loads
contiguous weight outputs. This bounds per-thread metadata storage instead of
keeping an increasingly wide list in every thread. The released non-gated
consumer restricts output width to 2048 and input width to 5632/8192, so it is
not a direct512-output Pythia kernel. Its BF16 multiply followed by FP32 sum
also differs from our FP32-product numerical contract.

The [producer](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/custom_models/twell_modules/matmul_d2t.cu)
combines dense multiplication, ReLU and packing using Hopper machinery. Its
fixed-capacity writes and optional wrapping do not supply the lossless overflow
guarantee this experiment requires. The
[training hybrid format](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/custom_models/hybrid_modules/KNOBS.md)
adds a dense tail, but its documentation explicitly permits discarding overflow
and says exhausted tail capacity drops rows. Those modes cannot be copied into
our correctness-preserving study. Any new format must handle worst-case input
without pruning, with tail/chunk costs measured.

The [benchmark entry point](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/benchmark_inference.py)
defaults to batch 64, sequence 2048 and BF16. The extension defaults to sm90a.
Sakana's non-gated and two-up-projection gated MLPs are distinct; our T2
threshold gates do not turn Pythia into the latter. Thus published ratios
cannot predict our batch 1 RTX5090 latency.

## What the repository already tried

[Run023](../../../runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/README.md)
used this same upstream commit. Its signed, shape-adapted, lossless packer and
consumer passed the then-declared checks, but none of 108 pack-plus-kernel
linear measurements beat dense. The best sparse full-model H100 result was
0.9835x at batch 1. These were different checkpoints and an eager Hopper
protocol, so this is a warning about a port, not a result for the current pair.

[P0 and K001](../../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/candidates/k001/README.md)
already adapted the TwELL consumer, including FP32 products and Pythia bias.
P0 reserved 288 words per 256 inputs to preserve all nonzeros; K001 removed the
intermediate, ballot-scanned inputs, used multiple row warps and vectorized
weight loads. K009 selected its last-two-layer 70M policy. Numerical failures
were already traced to reduction/rounding propagation; merely using FP32 does
not guarantee full-model equivalence. These existing kernels should be
controls, not renamed discoveries. A new packed variant must justify itself
through tighter lossless storage, scheduling/reuse or producer integration.

K004 already bypassed all-zero16x32 tiles and ran tensor-core dot products on
the remainder; K007 varied its launch geometry. Therefore the proposed family C
must add active-K compaction inside partially occupied tiles, rather than
relabeling empty-tile skipping as a new algorithm.

Run049 further shows why increasing a scalar capacity is inadequate: at .1,
limit64 reduced issued h/z MMA work about 74% but increased scalar products
about 10.6x versus limit8 and produced register spilling. Limits16/32/64 all
failed the final elementwise bound. This is evidence against that particular
family extension, not evidence that 98.9% h zeros and 96.8% z zeros are useless.

## Large kernel archives and their appropriate use

The current [Sakana AI CUDA Engineer archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
contains 30,615 candidate rows, with reference programs, correctness flags,
profiles, timings and errors. It includes failures. Inspection of level1
rows 7--13 found both passing and failing labels; the inspected tiled GEMM
example is an ordinary FP32 square matmul, not a BF16 h/z sparse solution.
Use it for bounded examples of shared-memory reuse, tiling, reductions and
fusion, never as a bank of universally verified speedups. Its row-level
correctness labels do not replace our shape, overflow and full-model checks.

[GPU MODE KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook)
contains 18,162 PyTorch/Triton program pairs generated through Inductor. The
inspected GCN entry delegates matrix multiplies to external dense routines and
fuses ELU in Triton. Its lesson here is to retain strong GEMM libraries while
fusing surrounding pointwise work; the archive is not exclusively independent
hand-optimized sparse kernels. Dataset and underlying code licenses are
recorded separately and must be checked before copying an implementation.

A targeted search did not verify a corresponding official Nous Research
release. This is an attribution uncertainty, not a claim that none exists.
The two confirmed archives above are useful regardless of that attribution.

## Other hardware paths worth comparing

[CUDA12.8 PTX](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-sp)
specifies BF16 sparse MMA and recommends ordered metadata; sparse operand A
must obey its structural contract. This motivates an exact 2:4 eligible-tile
route with dense fallback. No prediction of 2:4 eligibility follows from the
marginal zero percentage, and metadata/packing can erase arithmetic gains.

[CUDA12.8 cuSPARSE](https://docs.nvidia.com/cuda/archive/12.8.1/cusparse/index.html#cusparsespmm)
provides SpMM as a useful independent reference. Its actual BF16 algorithm,
dynamic metadata and graph compatibility must be checked in the pinned
environment. Any input-dependent conversion or host synchronization counts;
prepacked static-matrix timings cannot stand in for dynamic activation costs.

## Profile result and interpretation

Coverage: one retained Run049 process per checkpoint, four graph-profiled
inputs, six layers and 24 calls per projection type. Eager CPU shape metadata
labels the graph GEMM subsequence only after exact name/order/count checks.
The full eager/graph sequences differ for gated checkpoints, so a positional
zip of all kernels would be invalid. Duration units are microseconds.

| Condition | h/z GEMMs | Gates + associated copy | Saving to match efficient Base |
| --- | ---: | ---: | ---: |
| Base |202.854 us|0|0|
| T2/Ph .05 |203.022 us|82.047 us|69.097 us|
| T2/Ph .1 |202.846 us|81.782 us|70.488 us|

Caption: GEMMs and gate categories are sums of instrumented GPU event
durations divided by four forwards. The final column comes from the separate
unprofiled latency summary. Copy attribution is inferred from functional
masked_fill and its absence on Base. These columns do not form a causal
latency decomposition; overlap, caching, launches and instrumentation matter.

The budget is large enough to motivate the [multi-family proposal](../README.md).
It also makes an improved dense gate/fusion control mandatory. h/z GEMMs alone
occupy only about 15% of current full-forward latency; an arbitrarily fast h/z
consumer cannot eliminate the dense head, attention or up-projection costs.
No speedup claim, new manuscript narrative or finding promotion follows from
this exploratory review.
