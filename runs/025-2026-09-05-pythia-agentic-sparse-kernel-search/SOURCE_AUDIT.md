# U0 to P0 source audit (before CUDA execution)

Question: can the official optimized path execute our exact retained Pythia
checkpoints without dropping signed nonzeros or changing trained gates?

Source: [official Sakana repository, pinned commit](https://github.com/SakanaAI/sparser-faster-llms/tree/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5).
`upstream/matmul_t2d.cu` and the MIT license are preserved unchanged. The file's
canonical Git-object LF SHA256 is
`2f96866ee9227eefaf4707524a4df0f51a7a44ef7275d9a24a7e6a3ba6b302c4`.
The historical Run 022 hash
`c01fda6953daabf1243ab5c2a2e6da8d0f4400c6cf490595884548dc44389be6`
is the same source with CRLF line endings. This preflight initially rejected
that historical hash; direct Git-object comparison established the cause.
We fixed the expected identity, not the source. No historical file was edited.
The vendored file's original trailing whitespace is intentionally retained
to preserve its SHA256. Run-local attributes disable whitespace cleanup for
that reference and preserve the exact serialized inventory bytes.

## Compatibility findings

| Official assumption | Pythia requirement | P0 adaptation |
| --- | --- | --- |
| Non-gated T2D dispatch: N=2048, K=5632/8192 | N=128/512/1024 and other projection widths | Dynamic K/N, output tiles of 256, predicated tail loads/stores |
| Factor-8 tile: count + at most 31 payloads | A0 can be dense; all signed nonzeros must survive | Exact count + 256 payload capacity, padded to 288 words per 256-value tile |
| Fused D2T uses positive ReLU output and fixed compression | GELU A0, biases, signed attention operands, threshold gates | Separate exact signed packing of the actual post-gate BF16 operand |
| Hopper WGMMA producer | RTX 5090 is Blackwell | Do not compile the Hopper-only producer as a purported portable baseline |
| BF16-rounded products, FP32 sum | Match dense BF16 operands / FP32 accumulation | FP32 FMA and bias before final BF16 rounding; fixed numerical gates |
| Launch without explicit stream | PyTorch current-stream execution | CUDA device guard/current stream and launch error checks |
| Full-mask shuffle with width assumptions | N=128 regression must be covered | All 32 lanes participate; predicate only memory operations |

`kernels/twell_pythia.cu` is an explicitly documented descendant of the
upstream tile/index/BF16 warp-broadcast T2D algorithm, not the unchanged
published optimized FFN. It retains eight output elements per lane and the
packed word layout. Capacity and precision adaptations are correctness
integration; they are NOT credited to later agent optimization.

Full-capacity packing is deliberately not a storage compression claim. It can
be slower than dense, especially at A0. A later candidate may introduce safe
overflow/dense routes, but never discard values to obtain a gain. Dynamic
packing and all four projection calls are inside the full-model timer.

## Coverage and limits

P0 wraps W1, W2, fused QKV and attention-output Linear modules, preserving the
canonical loader's a/m/h/z gates, post-RoPE q/k/v gates, biases, residuals,
and LM head. QK, causal softmax, and PV remain dense SDPA. No actual sparse
attention kernel or winning optimized candidate exists yet. That work remains
in the approved bounded development phase and must be labeled new attention
code integrated with Sakana, not attributed to an upstream attention kernel.

The native timing reference swaps the original Linear objects back. All
timed input IDs rotate, paired in seed-fixed randomized order. Full logits,
not only the final token, are materialized. Capture hooks and profiling are
outside timing. Diagnostics which install attention ports run LAST; the
model is discarded afterward. Native/adapter-dense equality is CPU-tested.

Canonical retained `R_model`/`R_model_max` integer-count records are included
as source evidence. Calibration BF16 activation counts are reported separately
on 64 train-split blocks; they are not mislabeled canonical validation counts
or runtime savings. Full frozen evaluation still needs the planned complete
count collection. No gradient-interaction claims can be reconstructed by
this inference-only benchmark.

## Fixed pretrial gates and test limits

Primitive: relative L2 <= 0.02 AND elementwise
`abs(diff) <= 0.125 + 0.02*abs(reference)`; exact packing equality additionally
required. Model: relative L2 <= 0.02 AND elementwise
`abs(diff) <= 0.25 + 0.02*abs(reference)`, AND absolute complete-validation
loss difference <= 0.001. Both operands must be finite. A failure stops the
attempt; thresholds must not be loosened after inspecting a result.

P0 primitive tests use deterministic signed/zero/full inputs, bias, all
projection shapes, and a non-default CUDA stream. CPU oracle tests do not
validate CUDA compilation, memory safety, launch behavior, or GPU performance.
The paid pilot must establish these before P0 is frozen for search.

The unmodified U0 positive control runs separately on H100 using SparseLM0.5B
revision `7c2a0473ec982facd5b21af81fe39c1783f3407e`, the official FFN benchmark
at B=64,T=2048, 5 warmups and 50 repetitions. Its scope is not full-model
Pythia performance. Preserve failures and unsupported configurations.
