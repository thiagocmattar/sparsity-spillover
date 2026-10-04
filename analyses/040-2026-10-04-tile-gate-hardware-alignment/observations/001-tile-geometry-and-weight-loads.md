# Which tile structure can avoid GPU weight loads without breaking causality?

Question: select a scientifically explicit tile layout for the requested
training gate using the retained 14M, 31M and 70M inference implementations.
The target is transferable weight-load avoidance, not an unmeasured speedup.

Method: inspect source control flow, reduction dimensions, shape restrictions,
metadata consumers and retained numerical/performance observations. This is a
static source audit with source SHA256 records, not new training or profiling.
No figure or benchmark is generated.

## Existing paths

**14M, frozen K050:** K050 imports K049 for h/z output projections. In
`K049/joint.cu`, `inspect` creates per-row support bits in groups of16 features.
`joint` ORs the support of complex rows across eight real token rows. The
`accumulate` loop visits only pending K16 groups; the loads of weight fragments
`b0/b1` occur inside that loop, after the support selection and zero test.
An empty8x16 activation block therefore bypasses its corresponding weight
fragment requests and MMA instructions for this execution group. Short rows
use separate SIMT loads for only their surviving features; empty short rows
load no projection weights. The MMA atom is m16n8k16, with eight actual rows
and eight padded rows. The implementation hard-codes h512/z128/output128.

**31M, Run054:** the retained opt073-derived h/z consumer also uses eight
token rows and K16 support before loading `b0/b1`, with h1024/z256/output256.
It uses multiword support and a short-row cutoff of eight entries. Its
`accumulate` iterates `pending` K16 bits before the weight loads. This already
supplies a second size with the same reduction granularity, but is a shape
specialization, not a single generic binary or a tile-gate training result.

**70M, Run051:** the promoted policy uses feature-union compaction at h in
layers1--5, with row groups4/8/16 and a gathered BK32, output tile128. Layer0 h
and all z projections remain dense. `union_index` includes a feature only if
some row is active. `gathered` loads weights only for those feature indices,
with a mask excluding padded compacted positions. Contiguous K16 zeros are
compatible with this support representation, but an empty group in one token
does not remove its features from the union if another token uses them. This
existing policy does not establish exploitation of tile gates at z.

These consumers currently fuse or expect the original scalar gate. A later
tile-trained replay must preserve the tile gate in its producer or fuse that
exact operator before support detection. Reapplying only the old scalar
threshold to raw activations would change the trained model. Source-level
support compatibility does not make the complete existing replay a drop-in
tile-gate implementation.

**70M, Run053 aligned development:** `wide_sparse.tile_prepare` materializes
elementwise-gated values and K16 support for row groups16/32.
`tile_consume` places activation loads, weight loads and `tl.dot` inside
`if NO_SKIP or active`. An empty execution tile therefore bypasses those
source-level loads. This dimension-parameterized component is a concrete
starting point for a later generic consumer. The existing producer is still
elementwise: it does not implement the requested any-element tile gate.

Run053's aligned variants passed component checks but none passed the joint
moderate-threshold speed margin. At z.1 with32 rows, retained reduction work
was74.77%/62.67% at kappa=.05/.1, compared with14.10%/12.39% for the union path.
Individual zeros left occupied coarse tiles. No aligned full-model kernel was
promoted. Tile training may change this occupancy; that is the requested
hypothesis, not an observed result.

**MMA bypass is a separate mechanism:** K042's a/m CUTLASS override receives
both `FragmentA` and `FragmentB` and bypasses the MMA on zero A. It leaves the
GEMM weight-loading pipeline unchanged. Thus its MMA count reduction alone
does not establish skipped weight loads. The h/z and aligned paths above make
the relevant branch before their explicit weight load sites.

## Gate shape versus execution shape

For a linear layer `Y = X W^T + b`, let C be16 contiguous input features.
If `X[token,C]` is zero, that token needs no contribution from `W[:,C]`.
For a consumer sharing a weight load across R token rows, the load can be
omitted only if C is zero in all relevant rows (or the consumer dispatches
those rows separately). The bias, residual and other active contributions
still execute.

The proposed architectural gate partitions each token independently into1x16
tiles. Its reject mask uses only that token's raw gate inputs, following the
literal requested any-zero/any-below-kappa rule. Later inference may combine
the resulting support bits over8/16/32 rows and skip the weight group when the
union is empty. This grouping schedules an unchanged causal computation.

Applying the architectural any-element gate to an8x16 tile spanning tokens is
different: an earlier token's output changes when a later token trips the gate.
It breaks causal prefix invariance and can let training labels influence the
gate of earlier tokens. The causal attention mask does not repair a gate with
this dependency. A future-token perturbation/prefix test must cover the new
per-token gate. No cross-token gate is selected by this audit.

For single-token decoding there are no future token rows to share. The1x16
format remains well defined, but the execution strategy is GEMV/small batch
rather than the current T2048 prefill/full-sequence benchmark. Decoding speed
requires its own consumer and measurement; no claim carries over automatically.

## Portability and recommended initial geometry

| Architecture | FFN h input width | Attention z input width | h groups of16 | z groups of16 |
|---|---:|---:|---:|---:|
| Pythia14M | 512 | 128 | 32 | 8 |
| Pythia31M | 1024 | 256 | 64 | 16 |
| Pythia70M | 2048 | 512 | 128 | 32 |

All three widths divide16. The BF16 m16n8k16 instruction in the retained
CUDA consumers also uses16 reduction features; this matches the
[NVIDIA PTX definition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).
K32/K64 consumer steps may combine these elementary groups without changing
the architectural gate. Larger gate groups32/64 are different, more aggressive
training interventions and are not selected implicitly by consumer BK.

Per-token K16 support can describe linear projection inputs in other families
too. A portable implementation must obtain K/N from the actual projection,
handle tails for nondivisible widths, retain the original family-specific
activation/site semantics, and tune execution rows/output tiles for device,
precision and workload. Existing fused GPT-NeoX branches, head layouts and
hard-coded widths are not automatically portable. The format can generalize;
an identical measured speedup or globally optimal tile cannot be promised.

Skipped source-level requests are not measured DRAM bytes. Another token group
may load the same weights; L1/L2 may already cache them. Metadata scans, mask
generation, launch cost, reduced reuse and padding remain costs. Neither empty
tiles nor software MMA/reduction counters alone establish memory bandwidth
savings or end-to-end speedup. Later profiling and a same-checkpoint sparse
versus matched dense execution control would test that claim.

Recommendation for the first14M training question: propose1x16 at h/z with Ph
unchanged and kappa=0/.1. Retain both per-token gate occupancy and hardware
execution occupancy at8x16,16x16 and32x16, per site/layer and count-pooled, so
the future inference study can see whether shared weight-load skipping is
available. No additional size, kernel search or latency run is authorized by
this audit. The risk of dead tiles and the backward rule remain in the proposed
training design; hardware compatibility does not resolve trainability.

## Sources

- 14M: [K050 installer](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k050/candidate.py),
  [K049 wrapper](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/candidate.py),
  [K049 support and loads](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu),
  [K042 fragment bypass](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/projection.cu).
- 31M: [executed scope](../../../runs/054-2026-09-24-pythia31m-t2-ph/README.md),
  [joint kernel](../../../runs/054-2026-09-24-pythia31m-t2-ph/latency/kernel/joint/joint.cu).
- 70M: [Run051 primitives](../../../runs/051-2026-09-22-pythia70m-frozen-kernel-grid/primitives.py),
  [selected policy](../../../runs/051-2026-09-22-pythia70m-frozen-kernel-grid/provenance/selection-final.json),
  [Run053 union/aligned sources](../../../runs/053-2026-09-23-local70m-high-sparsity/wide_sparse.py),
  [aligned outcome](../../../runs/053-2026-09-23-local70m-high-sparsity/observations/005-aligned-tile-outcome.md).
- Architecture dimensions: [pinned scale setup](../../../manuscript/draft/10-appendix-experimental-setup.tex),
  Run054 executed kernel shape checks and Run051/053 operator dimensions.
- [Audit file hashes](../data/source-identity.json). No new runtime or memory
  transaction measurements were made.
