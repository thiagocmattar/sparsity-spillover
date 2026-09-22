# Implementation provenance

These run-local kernels are implementations of the approved Analysis036 design.
External repositories are source references, not executable dependencies fetched
at run time. Exact reviewed identities and licenses are recorded in
`analyses/036-2026-09-22-70m-kernel-family-exploration/data/source-inventory.json`.

- TEAL informs Family B's direct activation-conditioned weight loads and split-K
  comparison. The original decode speedups are not predictions for this workload.
- Sakana's Sparsity's Useful Usual Suspects/TwELL representation motivates
  Family A's packed index/value words and warp-distributed streaming. This port
  independently uses BF16 values, FP32 accumulation and worst-case segment
  storage, with no dropped overflow or quantization.
- Family D uses the already retained CUTLASS headers in Run028's verified archive,
  including their BSD-3-Clause license. Metadata reordering follows the public
  `tools/util/include/cutlass/util/host_reorder.h` algorithm. Its BF16 sparse GEMM
  accumulates into FP32, then adds lossless overflow and bias before BF16 output.
  No released checkpoint is substituted for the retained trained models.

All source snapshots used for attempts are retained locally under
`provenance/source-history/`; each attempt or upload records SHA256 identities.
