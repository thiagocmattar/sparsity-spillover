# Run 023 — Pythia-70M Sakana-derived sparse-kernel sentinels

**Status:** six-condition sentinel phase executed and verified; remainder not
launched; no active RunPod compute.

## Question

Can an explicitly labeled derivative of the official Sparse-er/Faster-LLMs
implementation execute the retained Pythia-70M ladder faithfully, and does
measured prefill speed track canonical logical opportunity (`R_model`) or the
narrower opportunity covered by the kernels that actually run
(`R_covered`)?

This is a runtime benchmark of completed Run-018 checkpoints. It does not train
or tune a model, change a gate, apply TEAL clipping, or reinterpret `R_model`
as removed FLOPs or speedup.

## Fixed source evidence

All checkpoints are the Run-018 seed-1234 random-pretraining realizations after
712 optimizer boundaries and exactly 1,493,172,224 MiniPile input tokens.
Released Pythia weights were never loaded. The architecture is six layers with
`d_model=512`, `d_mlp=2048`, eight 64-dimensional heads, vocabulary 50,304,
and sequence length 2,048. Every checkpoint file and its canonical
`logical_products.json` is size/hash pinned in `config.yaml`.

Every loss check covers all 500 validation documents through all 338 complete
2,048-token blocks (692,224 input tokens); the 1,444-token tail is reported as
excluded. Source FP32 parameters under FP16 autocast use Run 018's batch size
of four and must reproduce its loss within 0.0002. The matched native/sparse
BF16 equivalence passes use batch size 32, and their losses must agree within
0.001. A7 Q-only/V-only opportunity counting is a separate complete pass at
batch size one, matching Run 018's logical-product diagnostic rather than a
timing batch.

The twelve-condition inventory is A0, A1-H, and A4-OL1/A7-OL1 at
`kappa={0,0.01,0.05,0.1,0.5}`. The first execution phase contains six
sentinels:

- A0 and A1-H;
- A4-OL1 at kappa 0 and 0.5; and
- A7-OL1 at kappa 0 and 0.5.

The six interior A4/A7 thresholds are a separate remainder phase and are not
authorized by a sentinel launch. They require review of correctness,
break-even behavior, and measurement stability first.

## Sakana-derived implementation

The official repository is pinned at commit
`661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`. The official SparseLM0.5B
benchmark runs from a clean checkout as a positive control. Pythia runs from a
second checkout with the exact run-owned `sakana-pythia70.patch`; results are
labeled **Sakana-derived**, never official Sakana Pythia support.

The patch makes only the compatibility changes needed here:

1. all warp lanes participate in full-mask shuffles for output widths below
   256, fixing the N=128 failure isolated in Run 022;
2. a CUDA ballot-prefix packer preserves every signed BF16 value satisfying
   `x != 0`, in column order, with no ReLU assumption, pruning, or capacity
   truncation;
3. reusable pack/output workspaces avoid allocator cost inside timed model
   paths; and
4. dtype, shape, stride, capacity, device, and launch contracts fail closed.

The linear adapters preserve the original bias and the existing topology/gate
order. A0/A1-H replace only `h -> W2`. A4/A7 replace
`a -> QKV`, `m -> W1`, `h -> W2`, and `z -> Wo`. The LM head remains
dense.

## Runtime measurements

The workload is uncached full-sequence prefill, not generation: `use_cache`
is false, with no `torch.compile` and no CUDA graphs. All reported runtime
comparisons for a phase use one exact Hopper SKU. Batch sizes are 1 and 32.
Variant order is deterministically randomized within paired timing blocks.

For each condition the run records:

- native dense, adapter-dense, and sparse-linear full-model latency,
  tokens/second, tails, paired block speedups, and peak memory;
- pack-only, kernel-only, and pack-plus-kernel timings for every covered
  operation and layer;
- exact/near-zero integer counts, activation RMS/L2 summaries, row occupancy,
  and operation-width tile occupancy;
- source, dense-BF16, and sparse-BF16 complete-validation losses; and
- environment, GPU, upstream commit/blob IDs, patch, checkpoint, cache, and
  result identities.

`R_covered,full` uses the unchanged canonical Run-018 model-product
denominator and the canonical integer zero-product count for only the linear
operations replaced in the timed full model. This is the only covered
opportunity paired with full-model speedup. For A7, a separate
`R_covered,linear+attention` additionally includes the full-validation Q-only
and V-only causal integer opportunities exercised by the standalone attention
composition. It is paired only with those separate component/composition
timings. It does not credit the Q/K union because the tested QK composition
packs Q only, and it does not credit a P/V union beyond V because the tested PV
identity is `(V^T @ P^T)^T`. Physical full-GEMM work is recorded separately
from valid causal products.

## Attention boundary and stop rule

Sakana's released high-level path specializes feed-forward computation; it is
not a drop-in causal-attention implementation. Run 023 leaves attention dense in the full-model sparse-linear path and gives A7 a separate
batch-1, per-head composition:

- Q-packed `[2048,64] @ [64,2048]` for scores; and
- V-packed `[64,2048] @ [2048,2048]`, transposed back, for PV.

The comparison includes packing, eight-head dispatch, transposes, scaling,
causal masking, and softmax. It is compared with the same unfused per-head
dense composition. If it does not break even, the attention path stops as a
non-break-even result. This run does not authorize development of a new fused
causal sparse kernel.

Correctness stops the entire phase on an upstream-positive-control failure,
source/blob/hash mismatch, N=128 regression, signed-pack mismatch, dropped
value, unsafe capacity, nonfinite value, primitive relative-L2 error above
0.02, source-loss mismatch, or sparse full-validation loss mismatch. A sparse
slowdown is a valid result, not an infrastructure failure.

## Diagnostics retained and deferred

Run 018 already retains per-layer weight norms and the training-time OL1
gradient-boundary diagnostics; this run preserves their provenance rather than
pretending to reconstruct them. Forward-only exact/near-zero counts, RMS/L2,
occupancy, and logical coverage are newly recorded. No gradient interaction is
available from inference, and no new checkpoint is needed because all source
checkpoints remain retained and hash pinned.

A0/A1-H post-hoc TEAL frontiers are deliberately deferred. They would add
operating points and should be benchmarked only after the unmodified trained
sentinels establish the runtime path.

## Paper interpretation

The paper-relevant outcome is a calibration table linking validation loss,
canonical `R_model`, kernel-specific `R_covered`, and measured speedup. A
positive association would support `R_model` as a useful opportunity signal
under a declared kernel coverage boundary. Weak or reversed association,
packing domination, or no end-to-end break-even would show that logical
opportunity is insufficient without layout, occupancy, and integration costs.
One seed, one GPU SKU, full-sequence prefill, and unfused attention limit the
claim.

## Files and launch boundary

- `config.yaml`: twelve hash-pinned conditions and the six-condition phase.
- `sakana-pythia70.patch`: exact derivative against the pinned official tree.
- `benchmark_core.py`: integer pooling, ELL checks, coverage, and paired timing.
- `pythia_sparse.py`: run-local linear adapters and attention compositions.
- `00_prepare_inputs.py`: phase-specific allowlisted transfer inventory/archive.
- `01_static_preflight.py`: checkpoints, cache, topology, source blobs, and patch.
- `02_remote_preflight.py`: compiled Hopper checks over all required shapes.
- `03_benchmark.py`: complete validation, occupancy, primitive, and model timing.
- `04_verify.py`: fail-closed phase artifact verifier.
- `00_setup_remote.sh`, `05_start_worker.sh`, and `06_monitor.py`: pinned,
  disconnect-safe remote lifecycle.
- `DEPLOYMENT_PLAYBOOK.md`: provisioning, monitoring, retrieval, and teardown.

The current implementation is not launch authorization. Live stock, price,
balance, maximum billable envelope, transfer inventory, and the exact Pod
definition must be reported and explicitly approved before provisioning.

## Local verification

The static preflight passes all twelve checkpoint/cache/topology/hash records,
and the exact derivative patch reverse-checks against the pinned upstream
commit. All 15 focused Run-023 tests and all 207 tests in the complete
bootstrap suite pass; Python compilation and Bash syntax checks also pass. The
sealed sentinel allowlist contains 63 files: six final model-only checkpoints,
their canonical logical-product records, the public validation cache, and
source/test files, with no optimizer state, training cache, or credentials.
The local RTX 5070 Ti Laptop GPU is not Hopper and no local `nvcc` is
available, so compilation and all CUDA behavior remain deliberately gated by
`02_remote_preflight.py` before any checkpoint benchmark. No cloud resource
was created during implementation.

## Sentinel execution outcome — 2026-09-04

The approved sentinel phase ran on one community H100 NVL at `$2.59/GPU-hour`.
The official unmodified SparseLM0.5B control passed at `1.2876x`, all seven
Pythia shape gates passed, all six source losses reproduced exactly, and every
sparse/dense BF16 loss difference was below `0.001`. The final benchmark
covered 108 linear primitives, 12 A7 layer-level attention compositions, and
all 338 complete validation blocks per condition.

No tested sparse path broke even. The best full-model result was A4-OL1 at
`kappa=0.5`, with paired speedup `0.9835x` at batch one and `0.7066x` at batch
32. None of the 108 pack-plus-kernel linear measurements exceeded `1x`, and
none of the 12 attention compositions exceeded `1x`; the best attention result
was `0.3612x`. The configured attention stop rule therefore fired. The six
interior A4/A7 thresholds remain unlaunched and require a new explicit decision
if their value as a negative association curve is judged worth measuring.

Attempt 005 completed every benchmark but its original verifier incorrectly
required strictly positive RMS for an all-zero activation record. A4-OL1 at
`kappa=0.5`, `z.layer_3` contained exactly 354,418,688 zeros out of
354,418,688 pooled elements, so RMS zero was correct. Verification-only
Recovery 007 accepted finite nonnegative RMS while requiring RMS zero if and
only if the integer-pooled record is entirely zero. It passed without changing
or rerunning any benchmark.

The measured benchmark body took 860.41 seconds. Pod lifetime through confirmed
termination was bounded by 1.0558–1.0605 hours; estimated experiment-incremental
cost was `$2.75–$2.76`, including transient Pod storage. The billing endpoint
was still delayed at teardown, so this is a rate-times-duration estimate rather
than a settled invoice. After hash-verified retrieval and independent local
verification, the Pod was terminated. RunPod reported zero Pods and zero
endpoints; the pre-existing unattached 100 GB standard network volume was left
unchanged.

See `observations/001-pythia70m-sentinel-sparse-kernel-feasibility.md` for the
scientific interpretation, `results/sentinel-summary.csv` for the calibration
table, and `launch-control/runpod-closeout-20260904.json` for execution and
teardown provenance.
