# Run069: vocabulary projection implementation efficiency

Status: design approved by the user's **Go on** on 2026-10-09; implementation
and local pre-launch verification in progress. No cloud launch or new speedup
result exists for this run.

## Question and matched scope

Can we locate and reduce the Run068 custom head's projection penalty against
native PyTorch/cuBLAS without changing any trained gate, operand, or output?
This targets kernel efficiency, not a new sparsification or training direction.

Reuse the two verified Run068 step-712 Pythia-14M checkpoints. Their original
random initialization, seed1234, matched training schedule, optimizer and pressure
history remain unchanged. There are **zero new optimizer updates**. Both have
h/z corner8x16 gates at kappa0 and Ph+head OL1 pressure; their final-LayerNorm
head-input corner16x16 gates have kappa0.1/0.5. A tile survives iff its FP32 signed
upper-left value is strictly greater than kappa. Live interiors retain their
signed values. Only head execution varies; body execution remains native dense.

The workload is uncached B1/T2048, `[2048,128] @ [128,50304]`, with every logit
materialized. FP32 parameters and gate certificates, FP16 operands/output, FP32
accumulation, TF32 and reduced-precision reduction disabled. Static weight casts
and packing, staging, loss, softmax and token selection are outside latency.

Complete final numerical qualification covers all500 MiniPile validation
documents,338 complete2048-token blocks,692224 input tokens and691886 predictions.
The1444-token tail remains excluded. Canonical trained loss is retained from
Run068; B1 qualification loss is separately reported. The unchanged acceptance
bounds are in config.json. Exactness is reported even when the declared bounds
permit small differences; no bounds are relaxed to accept a faster candidate.

## Implementations and diagnostics

- Native PyTorch/cuBLAS and the unchanged Run068 N256 custom kernel are controls.
- Legacy N128/N256/N512 isolates tile-size selection.
- Direct N256 changes the result-store phase; prefetch N128/N256 changes operand
  scheduling; reuse2 N128/N256 reuses weights across two independent16-row tiles.
  The reuse branch uses a union only to decide whether a weight group is needed;
  each tile's MMA decision remains independent. It never merges the trained gates.
- Every custom candidate runs with skipping disabled and enabled. Compiler
  resources, real-operand support and counted MMA paths, and graph replay checks
  are retained. The Count specialization is absent from timed graphs.
- An exploratory projection screen uses the same64 deterministic inputs and
  seven paired randomized passes as Run068. Up to two candidate implementations
  are selected separately for no-skip/sparse device latency. Failed numerical
  candidates are retained and excluded. Fresh final processes retain the current
  kernel and native control; three processes per checkpoint measure projection,
  gate+cast+projection and complete forward. These boundaries overlap.
- Nsight Compute profiles four fixed inputs per mode separately from latency.
  Retain registers, spills, occupancy, scheduler/stall, tensor-core and memory
  workload evidence, raw ncu-rep files and source/compiler records. Counter access
  can be restricted by the host; such a failure is explicit missing evidence.
- The zero-MMA synthetic path diagnoses fixed block/store work. Its runtime is
  **not** a lower bound for real inference and its zero outputs are not a quality
  evaluation. Do not subtract it to claim an exact compute/store decomposition.
- Retain exact real pre-gate inputs, token indices, head weights and checkpoint
  hashes for reproducible profiling. Source checkpoints, activation diagnostics,
  weight norms and training gradient geometry remain retained in Run068. No new
  gradient geometry or clipping sweep is collected because there is no training.

Support: reproducible reduction of the no-skip deficit against matched native
execution and numerical preservation, followed by measured sparse improvement.
Refutation/stop: bounded variants fail to reduce the deficit, profiling shows
little removable work, or gains disappear in independently measured full forward.
A kernel-local improvement does not establish a useful quality-latency frontier.
No combined optimized-body/head execution or larger model is inferred.

This operational diagnostic supports the distinction between logical opportunity
and measured runtime in the research contracts. No manuscript claim or TeX is
changed. Source-level possibilities remain hypotheses until measurements support
them; profiling cannot be replaced by optimistic explanations.

## Verification and launch

CPU mathematical tests, CUDA component checks (including opposite neighboring
masks, column tails, all-live/all-dead cases and full vocabulary), graph replay,
and a two-block trained-model smoke run precede launch. The laptop5070Ti is only
development evidence; final accepted timing requires one RTX5090 on the same
session for all modes. Full bootstrap tests run before the launch packet closes.

Proposed cloud envelope: one Secure RTX5090, pinned image/runtime,30GB container
disk and40GB persistent Pod workspace, no new network volume, flexible maximum
four GPU hours and USD6 total. Live availability/pricing and measured local fit
will be recorded in prelaunch/LAUNCH.md. The existing Run052 Pod and shared
network volume are outside ownership and remain untouched.

Monitor every60seconds and report stage, progress, latest qualification loss,
throughput and refreshed ETC. Warning conditions: nonfinite/numerical failures,
stale events beyond the declared stage timeout, disk exhaustion or budget/deadline.
All logs/artifacts live in persistent /workspace and the pipeline runs detached.
Transfer and hash-verify snapshots, raw timings, qualification, compiler/profiler
reports, manifests and receipts before deleting the owned Pod. A provider/local
deadline is a backstop. Explicit launch approval is still required for this packet.

Source: [Run068](../068-2026-10-08-pythia14m-aligned-vocabulary-tiles/results/REPORT.md).
No result-bearing figure is created before final evidence exists; any final PDFs
will have observation records and an observations/INDEX.md in this run.
