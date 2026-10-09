# Run069: vocabulary projection implementation efficiency

Status: design approved by the user's **Go on** on 2026-10-09. Implementation
and local pre-launch verification are complete. Launch approval is pending.
No cloud launch or accepted RTX 5090 speedup result exists for this run.

## Question and matched scope

Can we locate and reduce the Run068 custom head's projection penalty against
native PyTorch/cuBLAS without changing any trained gate, operand, or output?
This is an implementation diagnostic using existing checkpoints.

Reuse both Run068 step-712 Pythia-14M checkpoints, preserving their random
initialization, training seed 1234, data, optimizer, schedule and pressure history.
There are zero new optimizer updates. Both have h/z corner 8x16 gates at kappa 0
and Ph+head OL1 pressure. Their head-input corner 16x16 gates, after affine final
LayerNorm, have kappa 0.1 or 0.5. A tile survives iff its FP32 signed upper-left
value is strictly greater than kappa; live interiors retain their signed values.
Only head execution varies. The body uses native dense execution throughout.

The workload is uncached B1/T2048, `[2048,128] @ [128,50304]`, materializing every
logit. FP32 parameters and gate certificates, FP16 operands/output and FP32
accumulation remain matched; TF32 and reduced-precision reduction are disabled.
Static weight casts/packing, staging, loss, softmax and token selection are outside
latency measurements. Runtime seed is 2801; timing-input seed is 2504.

Final qualification covers all 500 MiniPile validation documents: 338 complete
2,048-token blocks, 692,224 input tokens and 691,886 predictions. The 1,444-token
tail is excluded. Run068 canonical trained loss is retained; B1 qualification loss
is reported separately. Unchanged acceptance bounds are in [config.json](config.json).
Exactness is reported even when those bounds allow small differences.

## Implementations and diagnostics

- Native PyTorch/cuBLAS and the unchanged Run068 N256 kernel remain controls.
- Legacy N128/N256/N512 varies vocabulary tile width. Direct N256 changes result
  stores; prefetch N128/N256 changes operand scheduling; reuse2 N128/N256 reuses
  weights across two independent 16-row tiles. A union mask decides whether to
  load weights, while each row tile's MMA decision stays independent.
- Every custom candidate runs with skipping disabled and enabled. Untimed Count
  specializations retain actual issued/skipped MMA counts on real timing operands;
  counters are absent from timed graphs. Compiler register/spill logs are retained.
- The exploratory screen uses 64 deterministic real inputs and seven randomized
  paired timing passes, matching Run068. At most two implementations are selected
  separately for no-skip and sparse device latency. Failed numerical candidates
  are retained and excluded. Final measurements use three fresh processes per
  checkpoint, retaining the native/current controls and selected candidates.
- Final timing boundaries are projection only, gate+cast+projection, and complete
  forward. They overlap and must not be summed. Host and CUDA-event timing are
  retained separately. Hardware UUIDs and sealed source identities are reconciled.
- Nsight Compute profiles four fixed inputs per mode separately from latency.
  Retain occupancy, scheduling/stalls, tensor-core and memory evidence, profiler
  reports and compiler records. Counter denial remains explicit missing evidence.
- A synthetic zero-MMA path diagnoses fixed block/store work. It is **not** a
  runtime lower bound and its zero outputs are not a quality evaluation. Do not
  subtract its latency to claim an exact compute/store decomposition.
- Retain exact pre-gate inputs, token indices, head weights and checkpoint hashes.
  Run068 retains the existing checkpoints, activation/weight diagnostics and
  training gradient geometry. No new training diagnostics or clipping sweep occur.

Support for the hypothesis requires a reproducible reduction of the no-skip
deficit against matched native execution, numerical preservation, and a measured
sparse improvement. Refuting evidence includes bounded candidates failing to
reduce the deficit, little removable work in profiling, or gains disappearing in
full forward. Kernel-local gains do not establish a useful quality-latency frontier.
The strongest optimized body is not integrated here; larger models are not tested.

This diagnostic supports the operational distinction between logical opportunity
and measured runtime. No manuscript claim or TeX changes. Source-level possibilities
remain hypotheses until measurements support them.

## Verification and execution

242 bootstrap tests and 7 focused mathematical/artifact tests pass. Local WSL
attempt 002 passes 160 CUDA component/replay checks and two-block full-model smokes
for both checkpoints; all smoke logits and losses match native exactly. Column
tails, full vocabulary, random/live/dead and opposite neighboring masks are covered.
Screen peak reserved memory is 4.171875 GiB; endpoint peak is 3.041016 GiB.

Local Nsight Compute 2025.1.1 denies counters with `ERR_NVGPUCTRPERM`. The failure
is retained; it does not establish the hardware bottleneck. Attempts 001/002 use
only two timing inputs and two passes on the RTX 5070 Ti laptop, so neither is
accepted RTX 5090 latency evidence. Attempt 002 adds real-operand Count records
and endpoint GPU UUIDs. Subsequent source changes strengthen identity recording
and launch/verification handling; CUDA arithmetic and gate semantics are unchanged.

The approved retention inventory is implemented. Proposed execution uses one
Secure RTX 5090, a pinned runtime, 30 GB container disk and 40 GB persistent Pod
workspace, with a flexible four-hour backstop and a proposed USD 6 service budget.
Monitor every 60 seconds, refreshing progress, latest qualification loss, throughput
and ETC. Persistent logs and a detached pipeline survive terminal disconnection.
The workstation guard stops provider compute at the deadline while retaining
unverified artifacts; it depends on workstation uptime/API access, and stopped
storage remains billable until recovery/deletion. Retrieve and verify all agreed
files before deleting the owned Pod. Existing foreign resources remain untouched.

See [launch packet](prelaunch/LAUNCH.md), [checks](prelaunch/checks.json), and
[Run068 results](../068-2026-10-08-pythia14m-aligned-vocabulary-tiles/results/REPORT.md).
No result-bearing figure is created before final evidence exists. Final PDFs will
have observation records and an `observations/INDEX.md` in this run.

## Execution closeout: 2026-10-09

The preceding sections and launch packet record the sealed pre-launch state.
The user subsequently approved launch. Run069 is now executed and locally verified;
Pod d9ryfnu2zhri2w was terminated after all agreed artifacts were copied and hashed.
No newly created billable resource remains. The deadline guard is cancelled.

All 160 cloud CUDA checks and all 14,196 full-model mode/block comparisons pass;
the latter are bitwise exact. Six fresh processes cover all 338 validation blocks.
The best no-skip projection gap shrinks from about 65% to 27-28% versus native.
Best sparse full-forward gains are 1.86% and 3.05% against the matched gated native
control. Existing head quality penalties remain; no practical global frontier win
is established. Nsight Compute denied counters, leaving hardware-cycle attribution
unresolved. Provisional compute plus storage is about USD 0.94, not a posted invoice.

See [diagnostic](results/DIAGNOSTIC.md), [all final measurements](results/REPORT.md),
[observations](observations/INDEX.md), and [cloud closeout](prelaunch/cloud-001/closeout.json).
The source protocol remains commit 9937f806; this appendix is post-run documentation.
Its original sealed README is retained at provenance/frozen-README.md and in that
commit/input archive. Replaying strict source verification requires the sealed
checkout, rather than this annotated README. Scientific runtime/configs are unchanged.
