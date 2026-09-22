# Run051: frozen Run050 kernel across the requested 70M grid

Status: full sweep running on the retained RTX5090 after successful smoke checks.

The user requested all kappa values for T2/Ph and T7/Pall, plus PyTorch Base
and optimized dense Base. This extends the already authorized evaluation;
the existing four-hour GPU extension remains the hard budget. No new kernel
selection, training, checkpoint changes, or manuscript edits are introduced.

## Fixed design

Eleven existing step712 checkpoints: Base c00; T2/Ph c22,c23,c24,c25,c26 and
T7/Pall c07,c08,c09,c10,c11, each at kappa0,.01,.05,.1,.5. Sources are the
verified Run045 catalog and Run047's later T2/Ph kappa.5 endpoint. Preserve
random initialization/seed1234, training data/order, optimizer and OL1 histories,
all original gates and weights. T2 means operational HZ, not historical A2.
The complete original training configurations are retained per checkpoint.
No new optimization or gradient diagnostics can be reconstructed by inference.

Run050's selection-final.json is byte-identical. Its primary policy uses a
compacted active-feature union and gathered tensor-core GEMM at h in layers1--5,
dense fused h at layer0, and dense gated z at every layer. The same policy is
applied to every checkpoint. T7's other ports retain the opt073 scaffold.
PyTorch, efficient native_hz, previous opt073, frozen dense_policy and sparse_c
are measured together; no per-checkpoint best-kernel selection occurs.

BF16, FP32 accumulation, B1,T2048, full50304 logits, CUDA graphs including
recurring scans/packing/gates, TF32/reduced-precision reduction disabled.
Three fresh processes per checkpoint; fixed64 validation inputs,7 randomized
paired passes (1344 observations per implementation/checkpoint). Numerical
qualification covers all500 MiniPile documents/338 complete blocks,692224 input
tokens,691886 prediction tokens;1444-token tail excluded. Bounds remain
atol.25+rtol.02 per logit, relativeL2<=.02 and pooled loss delta<=.001.
Smoke uses four fixed validation blocks solely to verify the frozen implementation;
there is no tuning. Failures and slowdowns remain visible in the table.

The question is whether the already frozen policy generalizes across these
thresholds/topologies. Lower qualified latency supports that scoped statement;
a slowdown or numerical failure refutes it for that cell. This does not test
whether retuning could do better. Loss is the matched BF16 inference loss;
canonical historical FP16 loss remains separately identified in provenance.
The manuscript's latency/quality claim is the motivation; no TeX is changed.

## Inventory and execution

Retain checkpoint/cache hashes and original provenance, full-validation
exact/near-zero counts,RMS/L2, weight norms, row/group occupancy, logical and
h/z work counters, compiler metadata/PTX, profiles, raw timing pairs, numerical
checks and failures. Full diagnostics collect dense_policy and sparse_c once
per checkpoint (Base has its equivalent dense fallback). Training gradient
interaction remains in the source runs. Hardware traffic counters remain
unavailable due to the previously observed ERR_NVGPUCTRPERM; do not claim them.

Use existing Pod kym4fmsrbsg1s6 at the live $0.99/hour price, unchanged30GB
container/40GB persistent disk and runtime. Eight additional immutable weights
need about2.25GB transfer; three are already present. Estimated execution and
verified retrieval45--75minutes after transfer; no new resources. Absolute
provider-stop deadline22:42:28UTC on22September2026; workload cutoff22:22:28UTC
reserves20minutes for retrieval. The independent local stop guard remains active.
Retain the Pod running after completion per the user's explicit instruction.

One detached sequential GPU pipeline with persistent logs. Read-only monitoring
at one-minute intervals; report completed processes, current validation loss,
throughput and refreshed ETC. Warn on failed qualification, less than8GiB GPU
headroom, less than5GB container space, stale progress for10minutes, or a
projected deadline overrun. Copy and SHA256-verify all artifacts locally.

## Verification

All245 local checks passed (242 bootstrap +3 grid/identity checks) in9.23seconds.
All run-local Python files compile. Seven kernel/diagnostic files and the complete
selection are byte-identical to Run050. GPU smoke completed in246.3seconds:
four checkpoints (both topology/density extremes), five graph implementations
each, all four-block numerical qualifications passing, plus T7 diagnostic,
work-counter, compiler and profile smoke for both dense_policy and sparse_c.
The98-file upload inventory (3,099,420,059bytes including reused checkpoints)
was verified by SHA256 before the smoke. Full sequential execution launched
at20:57UTC under the unchanged absolute deadline.

Transfer retry: Paramiko SFTP sustained roughly23KB/s to both persistent and
local disk. Both partial uploads were retained and stopped; the same verified
archive is now sent through native OpenSSH stdin, whose4MiB probe completed
in4.75seconds including connection setup. This is an infrastructure retry;
scientific inputs are unchanged.
