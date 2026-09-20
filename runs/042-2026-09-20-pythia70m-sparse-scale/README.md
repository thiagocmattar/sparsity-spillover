# Run042: 70M sparse execution and native-base scale comparison

The user approved a new launch and iterative optimization on 20 September:
up to four hours and USD15, with an explicit full-forward native dense base
reference and decomposition of the gains. This continues the previously
discussed h/z layout, inspection reuse and efficient fallback investigation;
both design and launch are authorized. No additional training is performed.

## Question and fixed comparison

Can a numerically qualified 70M implementation exceed the fractional latency
reduction of the retained 14M kernel, with each size referenced to native
T0/P0? Historical best 14M speedup is 1.426501x (29.90% latency reduction).
Re-measure 14M T0/P0 and best T4/Pall kappa0.5 on the same physical GPU; also
retain matching h-only recipe controls. The primary optimization checkpoint
is 70M T7/Ph kappa0.5, with T0/P0 dense control. Report absolute latency and
same-checkpoint-native comparisons separately. Winning the scale comparison
requires a larger native-base percentage reduction, not merely more saved ms.

Weights, gates, training provenance, thresholds, initialization/data seed1234,
final step712 and ordinary checkpoint losses remain fixed. T7 uses G+ at
a,m,h,z and Gpm at post-RoPE q,k,v; Ph is h-only OL1 with lambda=b=1. The
14M retained pressure controls retain their original matched training settings.
No optimizer/backward pass, quantization, threshold alteration, token removal,
input-dependent result caching or reduced output vocabulary is allowed.

RTX5090, BF16, batch1, T2048, causal full forward, all50304 logits. Shared
native SDPA graph scaffold and unmodified eager correctness anchor. TF32 and
BF16 reduced-precision reductions disabled. Qualification covers all338
complete blocks from all500 MiniPile validation documents: 692224 input,
691886 prediction tokens;1444 excluded tail tokens. Original bounds remain
atol0.25 + rtol0.02 per logit, relativeL2<=0.02, loss difference<=0.001.
Timing:64 original inputs x7 passes x3 fresh processes; timing seed2504,
runtime seed2801. Report synchronized host time, retain CUDA-event time.
All recurring inspection, packing, branching and full-logit work is timed.

## Search and interpretation

Start from Run040's qualified native a/m + native attention + sparse N256 h/z.
Use fixed first16 retained training blocks for candidate development. Keep
immutable candidate sources and record every failure. Explore row/output tile
layout, sharing sparse inspection, short-row execution and dense fallback;
profile other components and test evidence-motivated changes if needed.
At most32 named candidates, bounded by wall time; one policy per architecture
with no checkpoint/input-ID special cases. Freeze one candidate using training
data before final validation. Do not retune using final results.

Decompose with matched same-checkpoint controls: frozen predecessor, native
h/z replacement, sparse h/z disabled, native a/m/attention/norm/RoPE where
applicable, and component profiles. Sparse-toggle effects and dense-operation
improvements are separate; conditional effects are nonadditive. A native-base
gain from an unrelated dense optimization is not attributed to sparsity.
The target is evidence for this implementation and retained recipes, not a
universal scale law or equal-quality comparison between trained checkpoints.

## Retention and launch envelope

Retain checkpoints/cache identities, all timing samples, per-block correctness
and loss, exact/near-zero and RMS/L2 summaries, weight norms, row/tile/MMA
counters with explicit definitions, profiler traces, compiler resource data,
candidate sources and failures. Training-time gradient interaction cannot be
reconstructed here; original records remain retained. These extend the
previously approved diagnostic inventory. No manuscript edits or PDF rebuild.

One Secure RTX5090, pinned Run040 image/runtime,80GB Pod volume plus20GB disk.
Live catalog20 September: USD0.99/hour, low availability EU-RO-1. Four GPU
hours cost USD3.96 plus storage, within the USD15 total ceiling. Hard deadline
is the earlier of four hours from this goal's start or four Pod hours;
reserve the last30 minutes for final checks/recovery. Stop before exceeding
either cap. Existing Run041 and the shared network volume are unrelated.
Use detached persistent processes, a local Pod stop guard and credential-free
remote command deadlines. No provider credential is uploaded. Monitor at
60-second intervals and refresh ETC/spend. Numerical failure, stale worker,
memory below8GiB or disk below10GB trigger investigation. Transfer archives
and verify every hash before deleting the owned Pod. The owned Pod is
`nujok8uu8is06b`, created at13:56:18UTC on20 September; exact placement and
price are retained in prelaunch/lease-001.json.

## Initial development candidates and transfer retry

The first six candidates change h/z row groups (8/16) and output columns
(128/256/512). Candidates007--011 test dense attention scheduling separately:
query/key tiles64/64,64/128,128/64,64/256 and128/128 with eight warps.
Candidates012--014 inspect h/z once per row and reuse fresh summaries across
output-column blocks. All inspection launches and temporary reads/writes are
timed; no activation result persists between inputs. Candidate definitions are
immutable after registration. Attention changes must pass causal/output tests
and the original end-to-end numerical bounds; altered softmax reduction order
is a qualification risk, not grounds to loosen tolerances.

Candidates015--018 extend the scalar row limit from2 to4/8/16 (M8), or8 (M16).
The prior Run040 operand histogram gives99.97761% h rows and99.99981% z rows
with at most8 nonzeros, motivating this specific test. Larger scalar sums can
differ from MMA reduction order; the new operator checker uses the original
native bounds and exact independent instruction/scalar-work counts, while
recording bitwise equality separately. Full-model bounds are unchanged. These
are new implementations, not a relaxation of the original variants' checks.
Metadata/specification fields for each candidate supersede inherited starting-
kernel descriptions. Candidate selection still uses only development inputs.

The first archive upload and a native SCP probe were below0.1MB/s. They were
replaced by32 resumable SSH streams to the same owned Pod; all original archive
bytes and SHA256 remain unchanged. Software installation ran concurrently.
The safer deadline mechanism and approval-review decision are recorded in
prelaunch/guard-decision.md. Infrastructure overlays are inventoried before the
scientific pipeline begins.


## Follow-up development and attribution

Candidates019--021 make neighboring lanes read neighboring output-column
weights, with scalar limits2/4/8. Candidates022--024 test smaller query tiles
and reversed causal-block scheduling. Candidates025--028 change only the dense
full-vocabulary GEMM schedule; the unchanged sparse h/z implementation remains
separately identifiable. The final decomposition includes restoring the native
output projection, and fresh14M native-h/z replacements at both dense and sparse
checkpoints. These conditional controls must not be added as independent gains.

The extended-row checker added a large-cancellation synthetic case absent from
the original48-case suite. At its first failure, the new and retained frozen
kernels were bitwise identical while both differed from the native operator.
The original failures are preserved. `candidates/audit_short.py` independently
runs all80 cases, including the original candidate, to distinguish inherited
reference rounding from a new numerical regression. This audit does not alter
qualification or any final-model tolerance.

Each overlay is hashed and verified on the Pod; sequential pipelines prevent
simultaneous GPU timing. Updated harness files retain their per-attempt hashes.
Compilation and failed candidates count against the original time/cost budget.

## Author-requested continuation after the initial sweep

At 15:30 UTC the author instructed: "Keep going until you achieve the goal or
ends the authorized deadline." The initial 32-candidate sweep had completed.
This continuation extends the search with new immutable candidates while
preserving the original 17:47:59 UTC deadline, USD15 envelope, training-only
selection, one frozen final candidate, and every scientific/numerical constraint.
The original config records the initial sweep limit; it is not rewritten.

The training-only profiler `29_profile_development.py` attributes about0.441ms
to the output head,0.324ms to attention and0.145ms to fused h/z in opt032.
These instrumented durations diagnose costs and are not benchmark latency.
Candidates033--040 test eight dense full-vocabulary Triton schedules on top of
the same opt032 sparse components. Dense-head gains remain separate from sparse
skipping in the final attribution. No final-validation inputs are used to tune.

Candidates041--063 test parallel row inspection, longer exact short-row paths,
dense projection/head schedules, and alternative native attention backends.
The native attention alternatives failed full-model development bounds and are
excluded. Candidates041--052 originally counted useful scalar work but omitted
duplicate prepass work in mixed fallback groups; their timing still includes
that work. Candidate063 corrects the actual-work counters, with an independent
oracle and focused tests. Original sources and results remain unchanged.

Candidates064--074 investigate token-major attention output, joint h/z
inspection, and attention launch ordering. The K16 head candidates065--069
are unsupported by the installed CUTLASS pipeline and retain their compilation
failures. Candidate073 schedules the longest causal query blocks across all
heads before shorter blocks; this changes dense scheduling only. Candidate074
tests a warp vote for the existing fallback decision. All selection remains on
the fixed first16 training blocks, with unchanged weights, gates and tolerances.

`42_profile_attribution.py` additionally identifies the parallel h/z prepass in
the profiler, keeping it separate from the fallback kernel. The final evidence
checker `43_verify_final_evidence.py` requires all planned fresh processes,
complete validation and timing coverage, one physical GPU, full logits,
diagnostics and nonadditive replacement controls before reporting completion.
