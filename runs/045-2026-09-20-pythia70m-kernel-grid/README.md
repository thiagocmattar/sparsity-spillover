# Run045: matched 70M kernel comparison across 26 trained checkpoints

Status: design approved; implementation and local checks complete; launch pending.
No GPU launch or new cloud spend is authorized yet.

The user confirmed the 26-checkpoint, three-implementation comparison on
20 September 2026. The question is whether Run042's frozen opt073 improvement
extends across the retained 70M recipes, and how much additional optimization
improves on the original 14M-derived port. Positive, qualified paired latency
reductions support implementation-specific transfer. Slowdowns or numerical
failures limit that claim and remain in the record. This does not test a scaling
law, quality-matched model selection, or equal kernel-search budgets.

## Matched scientific contract

Reuse all 22 final step712 checkpoints in Run035 and four verified Run043 HZ
checkpoints: Base, ReLU, T4/T7 with Ph/Pall at kappa0/.01/.05/.1/.5, and T2/Ph
with one-sided h/z gates at kappa0/.01/.05/.1. The T2 name is manuscript notation;
its operational topology is HZ, not the historical A2=m,h registry entry.
Weights, architecture, thresholds and pressure histories remain unchanged.
The original random initialization, seed1234, MiniPile order, AdamW/OL1 recipe,
712 updates and 1.493B-token training budgets are retained as source provenance.
No training, optimizer steps, clipping sweep, or new kernel search occurs.

Each fresh process compares native PyTorch/SDPA, the frozen Run035 v2 port, and
Run042 opt073 on one physical RTX5090, sequentially with randomized paired
timing order. The only compatibility extension is registering HZ and admitting
it to the existing gate-aware adapter. All CUDA and candidate arithmetic remains
byte-identical to the frozen sources. Every topology must qualify independently.

BF16 operands/output, FP32 accumulation, batch1, sequence2048, causal full
50304-logit forward; TF32 and BF16 reduced-precision reduction disabled.
CUDA graphs include all recurring gates, inspection, packing and output logits.
Compilation, static layout preparation, capture and equal input staging are
excluded. Runtime seed2801 and timing seed2504 are fixed.

All500 MiniPile validation documents yield338 complete blocks,692224 input
tokens and691886 prediction tokens; the1444-token tail is excluded. Qualification
compares each implementation to the same checkpoint's unmodified eager output:
every logit atol.25+rtol.02, per-sequence relativeL2<=.02, pooled loss delta<=.001.
Three fresh processes per checkpoint use64 fixed inputs x7 paired passes:
78 final processes,1344 observations per implementation/checkpoint.
The primary latency is the geometric mean of all1344 synchronized host timings.
Process geometric-mean ranges and medians are retained as descriptive summaries.
All recipe speedups use the same native T0/P0 denominator; same-checkpoint and
original-port paired ratios are reported separately. No per-recipe best-backend
selection is introduced. The custom dense fallback is never a speedup reference.

## Diagnostics and retention

Retain raw host/CUDA timings, per-block numerical gates and pooled losses,
runtime/GPU/source identities, failures, and all final checkpoints locally.
On replicate1 collect full338-block exact/near-zero counts (0/.001/.01), RMS/L2,
weight norms, occupancy histograms, actual h/z MMA/scalar counters, and original
port attention counters for both specialized implementations. Reuse canonical
quality and pooled logical sparsity from the original full-validation records;
BF16 operand diagnostics are a separate estimand. Preserve analytic ceilings
with integer counts and units. Historical training gradient/OL1 diagnostics
remain in source runs; they cannot be reconstructed by inference. No post-hoc
clipping is applied. The user confirmed this diagnostic inventory is sufficient.

## Manuscript integration after the completed sweep

Keep14M as the main result and70M as the scale check. Replace the70M latency
panel, recalculate paired-pressure latency effects and all dependent counts,
update compact endpoint tables and cohort counts, and revise the introduction,
abstract, scale-check paragraph and conclusion surgically. Replace the older
two-checkpoint kernel appendix with a compact before/after comparison and the
retained Run042 dense/component controls. Do not combine old instruction counts
with new-kernel timings. Source observations and analysis generators accompany
each result-bearing manuscript change. Inspect a temporary PDF before replacing
main.pdf. Preserve unrelated author edits.

## Execution and recovery proposal

One Secure RTX5090; separate launch approval is required after local checks,
current quote and ETC are recorded. Persistent /workspace, detached controller,
unique attempts, deadline reserve for collection/transfer, and a provider stop
backstop. Monitor at60s; report completed processes, latest validation loss,
evaluation throughput and refreshed ETC. Warn on runtime/source drift, numerical
failure, nonfinite output, low disk/headroom, stale progress or deadline risk.
Return raw artifacts, diagnostics, source/runtime identities and inventories.
Verify every returned size/SHA256 before Pod deletion, then confirm removal.
Existing unrelated Pods and the shared network volume are outside this run.

## Prepared launch and local checks

The [launch proposal](prelaunch/launch-proposal.md) records the exact workload,
current quote and recovery plan. Proposed: one Secure RTX5090 at USD0.99/hour,
60-100min estimated ETC, maximum3 cumulative GPU-hours and USD5 incremental
cost including temporary storage. The quote has LOW stock; recheck at launch.
No Pod was created. The local12GB RTX5070Ti can load the model but cannot
provide the approved RTX5090 latency comparison.

Six focused tests and all242 bootstrap tests pass. Four real CPU checkpoint
smokes pass for Base, T7/Ph kappa.5 and T2/Ph kappa0/.1: native outputs are finite,
both frozen implementations install and the declared gates remain intact.
CUDA kernels and full338-block numerical qualification still require the
approved remote preflight/final execution. All18 root Python files parse;
both shell launch scripts pass Bash syntax checks. No manuscript edits or
compilation have occurred during preparation.

The local input archive is6,788,920,505bytes, SHA256
`b3f581e5dcc8c0aeee264a666ac2b9902e135d7ab5b409dc8bd0879825ea827b`.
Its inventory and85-file scientific source freeze are linked in
`prelaunch/input-bundle.json`. Checkpoints and binary archives are excluded
from Git. Its README is the initial preparation snapshot; this section and
the launch proposal record the subsequent completed checks without changing
the packaged scientific sources. The staged retained-source bytes were checked
against their on-disk hashes before commit.
