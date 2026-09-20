# O001: 70M native-base latency reduction after kernel optimization

## Question

Can a numerically qualified implementation for the retained 70M sparse model
exceed the fractional native-base latency reduction of the retained 14M kernel?
Each size uses its own native T0/P0 full-forward latency on the same physical
RTX5090. This is an implementation-specific comparison, not a scaling law.

## Method and coverage

Run042 uses BF16 operands/output with FP32 accumulation, batch 1, sequence 2048,
and every 50304 output logit. Weights, thresholds and trained checkpoints remain
fixed. No token removal, quantization or input-dependent result cache is used.
Every recurring inspection and packing operation is timed. Static weight-layout
preparation is outside steady-state timing for both declared implementations.

Candidate selection uses only the first 16 retained training blocks. Candidate
opt073 was frozen before final validation. Each final condition uses three fresh
processes, 64 timing inputs and seven passes: 1344 synchronized host observations
per implementation, plus retained CUDA-event measurements. Equal input staging
is excluded. Numerical qualification covers all 338 complete validation blocks
from 500 MiniPile documents, with 1444 trailing tokens excluded. Loss pools 691886
prediction tokens. Per-logit bounds remain 0.25 + 0.02 * abs(reference), relative
L2<=0.02, and validation-loss difference<=0.001 against native eager evaluation.

## Selected implementation

The h/z path uses a 128-thread row prepass, ascending-index scalar execution for
at most eight safe nonzero entries, and the retained M8/N256 MMA fallback.
Counters include duplicated scalar execution in mixed fallback groups. Attention
uses dense 64 x 128 tiles with four warps, token-major output, and long causal query
blocks scheduled across all heads first. The full-vocabulary head uses a dense
CUTLASS 128 x 128 x 32 schedule. Attention and head scheduling gains are distinct
from activation-zero skipping; native a/m projections remain in use.

## Qualified latency result

All nine final processes pass. The median of the three process medians is
1.136311 ms for 70M T7/Ph and 1.137580 ms for T4/Ph, compared with 1.697655 ms for the
native 70M T0/P0 base. The primary T7/Ph result is 1.494005x, or 33.065821% latency
reduction. The three primary medians span 1.135543--1.137198 ms. Losses are
5.346554325 for T7/Ph and 5.315318208 for T4/Ph; the base-model loss is 4.107689843.
These checkpoints are not quality matched. Kernel qualification compares each
checkpoint against its own native evaluation, not against the base-model loss.

The selected custom path takes 2.881526 ms on T0/P0. This inefficient dense
fallback is not the speedup denominator. Native-base latency is the explicit
reference above; same-checkpoint-native and skip-disabled controls are separate.

## Fresh 14M comparison

The native 14M base takes 0.682068 ms. The retained kernel takes 0.478992 ms on the
corrected T4/Pall control and 0.478309 ms on the historical realized T4/Ph control.
The best of these is 1.426000x, or 29.873765% latency reduction. The primary 70M
result exceeds both controls and the historical 1.426501x target. Even the
conservative 70M ratio formed from the slowest candidate process and fastest
native-base process exceeds the optimistic ratio for either 14M control.
This range check is descriptive, not a confidence interval.

## Conditional attribution

Each row restores one component while retaining all other selected components
on the same 70M T7/Ph checkpoint. Values are medians across three fresh processes.
The selected implementation takes 1.136311 ms. Effects must not be added.

| Replacement or toggle | Full-forward ms | Change from selected, ms |
|---|---:|---:|
| Disable custom h/z skipping | 2.698700 | +1.562388 |
| Native h/z projections, gates and residual additions | 1.361872 | +0.225561 |
| Native attention and its output layout | 1.270237 | +0.133926 |
| Native normalization and input-gate handling | 1.225275 | +0.088964 |
| Native RoPE, QKV layout and gate handling | 1.460302 | +0.323990 |
| Native full-vocabulary head | 1.201117 | +0.064806 |
| Native a/m projections (already used) | 1.135120 | -0.001192 |

The skip-disabled custom fallback is inefficient, so its 1.562 ms penalty cannot
be reported as a speedup against native execution. The native h/z replacement
provides a practical matched control: the combined sparse h/z implementation
reduces latency by 0.225561 ms, including its fusion/layout/inspection behavior.
This is not an isolated instruction-skipping effect. The a/m difference is
small process variation rather than a new implementation gain.

Dense optimizations matter substantially. The T0/P0 checkpoint takes 1.293405 ms
when native h/z is combined with the other selected components, at unchanged
loss 4.107690. T7/Ph at 1.136311 ms is about 12.15% faster than this alternate dense
implementation, with a different trained checkpoint and higher loss 5.346554.
The declared native-base reduction of 33.07% remains the primary comparison;
it must not be described as entirely caused by activation sparsity.

For 14M T4/Pall, native h/z replacement takes 0.556260 ms, compared with 0.478992 ms
for the retained kernel. Disabling h/z skipping takes 0.655975 ms. Thus useful
h/z execution structure occurs at both sizes, while the implementation and
other dense-operation gains affect the whole-model comparison.

## Structural diagnostics

Across 4,153,344 rows per site, 99.977608% of h rows and 99.999807% of z rows in
70M T7/Ph have at most eight nonzeros. Actual work counters retain 794368 issued
h MMA instructions, 4160 issued z MMA instructions, 953377792 h scalar products
and 348839424 z scalar products over the full 338-block coverage. Scalar totals
include duplicate prepass execution in mixed fallback groups. MMA instructions
and scalar products have different units; do not add their raw counts.

The separate instrumented profile attributes about 0.440 ms to the dense head,
0.223 ms to attention, 0.266 ms to a/m projections, and 0.096 ms to combined h/z
inspection, scalar execution and fallback. These are summed GPU kernel times
from a separate four-input profiling pass, not the reported host latency.

## Verification and closeout

The final evidence checker verifies 78 complete processes: 18 baseline, 9 final
qualification and 51 component controls. Every process has all 338 validation
blocks, 64 x 7 timing observations per implementation, full 50304-logit outputs,
the original numerical bounds and the same physical GPU UUID. All three final
conditions also retain full-coverage activation, weight and work diagnostics.

The 12,841,862-byte archive has SHA256
`1c9b6991ff637bdcb76a37d3bffd592966ad6584aa320d7297c0a2f532cbea7c`.
All 2584 transferred members were verified locally, as were 1435 retained
checkpoint/cache/source copies. Both the complete evidence audit and profile
attribution were reproduced locally without rewriting their results.

The owned Pod `nujok8uu8is06b` was deleted before 2026-09-20T16:35:01Z, ahead
of the 17:47:59Z deadline. The live Pod list confirms its absence and preserves
the unrelated Run043/Run044 resources. The local deadline guard was disarmed
after deletion. Recorded-rate GPU cost is at most approximately USD 2.62 plus
temporary disk, within the USD 15 envelope. The retrieved provider billing
snapshot is partial and is not represented as a settled final invoice.

## Caveats

This is one GPU, one workload and retained one-seed/one-pass checkpoints. Timing
ranges describe three process medians, not confidence intervals. Component
replacement effects are conditional and nonadditive. Logical opportunity,
issued MMA instructions, scalar products and runtime are different quantities.
Native instruction counts are unmeasured. The historical 14M h-only control is
the documented Run012 realized h-only objective, despite its original declared
four-site metadata; the corrected all-site control comes from Run015.

The supplemental 80-case h/z audit preserves eight extreme-cancellation failures
that also occur bitwise identically in the frozen predecessor. All 72 other
cases pass native operator bounds and all work counters match the independent
oracle. This inherited stress limitation does not relax the full-model bounds;
all 338 validation blocks pass those original bounds in every final process.

## Sources

- Frozen selection: `../provenance/final-selection.json`.
- Candidate: `../candidates/opt073/manifest.json` and its hashed parent manifests.
- Full comparison: `../20_summarize.py`, producing `../results/summary.json`.
- Profile attribution: `../42_profile_attribution.py`.
- Complete evidence audit: `../43_verify_final_evidence.py`.
- Verified results: `../results/final-verification.json` and `../results/profile-attribution-v2.json`.
- Recovery inventory: `../transfer/inventory-final001.json`.
- Local reproduction and teardown: `../prelaunch/local-final-verification.json` and `../prelaunch/teardown-final.json`.
- Input/checkpoint identities: `../provenance/inputs.json` and `../provenance/reuse.json`.
- Canonical legacy-control clarification: `../../../research/MANUSCRIPT.md`.

No manuscript changes or PDF compilation are part of this run.
