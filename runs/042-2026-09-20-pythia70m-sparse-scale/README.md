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
Use detached persistent processes and remote/local stop guards. Monitor at
60-second intervals and refresh ETC/spend. Numerical failure, stale worker,
memory below8GiB or disk below10GB trigger investigation. Transfer archives
and verify every hash before deleting the owned Pod. No paid resource exists
for this run yet; the launch packet will record checks and actual placement.
