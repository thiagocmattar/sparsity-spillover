# Frozen Run050 kernel across T2/Ph and T7/Pall

## Question and method

How does the same frozen Run050 policy perform across all retained thresholds
of the two requested 70M families, relative to PyTorch Base, optimized dense
Base, the same checkpoint's dense control, and previous opt073?

Reuse Base and the ten existing step712 checkpoints at kappa0,.01,.05,.1,.5.
There is no training, threshold change, extra pruning, or per-checkpoint kernel
selection. The Run050 policy and seven kernel/diagnostic source files are
byte-identical. Five implementations run in each of three fresh processes per
checkpoint on the retained physical RTX5090. BF16, batch1, sequence2048,
full50304 logits and CUDA graphs remain matched. Source manifests retain the
original random initialization, seed1234, optimizer/pressure and data histories.

## Coverage and table caption

Every numerical evaluation covers all500 MiniPile validation documents as338
complete blocks,692224 input tokens and691886 prediction tokens, excluding the
1444-token tail. Each latency pools1344 synchronized host observations:64 fixed
validation inputs x7 randomized paired passes x3 fresh processes. Compilation,
capture and equal input staging are excluded; recurring gates, scans, compacting,
gathers, projections and complete logits are included.

[Complete table](../results/complete-table.md),
[machine-readable table](../results/complete-table.json), and
[full reduction with raw-source hashes](../results/final-summary.json).
Loss in the displayed table is matched native BF16 loss. Historical canonical
FP16 loss is separately retained in the JSON and is not substituted into these
numerical checks. Speedup means reference latency divided by candidate latency.
The two Base denominators are measured in this session; previous Run050/Run045
timings are not pooled or rescaled into this sweep.

## Result

PyTorch Base averages1.647401ms and optimized dense Base1.257799ms.
All five T2/Ph checkpoints qualify in all three repeats. At kappa.05/.1 the
new kernel takes1.183922/1.171148ms, or1.391477x/1.406655x PyTorch Base and
1.062400x/1.073988x optimized dense Base. The gains over the same checkpoint's
strongest qualified dense control are5.55%/6.51% lower latency. Crossed
process/input bootstrap95% speedup intervals are[1.057830,1.059497] and
[1.068898,1.070225], respectively. These intervals concern these inputs and
three processes, not training seeds or the GPU population.

The policy does not generalize uniformly to T7/Pall. Kappa.01/.05 qualify, but
take1.387667/1.345014ms:10.77%/7.46% more time than their own dense controls.
They beat PyTorch Base but do not beat optimized dense Base. At kappa.5,
1.147831ms is qualified and beats both Base references. Thus a gain against
PyTorch Base alone does not establish a benefit from sparse execution.

Full-validation h exact-zero fractions at kappa.05/.1 are98.3228%/98.9187%
for T2/Ph versus89.4733%/92.2609% for T7/Pall. The latter are actual operands
of the frozen execution, including the unqualified.1 row. Per-layer group
occupancy and work counters are retained; global zero fractions alone do not
identify whether a different kernel would be faster.

T7/Pall kappa0 and.1 fail the original pointwise logit criterion in every
repeat. At kappa0, block332 fails; at.1, blocks299 and330 fail. Their pooled
loss differences are only+0.000021184 and+0.000029737, and relative-L2 criteria
pass, but that does not waive the predeclared elementwise bound. Display their
measured latencies with an unqualified marker and omit usable-speedup claims.
All four control implementations qualify throughout the grid. Across five
graph backends,159 of165 checkpoint/process/backend cells qualify; the six
failed cells are precisely those two frozen-policy rows across three repeats.

## What changed relative to opt073

Opt073 combines short-row scalar execution (up to8 nonzeros) with a masked
matrix fallback for h/z. Run050 instead gathers a compact union of active h
features across small token groups and performs tensor-core GEMM on the
corresponding weight rows. Its fixed winner sparsifies h in layers1--5, uses
dense fused h in layer0, and keeps z dense in every layer. Gate/output fusion
and the inexpensive no-gate Base fallback also remove substantial overhead.
The token-group sizes4/8/16 are not nonzero cutoffs. This sweep preserves that
policy for every row rather than tuning it to T7 or to each threshold.

## Diagnostics, limits and sources

Retain full-validation exact/near-zero counts and RMS/L2, weight norms,
row/group occupancy, h/z work counters, compiler/PTX metadata, profiles,
raw host/CUDA timing pairs and per-block numerical gates. The table JSON pools
integer h/z counts before dividing. Counts refer to actual BF16 execution;
canonical historical logical sparsity is a separate stored field. Requested
weight accesses and loop-derived work are not measured DRAM/L2 transactions.
The earlier hardware-counter probe on this same retained Pod returned
ERR_NVGPUCTRPERM; this sweep does not claim hardware traffic counters.

The inference-only diagnostic inventory cannot reconstruct historical gradient
interaction; those records remain in the original training runs. All checkpoint
and cache identities are retained. The final artifact audit and independent
timing/qualification reconciliation are recorded in
[retention-audit.json](../results/retention-audit.json) and
[verification.json](../results/verification.json).

Generating sources: [02_benchmark.py](../02_benchmark.py),
[03_diagnostics.py](../03_diagnostics.py), [05_reduce.py](../05_reduce.py),
[06_table.py](../06_table.py), and [07_verify.py](../07_verify.py).
This is a descriptive frozen-policy evaluation, not a new kernel search,
an equal-loss comparison, or a manuscript/finding promotion.

## User-approved interpretation, 22 September 2026

The user subsequently approved consolidation as
[F003](../../../research/findings/F003-70m-sparse-h-gain-does-not-establish-broad-hz-exploitation.md).
**This kernel does not demonstrate effective exploitation of the broader h/z
sparsity.** The supported result is a modest sparse h benefit in five layers:
5.55%/6.51% lower latency at T2/Ph kappa=.05/.1 against the same checkpoint's
optimized dense control, rather than attributing the entire approximately
1.4x PyTorch Base speedup to sparsity.

At kappa=.5, T2/Ph and T7/Pall reduce matched dense latency by8.43%/8.25%.
T2/Ph h/z are99.9077%/99.8470% zero, but every z projection and the first h
projection remain dense. Relative to T2/Ph .1, .5 saves only24.75microseconds
(2.11%) while loss increases from4.1901 to4.8465. This is limited coverage
and a poor additional quality/latency tradeoff in the measured policy, not
evidence that the sparse activation structure is intrinsically unusable.
The policy remains frozen from moderate T2/Ph; broader or separately tuned
high-threshold exploitation was not demonstrated. No manuscript was changed.
