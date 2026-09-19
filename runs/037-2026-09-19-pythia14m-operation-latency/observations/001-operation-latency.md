# Conditional operation latency: Pythia-14M T7/Pall at kappa=0.5

## Question

Does the final kernel benefit from h/z skipping while additional QK/PV
sparsity fails to reduce full-model latency? The user approved six operation
groups and the kappa=0.5 checkpoint, followed by explicit launch approval.
See the [approved design](../../../analyses/024-2026-09-17-h-only-kernel-latency/PER-SITE-LATENCY-DESIGN.md).

## Method and coverage

Use the retained Run029 c30 T7/Pall final checkpoint, weight SHA256
`f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
Weights and all seven gates stay fixed. Ten execution modes include untouched
K050, the independently controlled version with all paths on, all off,
projection-only, and six modes disabling only a, m, h, z, QK, or PV.
QK covers q/k jointly; the h/z fusion remains intact. This is an execution
ablation, not retraining or removal of interventions.

One RTX5090, BF16, batch 1, sequence 2048, uncached causal inference, CUDA graphs,
and full 50304-token logits. Thirty fresh processes (three per mode) use the
same 64 validation identities and seven passes, in seed2504 randomized order.
Each process checks all 338 complete blocks from 500 MiniPile validation
documents: 692224 inputs, 691886 prediction tokens, 1444-token excluded tail.
One full diagnostic pass per mode pools integer counters across all layers
and 338 blocks. Twenty direct h/z cases and ten eight-block smokes precede
the scientific matrix; smoke timings are excluded.

`saved time = latency(path disabled) - latency(all paths enabled)`.
Positive values indicate a benefit from enabling that path. The conditional
ratio divides those same latencies. These effects are not additive shares.
Geometric means pool 1344 host timing samples per mode. Process-extrema spans
compare the smallest/largest of three process means against the opposite
extreme of the full mode; they are descriptive spans, not confidence intervals.

Sources: unchanged GPU scripts `02_benchmark.py`, `03_execute.py`,
`06_cuda_checks.py`, `07_reduce.py`; local independent audit/table generator
[`12_report.py`](../12_report.py). Raw samples and per-block checks are in
`artifacts/attempts/`; exact values and source hashes are in
[`operation-latency.json`](../results/operation-latency.json) and
[`conditional-effects-audit.json`](../results/conditional-effects-audit.json).

## Table and results

**Caption: Conditional latency effects of sparse execution paths.**
Pythia-14M T7/Pall at kappa=0.5, with fixed weights and gates. Each row disables
one skipping path while leaving the others enabled. Bypass is the fraction
of issued-plus-bypassed matrix instructions avoided; it is not a fraction
of elapsed time. h/z bypass includes scalar substitution, and attention
counts include causal masking/padding. Full mode latency is 0.521078ms.

| Path | Operation | Bypass (%) | Path disabled (ms) | Time saved by enabling (us) | Process-extrema span (us) |
|---|---|---:|---:|---:|---:|
| a | QKV projection | 8.44 | 0.519449 | -1.63 | [-5.99, +3.84] |
| m | FFN-up | 0.00 | 0.518717 | -2.36 | [-5.11, +2.73] |
| h | FFN-down | 94.50 | 0.671148 | +150.07 | [+145.45, +154.39] |
| z | Attention-output projection | 99.64 | 0.550038 | +28.96 | [+23.71, +35.69] |
| q,k | QK scores | 58.00 | 0.518164 | -2.91 | [-7.29, +1.06] |
| v | PV product | 67.21 | 0.514900 | -6.18 | [-9.02, -2.18] |

The h path has the largest clear conditional benefit, followed by z.
Their conditional speedups are 1.2880x and 1.0556x. PV skipping has a clear
net cost in this process-extrema comparison. The a, m and QK point estimates
are slightly negative, but their spans include zero: do not claim that their
individual signs are resolved.

With all projection skipping fixed, enabling both attention paths raises
latency from 0.514037 to 0.521078ms: **+7.04us, or+1.37%**. The process ranges
are disjoint (projection0.512032–0.515731ms; full0.519136–0.522158ms).
All-off latency is 0.671637ms, giving full mode a 1.28894x gain over the same
optimized implementation with skipping disabled. That reference is not A0.

Untouched K050 measures 0.518482ms versus 0.521078ms for the controlled full
mode (+0.5007%). Their process ranges overlap (frozen0.516643–0.520358ms;
full0.519136–0.522158ms), satisfying the predeclared fidelity rule. Overlap
does not prove exact performance equivalence.

All thirty processes qualify. Pooled BF16 validation loss is exactly
5.83130066301001 for every accelerated/reference evaluation, and maximum
absolute logit error over all recorded block checks is0.0. This BF16 inference
check is not a replacement for the paper's ordinary FP16 endpoint loss.
The audit rebuilds 26,880 raw candidate/native timing records, checks every
disabled path has zero bypass/scalar counts, verifies unchanged enabled-path
counters, and confirms that logical sparsity counts stay fixed across modes.

## Interpretation and limits

This supports the proposed mechanism for this checkpoint and workload:
h/z exploit activation structure before requesting corresponding weights,
whereas the attention zero checks follow operand loading and leave softmax
work intact. The experiment measures the net benefit of each execution path;
it does not separately measure DRAM traffic, checking overhead, softmax time,
or arithmetic time. The code establishes ordering; timing does not isolate
each causal component of that explanation.

High bypass alone does not guarantee lower latency. The clear PV overhead
and grouped attention slowdown answer the attention question, while QK's
small individual effect remains unresolved. Scalar replacement and h/z
fusion mean the conditional effects cannot be summed or called percentages
of total gains. No result is extrapolated to 70M, other kappas/recipes,
another device, or cached decoding. This new Pod session's absolute latencies
must not silently replace historical figure measurements.

The result is retained for the requested manuscript subsection
`manuscript/draft/kernel-autoresearch.tex`. No new result-bearing manuscript
text or figure was inserted during this launch; its current table still
uses the previously documented Run029 controls.

## Recovery and cost

All 365 archived files, plus final worker/guard records, were SHA256-verified
locally before Pod deletion. The checkpoint/cache identities remain retained.
Infrastructure fixes (PATH, verified local runtime/cache, timestamp correction,
direct executable with original compiler paths) are fully logged and precede
the scientific matrix. No scientific input or kernel source changed.

The Pod was deleted at 14:38:04 UTC after 69.22 GPU-minutes. Estimated incremental
cost is USD1.1493, below USD2; no active Pods remain. The pre-existing network
volume is unchanged. See [teardown](../artifacts/closeout/teardown.json) and
[verification](../artifacts/verification.json).
