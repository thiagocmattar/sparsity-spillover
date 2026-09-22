# Qualified latency gains at moderate kappa

Question: can existing T2/Ph h/z sparsity at kappa .05/.1 beat a strong dense
implementation without changing the checkpoint, gate or numerical bounds?

**Result: yes for this workload and these checkpoints.** The shared gathered
tensor-core policy qualifies at both kappas, reduces latency by more than5%
against the best dense h/z control, and beats efficient Base in every process.

## Method and coverage

Four families received six initial configurations each: packed streaming A,
masked-load split-K B, gathered tensor-core C, and exact2:4 sparse MMA with
lossless overflow D. A/C earned refinement trials; six refinements were used.
All packing, gate, list construction, reduction and combination work is timed.
Static weight-layout preparation, compilation, capture and equal input staging
are excluded for every backend. No new pruning, training or parameter change.

Training cache blocks0:64 and64:128 determine a single minimax site/layer policy
shared by both kappas. The freeze is
[`selection-final.json`](../provenance/selection-final.json), SHA256
`5945124e84d7b3519a8a7d57baa4449000083c3a9e4c61c23293946ff57d63d7`.
No validation result changes this policy.

One retained RTX5090; BF16; batch1;2048-token uncached full forward;50304 logits;
CUDA graphs; unchanged step712 checkpoints and software/precision pins inherited
from Run049. The reference is the same-checkpoint unmodified eager model.
Every implementation is checked on all338 complete validation blocks from the
500-document source:692224 input tokens,691886 prediction tokens,1444 excluded
tail tokens. Timing uses64 fixed validation inputs, seven paired passes and
three fresh processes per checkpoint:32256 graph timing samples in total.

## Latency and attribution

Values are geometric mean host milliseconds across the three processes.

| Implementation | Base | T2/Ph .05 | T2/Ph .1 |
| --- | ---: | ---: | ---: |
| Native PyTorch graph |1.646161|1.704993|1.711770|
| Prior efficient dense h/z (`native_hz`) |1.256577|1.325069|1.330147|
| New fused-gate dense policy |1.256508|1.251942|1.257072|
| Gathered sparse policy C |1.257216|**1.182477**|**1.175177**|
| Packed sparse policy A |1.256856|1.225938|1.227437|
| C layout with skipping disabled |1.256865|1.784300|1.788752|
| Earlier opt073 control |2.715776|2.137523|1.963710|

The primary C policy reduces time by **5.5486% /6.5147%** against the best
qualified dense control on the same .05/.1 checkpoints. Paired speedup95%
intervals are[1.058232,1.059229] and[1.069283,1.070069]. Resampling processes
as well as input blocks gives[1.058234,1.059227] and[1.069261,1.070100].
A more conservative sensitivity analysis resamples the same input identities
jointly across processes and also resamples processes: its intervals are
[1.057831,1.059512] and[1.068986,1.070323], still fully favoring sparse.
These intervals describe this timing experiment, not a GPU population or
training-seed population.

Relative to native Base, speedups are **1.392130x /1.400777x**. Relative to
efficient Base, they are1.062666x /1.069266x; every paired process round favors
the T2 sparse execution. Base's no-gate route adds at most0.0527% in the three
observed comparisons, below the2% bound. Cross-checkpoint Base ratios compare
separate fresh processes; same-checkpoint sparse/dense ratios use paired timing.

Fusion is a separate benefit: native_hz to the new dense policy removes about
73 microseconds at either kappa. Sparse execution then saves another69.47 /
81.89 microseconds. The slow skip-disabled C layout is an attribution control,
not the denominator used to establish the sparse benefit.

## What the winning kernel exploits

C constructs the union of active K features across a small group of token rows,
loads only corresponding weight rows, and computes the compact tile with BF16
tensor cores and FP32 accumulation. There is no maximum-NNZ truncation.
Row groups8/4/4/4/16 are used in h layers1--5 (zero-based); each uses N128/K32
tiles. h layer0 stays dense. Every z layer stays dense. A row-group size of8 is
the number of token rows sharing a feature list, not the old eight-NNZ cutoff.

This demonstrates usable h structure in T2/Ph at moderate kappa. The global
zero fraction alone did not identify the winning algorithm or dispatch.
Among tested implementations, conversion and compute costs leave dense z
execution faster. This does not establish that no future z kernel can win.

## Numerical qualification

All72 process/backend graph evaluations qualify; none of their24336 block
comparisons fails. Bounds remain `abs(error)<=.25+.02*abs(reference)`,
relativeL2<=.02, and absolute pooled-loss change<=.001.

| Checkpoint | Native BF16 loss | C loss | C loss delta | Maximum relativeL2 |
| --- | ---: | ---: | ---: | ---: |
| Base |4.107689843|4.107689843|0|0|
| T2/Ph .05 |4.200080141|4.200073885|-0.000006256|0.000602925|
| T2/Ph .1 |4.190060599|4.190054148|-0.000006451|0.001004313|

The largest absolute logit difference for C is0.5, which passes the declared
elementwise bound because it includes the reference-magnitude term. Sparse
compaction changes floating-point reduction order; it is not bitwise identity.
These BF16 losses must not replace the paper's canonical FP16 losses silently.

## Scope and provenance

This result covers two existing moderate-kappa checkpoints, one training seed,
one GPU SKU and full-sequence inference. It does not test T7/Pall, decode or
other architectures. B/D's unsuccessful initial trials do not refute those
algorithm families generally. No manuscript or research finding is promoted
from this run without the user's narrative approval.

Source scripts: [`11_reduce.py`](../11_reduce.py),
[`05_benchmark.py`](../05_benchmark.py), and
[`04_select.py`](../04_select.py). The complete numerical/timing source hashes,
process values, confidence intervals and acceptance decisions are in
[`final-summary.json`](../results/final-summary.json). Full structure, work,
profile and retention details are recorded in observation002.
