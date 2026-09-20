# 70M overhead diagnosis and bounded optimization

## Question and verdict

Why did the original 70M port need roughly a twofold improvement over its own
dense path just to match native PyTorch, and can an implementation change
produce a genuine native-base gain without changing the model?

Yes, for the approved T7/Ph checkpoint at kappa0.5: the selected implementation
achieves **1.328912ms, or1.195408x the native base-model speed**. The original
port achieves1.571075ms, or1.011150x, in the same GPU session. The dense h/z
fallback accounts for most of the original base-model deficit. This is not
evidence that every 70M recipe now accelerates, nor that the dense path is fixed.

## Method and coverage

Retained step712 checkpoints c00=T0/P0 and c21=T7/Ph, kappa0.5, with their
original weights and gates. No training or quality/sparsity intervention was
changed. One RTX5090 (UUID28d34bbf-a5cf-5138-145b-66eeb4c2f700), pinned runtime,
BF16, batch1,2048 tokens and all50304 logits. Native execution uses the same
graph benchmark scaffold. Recurring gates and preprocessing are timed;
compilation, capture and equal input staging are excluded.

Nine diagnostic modes were evaluated for each checkpoint in three fresh
processes. Every process checked all338 complete validation blocks from all500
documents (692224 input tokens,691886 predicted tokens;1444-token excluded tail).
Timing uses the original64 inputs, seven passes and three processes, giving
1344 observations per mode/checkpoint. Tables report geometric means of
synchronized host latency. CUDA-event timings are retained independently.
Profiling and counter collection occur separately from these latency samples.

Optimization used only16 fixed training blocks. Three immutable candidates
were qualified on both checkpoints; opt002 was selected by T7/Ph training
latency and frozen before the six final evaluation processes. No tuning used
final validation latency. All operators, bounds, coverage and source hashes
are in [closeout verification](../results/closeout-verification.json).

## Final measurements

| Checkpoint and implementation | Latency (ms) | Speedup / native T0/P0 | Three process means (ms), min–max |
| --- | ---: | ---: | ---: |
| Native T0/P0 reference | 1.588592 | 1.000000x | Reference fixed across rows |
| T0/P0, original port | 3.263012 | 0.486848x | 3.259608–3.265138 |
| T0/P0, selected opt002 | 2.959854 | 0.536713x | 2.956318–2.964318 |
| T7/Ph, original port | 1.571075 | 1.011150x | 1.570713–1.571600 |
| T7/Ph, selected opt002 | 1.328912 | **1.195408x** | 1.326521–1.332281 |

Caption: one fixed native base-model denominator,1.5885916054020222ms, from
the c00/full processes. The ranges describe three process geometric means;
they are not confidence intervals or token/sequence quantiles. For opt002,
the same-process frozen T7/Ph reference is1.573949ms, so its paired latency
reduction is0.245037ms (15.57%). Relative to *native T7/Ph with the same gates*,
opt002 is1.364400x faster; that is a different question and denominator.
All full-precision cells and source hashes are in
[component-overhead.json](../results/component-overhead.json).

The old2.047x figure divided the historical specialized T0/P0 latency by
specialized T7/Ph latency. It was not a native-PyTorch-base gain. This fresh
session uses its own measured native reference; no old and new absolute
timings are mixed.

## What causes the overhead?

Each row below replaces only the named component in the original port, with
weights/gates unchanged. Positive values mean the replacement reduces latency.

| Replacement/control | T0/P0: original minus replacement (ms) | T7/Ph: original minus replacement (ms) |
| --- | ---: | ---: |
| Native h/z | **+1.660659** | −0.108228 |
| Native a/m | +0.071752 | +0.089445 |
| Native attention | +0.132413 | +0.115073 |
| Native normalization | −0.017622 | −0.072243 |
| Native RoPE | −0.139289 | −0.259397 |
| Disable h/z skipping | −0.042956 | −1.730113 |
| Disable all sparse skipping | +0.055512 | −1.646223 |

Caption: within-process conditional effects, using each cell's own paired
frozen graph. Effects are **nonadditive**; they are not an allocation of total
latency into independent causes. A skip toggle also changes compiled paths
and scheduling, so its net effect is not a pure measurement of inspection cost.

Native h/z replacement removes about1.66ms from the dense base path, nearly
the full1.67ms original-port deficit. Source inspection shows why this path is
poorly suited to dense operands: it pads eight real rows into sixteen-row
matrix instructions, inspects each activation group separately for four output
column blocks, and loads weights directly rather than using a shared-memory
weight pipeline. The experiment identifies the component responsible; it does
not assign an exact latency fraction to padding, inspection or memory traffic.

Full-validation counts show no short rows (at most two nonzeros) at any of
a,m,h,z in T0/P0. T7/Ph has94.87598% short rows at h and99.59483% at z; a and m
still have none. Thus the sparse h/z path can avoid its expensive general
matrix fallback for most rows, whereas the dense model cannot. Instrumented
GPU traces agree: original fused h/z takes1.870860ms on T0/P0 and0.201019ms on
T7/Ph (means of three four-input profiles). Native h/z GEMMs take about0.201ms.
The native GEMM scope excludes the gates/residual work fused into the custom
operation, so the full-model native-h/z replacement is the appropriate net
comparison. Profile durations are diagnostic, not substitutes for benchmark
latency. See [profile attribution](../results/profile-attribution.json).

## What changed, and what did not help?

The selected implementation keeps sparse h/z, fused normalization and RoPE,
but uses native PyTorch a/m projections and causal SDPA attention. Threshold
gates remain active at their original sites. Its h/z output tile widens from
128 to256 columns, using eight compute warps and two column blocks instead of
four. This halves repeated activation-group inspections across the512 output
columns. It does not imply a measured halving of physical memory traffic.
The short-row rule, accumulation order, BF16 rounding and counter definitions
remain unchanged. The same policy runs both checkpoints.

| Candidate | Change after native a/m + attention | T0/P0 development ms | T7/Ph development ms |
| --- | --- | ---: | ---: |
| opt001 | Original N128 h/z | 3.062644 | 1.377607 |
| **opt002** | **N256 h/z** | **2.954795** | **1.330982** |
| opt003 | N512 h/z | 3.556090 | 1.374374 |

Caption: geometric means over the fixed16 training blocks, five passes,
one process per candidate/checkpoint. All three qualified; none was discarded
for numerical failure. N512 was slower than N256, especially on dense inputs.
Reduced parallelism/register pressure are plausible explanations, not isolated
measurements. No larger search, dense dispatch, other recipe or threshold
evaluation was performed. The dense path still takes2.96ms versus1.59ms native.

These are **output-column widths**, not changes to the eight-row input
group or its K16 support mask. Their sparse bypass rule and total instruction
counter definitions are unchanged; the operator checks confirm matching
counts. Therefore the N512 slowdown does not show that larger tiles reduce
the frequency of skipping. The data support a nonmonotonic scheduling result:
N256 improves on N128 by about3.4% within the native a/m + attention composition,
and N512 regresses. The wider-tile contribution is only part of the final
gain; replacing a/m and attention is also material. An input-row tile sweep
and a matched cross-size sweep remain untested. They would be needed to
separate grouping-induced loss of bypass from scheduling/resource effects
or establish a general rule about increasing tile size with model size.

Native normalization/RoPE replacement and disabling skipping also failed to
improve T7/Ph. Its winning policy therefore preserves those useful custom
paths. Using native attention means the selected implementation does not
exploit activation-induced QK/PV matrix skipping; those counters are explicitly
unmeasured, not inferred from logical sparsity.

## Numerical and scientific limits

All28 initial operator checks,72 model controls,18 smokes,54 diagnostics,
six development processes and six final processes passed. N256 and N512 each
passed48 additional operator cases, including bitwise agreement with the
original operation and actual-counter agreement with an independent oracle.

Full-validation T0/P0 loss remains4.107689843018305. T7/Ph optimized and frozen
losses both equal5.346554324627495, versus5.346554364148095 for native execution
(difference−3.95e−8). T0/P0 logits are bitwise equal; the largest T7/Ph logit
absolute difference is0.5, satisfying the declared elementwise combined
absolute/relative bound, with maximum relativeL2=1.059e−5. These are the BF16
benchmark qualification losses, reproducing Run035; do not substitute them
for the paper's canonical FP16 checkpoint losses.

The optimization preserves checkpoint quality but does not erase T7/Ph's
quality cost relative to the base model. Results cover only two checkpoints,
one GPU, one shape/runtime and three within-session replicates. They establish
a measured1.195x native-base gain at this point, not a universal70M speedup or
a model-size scaling law. Logical-product opportunities, sparse MMA bypass,
and runtime gains remain distinct. Diagnostic BF16 operand statistics are
not a replacement for canonical S_model accounting.

## Recovery, sources and cost

The6,629,461-byte archive (SHA256
`5f4f46271a89242d00cdedae2f3fb33db34aad9d60797d30f308e1eb936f5e40`)
and all754 members verified locally. The separately copied freeze helper
also matched its frozen source hash, and all1404 immutable input/source copies
were reverified before deletion. Raw timing, quality, traces, counters,
activation/weight statistics, failed infrastructure attempts, candidate
sources and checkpoint/cache identities are retained. Both original final
checkpoints remain local. The final reduction reproduces byte-for-byte.

The owned pod was deleted at12:58:51 UTC after1.559 GPU hours, approximately
USD1.54 compute / USD1.57 including disk, below the USD6 cap. Posted billing
was partial at teardown; this is an elapsed-time estimate. Both stop guards
were retired. The independent Run041 pod and pre-existing network volume were
preserved. No manuscript edits, PDF rebuild or push were performed.
The user subsequently offered an additional GPU hour for exploration, then
asked to finish this round before another launch. No follow-up pod was created;
that additional hour remains unused.

Generating sources: `02_benchmark.py`, `controls.py`, `diagnostics.py`,
`profiling.py`, `candidates/opt002/`, `candidates/develop.py`,
`candidates/check_joint.py`, `candidates/profile_selected.py`,
`prelaunch/freeze_selected.py`, `07_reduce.py`, `14_profile_attribution.py`
and `15_verify_closeout.py`. Immutable manifests, development selection and
the exact effective policy are retained under `provenance/` and `candidates/`.
