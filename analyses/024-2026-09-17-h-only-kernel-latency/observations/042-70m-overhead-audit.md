# Why is the specialized 70M base model slower than native PyTorch?

## Question and status

The 70M specialized base takes roughly twice the time of the same checkpoint
under native PyTorch. Is this a property of the sparsity intervention, or
overhead introduced by this implementation? This audit re-reads existing
timings, full-validation counters and source. No new GPU measurements are
performed. It identifies concrete extra work and proposes a component
diagnostic; it cannot yet attribute milliseconds to individual components.

## Method, coverage and timing result

Reproduce with `../32_audit_70m_overhead.py`. The reducer checks both base
checkpoints against their original three qualified processes, each using all
338 validation blocks and 64 inputs x7 timing passes. It recomputes geometric
means from 1344 samples per implementation and size, checks matching input
identities and full50304 output logits, reconciles h/z counters with raw
diagnostics, verifies inspected kernel sources against the frozen Run035
identity, and retains 29 source hashes in `../data/70m-overhead-audit.json`.
Native and specialized times are paired within each size's original session;
14M and70M were measured in different RTX5090 sessions. Validation covers
all500 documents, with the1444-token incomplete tail excluded.

| Base model | Native host ms | Specialized host ms | Native CUDA ms | Specialized CUDA ms |
|---|---:|---:|---:|---:|
| 14M T0/P0 | 0.655442 | 0.651573 | 0.645728 | 0.641891 |
| 70M T0/P0 | 1.663883 | 3.313247 | 1.651332 | 3.295209 |

Table caption: full-sequence, batch1, BF16, full-logit inference with the
native and specialized implementations of the same unchanged base checkpoint.
Host timings include synchronized CUDA-graph replay; CUDA events measure
device execution. Compile, graph capture, static weight-layout preparation
and equal input staging are excluded from both modes. Runtime inspections
and preprocessing are included.

The70M host gap is1.649364ms and the CUDA gap is1.643877ms: almost the entire
difference is visible on the GPU. Their difference is only0.005487ms.
These differences of geometric means are descriptive, not an exact additive
partition of host and device time. Counter collection runs after timing;
the timed h/z and attention paths compile out counter writes.

## Confirmed implementation costs

1. **The base model pays for sparse machinery with no projection bypass.**
   Full-validation counters show zero skipped projection MMAs and zero
   short-row scalar substitutions at both sizes. Across4,153,344 site-layer
   rows per site, none at a,m,h,z has at most two nonzeros. The base has no
   threshold gates or training pressure, so their presence cannot explain
   its measured deficit.

2. **The h/z dense fallback uses half of each matrix instruction's rows.**
   `Run035/kernel/joint/joint.cu` passes eight real rows plus eight zero rows
   to m16n8k16 instructions. With no bypass, h and z each issue twice the
   scalar-product capacity required by their dense matrix products. This is
   confirmed independently by the retained issued-MMA and logical-product
   counts. It is not a claim that native PyTorch issues exactly half as many
   instructions: its actual instruction schedule has not been profiled here.
   Across a,m,h,z together, the custom dense projections issue17/12=1.416667
   times the logical matrix-product capacity; the excess comes from h/z.
   This is arithmetic accounting, not a predicted41.7% latency penalty.

3. **The shape port repeats h/z inspection across output tiles.**
   At14M the output width128 fits one N128 tile. At70M the output width512
   uses four blocks for each group of eight tokens. Every block independently
   scans the full h2048 and z512 rows before matrix execution. Source-level
   inspection reads therefore rise from1,310,720 to20,971,520 BF16 values
   per layer, or2.5 to40MiB of load requests. The arrays themselves grow4x;
   repeated scanning adds another4x. These are source-level requests, not
   physical DRAM traffic: cache reuse can serve them. The matrix fallback
   reads active inputs again and explicitly loads weights inside its K16
   loop. It has no shared-memory weight staging/pipeline in this source.

4. **Other components retain sparse checks and fixed schedules.**
   QKV and FFN-up retain the14M CUTLASS32x64x64, two-stage, Sm80-style
   matrix pipeline; each exact-zero fragment check occurs after operand
   loading. Attention checks both matrix operands before each bypass decision,
   again after loading. Its70M Flash attention tile/reduction schedule was
   corrected to match observed native dispatch for numerical compatibility;
   that correction was not a performance search. Checks and a fixed small
   tile may cost time even with no zeros. Their actual contributions remain
   unmeasured. The vocabulary projection remains native in both paths.

The70M attention wrapper's q/k/v `.contiguous()` calls are not evidence of
three extra copies: the preceding fused RoPE code already creates contiguous
buffers. A profile must establish actual copy kernels. Similarly, allocated
but unused legacy attention buffers do not establish recurring timed work.

## Why the difference can grow with size

Both architectures have six layers. Width grows128->512 and intermediate
width512->2048, so each dense projection's matrix-product count grows16x,
whereas the unchanged-vocabulary head grows4x. The eight-row padding already
exists at14M; its absolute projection work and the repeated inspections become
much larger at70M. Efficient dense matrix execution becomes more important.
The original port preserved a14M-specialized implementation rather than
performing an equally extensive70M optimization search.

This supports an implementation-specific explanation. It does not prove
that h/z accounts for most of the1.65ms, that removing padding will halve
latency, or that a revised kernel will yield large native-base gains. The
remaining sparse work and the unaccelerated head still impose a floor.

Existing results are consistent with this explanation: strong h/z sparsity
can avoid the expensive fallback and bring the70M implementation back near
native-base latency. This is not by itself evidence for improved scaling.
Run039 provides related14M evidence: porting the same style to low-sparsity
a/m slowed execution44.49%; the changed layout with sparse paths disabled
still slowed it43.14%. That experiment shows that a fallback/layout can be
costly independently of checking, but does not quantify the70M deficit.

## Missing measurement and proposed next step

The retained70M component substitutions were explicitly untimed correctness
diagnostics. Its sole retained profiler trace contains one native attention
call with synthetic inputs (58.847us), not a paired model/component profile.
It cannot be subtracted from model latency or compared with a missing custom
attention time. No70M all-skips-off latency is available.

The [proposed diagnostic](../70M-OVERHEAD-DIAGNOSTIC-DESIGN.md) replaces one
component at a time with native execution and separately disables sparse
paths, at fixed checkpoints/gates. It compares T0/P0 with the fastest retained
sparse T7/Ph endpoint at kappa0.5. Untimed GPU profiling complements the
qualified full-model timing contrasts. The user approved both checkpoints on
20 September and additionally authorized optimization guided by the diagnosis.
Implementation and launch approval follow; no new GPU result exists yet.

No manuscript claims or figures are changed by this audit. In particular,
the historical2.047x specialized-base ratio remains distinct from the1.028x
native-base ratio; neither is a prediction for a future implementation.

## Source provenance

- Run035: `kernel/joint/joint.cu`, `kernel/projection/projection.cu`,
  `kernel/attention/sparse_gemm.h`, `kernel/candidate.py`, `02_benchmark.py`,
  `20_numerical_isolation.py`, `21_profile_native_attention.py` and original
  `scientific-c00-r{1,2,3}-001` artifacts.
- Run029: original T0/P0 timing/diagnostic artifacts resolved by the reducer.
- Analysis024: `data/paper-checkpoints.json`, `data/operation-bypass.json`,
  [Observation041](041-kernel-appendix.md) for reference normalization.
- Run039: `results/am-load-avoidance.json` and
  [Observation035](035-am-load-avoidance-port.md) for the related14M experiment.

The new JSON contains exact values and source hashes; the source-code
inspection interpretation is recorded here rather than presented as a GPU
measurement. No figure was needed for this audit.
