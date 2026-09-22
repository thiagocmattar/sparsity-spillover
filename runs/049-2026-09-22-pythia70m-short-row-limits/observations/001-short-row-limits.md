# Larger h/z short-row limits at lower-loss 70M checkpoints

## Question and method

Can changing the short-row capacity from8 to16/32/64 improve T2/Ph kappa=.1
latency, while preserving numerical qualification, and beat native PyTorch Base?
Base and T2/Ph kappa=.05 provide matched controls. The capacity counts
**nonzeros per token row**, independently of the unchanged8-row matrix group.

The selected step712 checkpoints, weights, gates, input cache, precision flags
and M8/N256/K16 fallback are unchanged. New-limit backends use native h/z in
ungated layers; all Base layers therefore execute the efficient dense fallback.
The native_hz comparator also uses dense h/z on gated checkpoints, while
preserving their gates and the other opt073 components. This comparison adds
no activation-dependent dense/sparse dispatcher.

One physical Secure RTX5090 in EU-RO-1; BF16, batch1, T2048, full50304 logits.
Each checkpoint has three fresh final processes, each measuring six backends
on64 validation inputs in seven randomized paired passes. Each table cell is
the geometric mean of1,344 host timings. Compilation, graph capture and equal
input staging are excluded; recurring scans, gates and dispatch are included.
These are full-forward latencies, not training-update timings.

All500 validation documents are covered:338 complete blocks,692224 input
tokens,691886 prediction tokens and1444 excluded tail tokens. Every backend
is compared with unmodified same-checkpoint eager PyTorch on all338 blocks.
Bounds remain elementwise atol.25+rtol.02, relativeL2<=.02, and pooled loss
delta<=.001. Integer diagnostic counts are pooled before division.

Development used16 fixed training inputs and selected **native_hz** before
validation. All predeclared backends retain final measurements and failures;
the selection was not revised after inspecting validation.

## Results

Full-forward latency in milliseconds. A dagger marks failed numerical
qualification; these cells are timings of rejected implementations, not
qualified speedups. Native PyTorch means its CUDA-graph execution throughout.

| Checkpoint | Native PyTorch | Existing opt073, limit8 | Efficient dense h/z | Limit16 | Limit32 | Limit64 |
|---|---:|---:|---:|---:|---:|---:|
| Base |1.642892|2.707292|1.254255|1.254743|1.254369|1.254540|
| T2/Ph .05 |1.702279|2.131420|1.323351|2.105063†|2.449257†|3.088607†|
| T2/Ph .1 |1.703804|1.953529|1.324743|1.947634†|2.145114†|2.401166†|

The efficient dense h/z control gives **1.3099x** native Base throughput on
Base, **1.2415x** on kappa=.05 and **1.2402x** on kappa=.1. The latter is a
19.4% latency reduction against native Base and28.6% higher throughput than
same-checkpoint native PyTorch. Its gain comes from dense implementation
improvements. The gated .1 checkpoint remains5.6% slower than efficient Base.

For .1, native_hz process means span1.324608--1.324829ms; native Base spans
1.642556--1.643331ms. These are observed process ranges, not confidence intervals.
No additional sparse speedup is established by raising the cutoff. Limit16
changes raw latency by only about0.3% relative to opt073 and fails qualification;
limits32/64 are slower.

Full-validation BF16 loss for native and native_hz is identical at each
checkpoint: Base4.107689843, .05=4.200080141, .1=4.190060599. These BF16 numerical
qualification losses differ from the retained canonical FP16 losses used by
the manuscript's quality axis; the weights and quality-axis convention were
not changed.

All larger limits pass pooled-loss and relativeL2 checks but fail the
elementwise logit bound. At .05, limits16/32/64 fail1/2/8 distinct blocks;
at .1 they fail2/5/8 blocks. The same failures recur in all three processes.
For .1, limit16 fails blocks117 and329; limit32 adds78,113,128; limit64 also
fails115,260,330. Native, native_hz and opt073 qualify for all checkpoints.
All112 synthetic operator cases passed, demonstrating why full-model,
full-corpus qualification remains necessary.

## Why more eligible rows did not produce faster execution

The following .1 values use actual BF16 operands from each backend. Row/group
fractions pool all six layers and338 blocks. MMA and scalar totals are actual
instrumented h+z counters, independently checked against operand oracles.
Profile times are separate instrumented graph measurements on four inputs;
each component contains24 kernel calls (six layers times four inputs).

| Capacity | h rows eligible | z rows eligible | Groups with all8 paired rows short | Issued h+z MMA, billions | Scalar products, billions | Prepass ms | Fallback ms |
|---|---:|---:|---:|---:|---:|---:|---:|
|8|45.094%|19.748%|0.943%|2.3414|7.7739|.04742|.90398|
|16|60.584%|66.939%|15.982%|1.8674|30.8818|.09640|.85949|
|32|73.726%|93.286%|50.438%|1.3487|53.4469|.21444|.96114|
|64|94.432%|98.656%|78.802%|.6100|82.4504|.60804|.79323|

Limit64 issues about74% fewer MMA instructions, but executes10.6 times the
scalar products. Mixed groups also retain duplicated prepass/fallback scalar
work, which is included in these totals. The prepass grows from.047 to.608ms,
offsetting the fallback reduction. This explains the observed failure of
this capacity-only change to convert greater eligibility into lower latency.

Compiler reports provide supporting evidence: the timed fallback uses
71/121/158/255 registers at capacities8/16/32/64; the prepass uses56/96/168/255.
Limit64 spills registers in both components; its fallback reports152 spill-store
bytes and628 spill-load bytes. These compiler quantities are static resource
reports, not measured memory traffic. The profiles do not uniquely separate
all causes of slowdown.

## Coverage, interpretation and retained evidence

This run supports the dense-overhead part of the objective. It does **not**
establish an additional activation-sparsity benefit for the lower-loss T2/Ph
checkpoints through a larger scalar cutoff. It does not rule out different
kernel algorithms, independent h/z policies, matrix layouts or dispatch rules.
No new T7/Pall, training, other GPU, or other sequence-length result is claimed.

All requested diagnostics are retained: checkpoint/cache identities,
full-validation exact/near-zero counts and RMS/L2, weight norms, per-layer row
histograms, group occupancy at8/16/32/64, verified h/z work counters, graph and
eager profiles, raw host/CUDA timings, and per-block numerical checks. Native
h/z instruction counts remain unmeasured; absent counters are not zero.
Occupancy for a capacity not executing in that backend is counterfactual.

The initial diagnostics import failed after .05 validation because an archived
module shadowed the intended run-local module. The unchanged scientific
sources resumed through explicit import bindings in
`provenance/recovery-001/`. Failed attempt `final-c24-r1-001` remains intact;
attempt002 supplies the complete .05 replicate1. The original development
selection and remaining final order were preserved. A one-block recovery
smoke checked all five diagnostic paths and12 profile outputs before resuming.

The output archive and all298 inventoried files verify locally. The recovery
reducer reproduces the remote summary exactly. A local optional-control copy
collision was corrected from the verified archive; it did not alter scientific
inputs or remote results. The Pod is retained at the user's explicit request,
with its original compute-stop deadline of20:30:06UTC on22 September.

Sources: [verified latency summary](../results/summary.json),
[mechanism and failure summary](../results/mechanism.json),
[local transfer verification](../results/local-verification.json),
[inventory](../transfer/inventory-002.json), and
[recovery record](../provenance/recovery-001/manifest.json).
Generating scripts: [benchmark](../02_benchmark.py),
[recovery reducer](../provenance/recovery-001/reduce.py),
[diagnostics](../diagnostics.py), and [post-hoc interpretation](../11_interpret.py).
