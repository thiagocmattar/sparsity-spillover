# Full forward latency decomposition

Status: completed and verified. All artifacts are local; the owned Pod is terminated.

Question: where does resident-input, uncached full-sequence forward latency go,
and which actual execution stages account for Run063's specialization advantage?
Compare native, generic/specialized sparse and both no-h/z-skip controls on one
physical RTX5090. No threshold, weight, arithmetic, training or kernel tuning.

Reuse the two retained Run060 T2/Ph HZ checkpoints at kappa 0/.1, step712,
training seed1234, originally randomly initialized Pythia14M (6 layers, D128,
MLP512, 4 heads of width32, vocabulary50304). Their pressure/optimizer history
and exact manifests stay unchanged. This is inference: no initialization,
optimizer, training budget, gradients or clipping intervention is introduced.
The unchanged corner8x16 gate at h replaces GELU and at z precedes Wo; a tile
survives iff its top-left value exceeds the threshold, retaining its signed
interior. a/m have no gates. Both checkpoints remain retained locally.

BF16 B1/T2048, no KV cache/padding, complete logits. The numerical bounds from
Run063 remain fixed. All 500 validation documents/338 complete blocks/692224
input tokens/691886 predictions, with the excluded1444-token tail, qualify
every original and mapped/timed graph in all six fresh processes. Runtime seed
2801, timing seed2504;64 fixed inputs x7 randomized paired passes x3 processes
per threshold. Inputs are staged equally outside the primary latency boundary.

The operational runtime claim is actual synchronized-host/CUDA latency, not
R_model, zero mass or the manuscript's analytic ceiling. This diagnoses the
existing speed claim; no manuscript wording or scientific gate definition changes.
Support: stable differences localized to measured stages with numerical fidelity
and an explicit measurement-overhead audit. Refutation: the advantage vanishes,
is outside the changed stage, or profiling cannot support reliable localization.

## Measurement layers

1. Unmodified graph replay: synchronized host and outer CUDA events, plus
   separately measured equal staging and allocation/capture/setup.
2. Semantically equivalent explicit forward map: one external CUDA event per
   stage boundary in graph capture. Nonoverlapping intervals telescope to the
   timed graph total. Report arithmetic means for additive budgets, distributions,
   per-input/process variability and instrumented/uninstrumented ratios. Do not
   silently rescale stages to the baseline total.
3. CUPTI kernel traces of uninstrumented replay and annotated mapped eager
   execution: retain raw timelines, kernel counts/durations, CPU dispatch,
   GPU active union and gaps. Eager launch gaps are not graph-replay latency.
4. Diagnostic-only copies of the fused h/z CUDA kernels use SM clock timestamps
   for support/preparation, h accumulation, z accumulation and output rounding/
   residual stores. Record block/warp cycles and distributions, sparse density,
   actual kernel-event time and probe slowdown. These are overlapping SM-local
   work measurements, never additive full-device wall-time attribution.

The map includes positions, embedding, rotary-table generation, six repeated
blocks, final normalization and LM head. Optimized block stages: paired norms,
QKV projection, RoPE/QKV layouts, causal attention (QK/softmax/PV plus split
combine), context packing, MLP up projection, and fused gate/W2/Wo/residual.
Native stages expose separate norms, h/z gates, projections and residual adds.
Fused attention's kernel trace exposes split attention and recombination; its
QK/softmax/PV are a fused unit, without fabricated individual wall timings.

Retain all raw timings/traces, per-layer activation exact/near-zero counts at
0/.001/.01, RMS/L2, per-layer weight norms, empty8x16 tiles, MMA/SIMT counters,
full qualification, input indices/cache identity, source/environment hashes,
logs, timing-overhead and internal-cycle probes. Existing training gradient/OL1
evidence stays local; it cannot be reconstructed by this inference run. No new
checkpoint is needed. Publication figures are PDF with source-linked observations.

## Cloud definition

One Community RTX5090, rented USD.69/hour, compatible host CUDA13.0;
same pinned image/application as Run063. 40GB container+25GB persistent Pod
workspace; no network volume. Flexible two-hour/USD3 backstop, estimated
25–60min including transfer/setup/cold compilation and profiling; Run063 took
52min overall with retries while its six scientific processes took4m48s.
Local CPU verification only; RTX5090 required for frozen attention fidelity.
Monitor every60s, refresh ETC and report progress/loss/throughput/memory. Stop
on nonfinite outputs, numerical failures, OOM, disk exhaustion or stale progress
over300s (cold compiler activity inspected separately). Detached logging and a
scoped workstation stop guard preserve retrieval at the deadline. Copy and hash
verify every agreed artifact before deleting the owned Pod; audit resources.

Unrelated pre-existing stopped Run052 Pod qa00sm05lmrs7w and100GB network volume
9luykg5yc3 are not used. The user's explicit pre-approval covers this design,
the post-hoc inventory and launch; no further confirmation is required.

Prelaunch:257 tests passed (242 bootstrap,10 prior kernel focused,5 new additive
accounting checks) in9.87s. Python, PowerShell guard and shell syntax passed.
Secure capacity disappeared on final refresh; Community stock on CUDA13.0 was
available at.69/hour, within the unchanged two-hour/USD3 envelope. Owned Pod
p79ch842a0hagf was created21:35:54.414UTC; scoped stop deadline23:35:54.414UTC.
The110MB sealed upload has1436 individually inventoried files; immutable
Run063 source/checkpoint/cache bytes are dependencies, not edited historical
records. Analysis/report scripts are local postprocessing and do not alter the
sealed cloud executable. See [lease](prelaunch/lease-001.json),
[checks](prelaunch/checks.json) and [bundle receipt](prelaunch/bundle-receipt.json).

## Verified result and evidence

The accepted six-process cohort is `full-*-003`. Generic-to-specialized
synchronized host latency falls 568.726 to 509.939 us (10.337%) at kappa 0,
and 534.562 to 486.441 us (9.002%) at kappa .1. The separate CUPTI sample
localizes essentially the entire GPU-active difference to the six fused
gate/W2/Wo/residual kernels: 121.025 to 63.397 us, and 89.324 to 40.568 us.
All other optimized stages are nearly matched. With skipping disabled,
specialization is 3.744%/3.796% slower; this is a structure-dependent advantage.

In the specialized GPU-active budget, the vocabulary projection takes
38.313%/40.200%, attention 25.754%/27.125%, and the joint kernels
13.458%/9.075%. This is a resident-input forward of all 2,048 tokens with
50,304-way logits, not autoregressive one-token decoding. The complete
[observation](observations/01-forward-decomposition.md) maps every stage and
lists pooled and per-layer timings, controls, distributions and timer caveats.
The [four-page PDF](figures/01-forward-latency-decomposition.pdf) shows full
distributions, per-layer joint kernels and internal cycle diagnostics.
[Machine-readable reduction](results/decomposition.json) retains all modes.
[Complete CUDA kernel inventory](results/kernel-nodes.csv) adds every ordered
activity and its duration distribution; attention split/combine and rotary-table
substeps are also tabulated in the observation.

Each mode has 1,344 baseline and 1,344 event-partition samples per threshold.
Forty input/mode trace mappings cover 120 profiled graph forwards plus 40
annotated eager forwards. All 20,280 full-block logit comparisons have zero
absolute and relative-L2 error, and exact loss equality at
5.556248486078721/5.5987847543876. Activation/tile/MMA/SIMT diagnostics match
between consumers across all 338 blocks. All 384 internal probes preserve
production outputs exactly; 7,864,320 warp-phase cycle intervals are retained.
The checkpoints and gate rule are unchanged; the tile models' original quality
penalty remains.

The mapped event graphs add 18.00% to 24.93% CUDA latency. Their intervals close
only to their own instrumented total. CUPTI kernel-active durations are the
primary stage-localization evidence, with gaps and host overhead separate.
Internal SM clocks overlap across warps/blocks and include scheduling stalls;
they do not measure DRAM traffic or causal percentages. Fused attention exposes
split attention and recombination, not independent QK/softmax/PV wall times.
No profiling result was used to retune any kernel.

## Execution, retries and closeout

Both initial numerical smokes passed. Attempt 001 completed its first full
qualification and timings, then failed in the untimed internal density helper:
the helper flattened activations before calling a gate requiring rank three.
[Retry 001](infrastructure/retry-001/README.md) added a temporary batch axis
only around this diagnostic gate call. Its launcher name `profile.py` shadowed
Python's standard-library module, causing attempt 002 to fail during imports
before measurement. [Retry 002](infrastructure/retry-002/README.md) uses
`measure.py` and removes its launch directory from the import path. Original
code, failed evidence and each control snapshot remain preserved. All six
fresh processes were repeated as cohort 003; earlier measurements are excluded.
These adapters changed no production kernel, scientific input, acceptance
criterion or timed boundary. Native eager memcpy versus graph memcpy-kernel
representation is handled by the narrow local
[copy mapper](trace_copy_mapping.py), retaining original failed strict maps.

The accepted processes took 410.319 seconds in total. Their physical GPU UUID
was `GPU-ee230eab-cb21-c61a-45ee-53ae921bd72a`, with 32,607 MiB device memory.
The provider's `gpu.memory` field read 72, disagreeing with CUDA's device
capacity; CUDA's measurement was used as VRAM evidence. Setup, pinned package
download, cold compilation, transfer,
infrastructure retries and retrieval account for the longer cloud lease.

The final transfer verified all 247 files / 313,015,113 payload bytes against
SHA256 inventories. All previous attempt-001/002 inventory hashes were checked
again after final retrieval. See [receipt](transfer/receipt-003.json),
[inventory](transfer/inventory-003.json) and
[closeout](prelaunch/closeout-001.json). The Pod was terminated at approximately
22:16:50 UTC, after retrieval and local evidence checks; the scoped workstation
guard was then disarmed. The live account audit confirms the owned Pod is
absent, with only the pre-existing stopped Run052 Pod and shared volume remaining.
The 40m55.586s lease gives a provisional USD .471 GPU / .006 storage estimate,
approximately USD .477 total. Billing had no posted records at closeout, so this
is an estimate rather than a final invoice.

Local postprocessing adds three focused native-copy mapping checks to the five
accounting checks. All eight pass. The four PDF pages were rendered and visually
reviewed. The 257-test prelaunch suite and 22 on-GPU component checks had already
passed; all new mapped graphs then passed full qualification. Analysis/report
edits after retrieval are local reducers, not changes to the sealed executable.
No manuscript edit or finding promotion was made.
