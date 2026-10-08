# Run067 — sparse vocabulary-head consumer

Status: completed and verified on one RTX 5090; owned Pod deleted.
The user confirmed the design and separately approved the launch on 8 October 2026.
The kernel preserves logits and loss exactly, but is slower than dense cuBLAS.
See the [latency table](results/REPORT.md), [quality and kernel diagnosis](results/DIAGNOSTICS.md),
and [figure with process means](figures/01-sparse-vocabulary-head.pdf).

Question: does a consumer of Run066's structured head-input zeros reduce head
and complete forward latency without changing trained model semantics?

## Frozen scientific inputs

All four Run066 step712 14M checkpoints are included: kappa 0 and .1 crossed
with Ph and Ph+head pressure histories. No training or optimizer executes.
The original random initialization, seed1234 model/data order, training schedule,
orthogonal_l1 pressure histories and all retained checkpoints stay in Run066.
The model has six layers, D128, FFN512, four attention heads and V50304.
Parameters remain FP32; inference uses FP16 autocast and native causal SDPA.
The h/z block gates and all transformer operations remain matched.
Those block projections execute native dense kernels in all three modes. Only
the custom sparse vocabulary head skips work; this run does not integrate the
earlier BF16 h/z consumer into the FP16 production path.

The head input is after learned affine final LayerNorm. Its unchanged gate
rejects a sequence-local 8-token x16-feature tile iff the native FP32 top-left
value is <=kappa. Kept tiles preserve all signed interiors. Weights and output
logits have no gate. All 50304 scores for all 2048 positions are computed;
vocabulary softmax, token selection, loss, staging and setup are outside timing.

Run066's B32 production losses/densities are provenance. This B1 benchmark
requalifies every checkpoint against its own unmodified B1 eager FP16/SDPA
reference; it does not assume batch-dependent tile decisions match B32.

## Kernel and controls

- Dense: native cuBLAS projection with an explicit static FP16 weight cast.
- Custom no-skip: the custom preparation/projection path with tile skipping off.
- Custom sparse: the same path, skipping empty reduction groups before weight
  loads and tensor-core instructions.

All modes preserve FP32 checkpoint parameters and use the same FP16 weight cast
outside timing. The custom path additionally permutes those same FP16 values
into contiguous MMA fragments outside timing; packing time/bytes are reported.
Native FP32 post-gate corners certify support before activation conversion,
including positive surviving corners that underflow in FP16. The recurring
activation cast/certificate kernel is included in the primary forward and head
path timings. All logits are written, including fully empty input groups.

The primary consumer uses M16/N256/K16 with four warps. Two independently gated
M8/K16 tiles occupy one M16 instruction. A K16 contribution is skipped only when
both M8 tiles are empty. Thus actual consumer bypass is based on 16x16 occupancy,
not the larger 8x16 empty fraction; both are measured and labeled. M8 consumers
and N128/N512 are component-check options, not extra scientific conditions or
an unreported tuning search. FP32 accumulation proceeds in ascending K order.
Output stores are reordered through padded shared memory for contiguous writes.

## Numerical and timing qualification

The original eager model remains the reference. Transformers 5.12 changes its
causal-mask creation during CUDA capture and consequently SDPA arithmetic.
The initial native graph itself failed before a sparse comparison. The small
`forward.py` map calls the unchanged decoder layers with the same None mask
used by unpadded native eager inference. It preserves native causal SDPA in
capture. This map, dense head preparation and custom execution must all qualify
against the original model. Earlier failures remain in prelaunch evidence.

Every full graph covers all500 MiniPile validation documents:338 complete
2048-token blocks,692224 input tokens,691886 predictions;1444 tail tokens are
excluded. Numerical bounds remain atol.25/rtol.02 per logit, relativeL2<=.02,
and pooled loss delta<=.001. Exact equality is additionally reported.
Component tests cover threshold equality, FP32-to-FP16 underflow, signed
interiors, empty/live tiles, vocabulary tails, integer work counts and changed
input replay. Standalone head graphs also qualify on every timed real operand.

One physical RTX5090 runs all twelve full processes:three fresh processes per
model,64 fixed validation inputs x7 randomized paired passes. Runtime seed2801,
timing seed2504. Separate synchronized-host and CUDA-event replays measure
full-forward, gate/cast/projection, and projection-only boundaries. Only the
first two include recurring input preparation. Component times are separately
measured, not additive decomposition estimates. Local laptop timing is development
evidence and cannot establish RTX5090 gains.

Retain raw paired timing rows, qualification/errors/losses, input indices,
native exact/near-zero counts at0/.001/.01, RMS/L2 accumulators, 8x16/16x16
occupancy, independent actual/padded MMA counts, kernel/source/cache/checkpoint
identities, setup time, memory, logs and all attempt failures. Source Run066
retains final LayerNorm values, weight norms and gradient/OL1 diagnostics;
no new gradients or clipping sweep is applicable to this frozen inference run.
Head logical counts remain separate from existing block-only R_model and its
analytic ceiling. No manuscript edit or finding promotion is included.

A stable reduction against cuBLAS supports a practical head advantage here.
The no-skip comparison measures the structure-dependent benefit within the
custom implementation. A numerical failure disqualifies a path; no reduction
against cuBLAS refutes practical acceleration for this implementation/workload.
No cached decoding, larger-model, broader quality or measured DRAM claim follows.

## Operation

`01_prepare.py` seals endpoint/runtime/cache bytes. `04_local.py` performs WSL
development qualification. After separate cloud approval, `00_setup_remote.sh`
pins the environment and detached `03_execute.sh` runs component checks,
four eight-block smokes and the twelve full processes, stopping on failure.
`05_verify.py` checks the complete cohort, `06_seal.py` creates the return
inventory/archive, and `08_retrieve.py` verifies every copied member before
teardown. A scoped workstation `07_stop_guard.ps1` stops the owned Pod at the
flexible deadline while preserving its volume for retrieval. Launch, runtime,
transfer, cleanup and numerical scope belong to this run; see `LAUNCH.md` once
final local qualification and current catalog checks are complete.

## Local qualification evidence

The full local cohort `005` qualifies all four checkpoints, each with three
full graphs on all338 blocks:4056 full-logit comparisons. Every comparison has
zero absolute/relative error and exactly equal pooled losses. The six head
graphs per checkpoint also match exactly on all64 real timing inputs:1536
component comparisons. Peak allocated memory is1,710,866,432 bytes (1.593GiB).
The four qualification processes take86.36–89.60 seconds each, including
diagnostics and development timings. These timings are not RTX5090 results.

Final tag006 repeats48 CUDA component checks, all four eight-block full smokes,
and their real-operand head checks after setup/metadata-only additions. Kernel
CUDA, forward mapping and numerical limits are unchanged from full tag005.
Final CPU checks:242 bootstrap+12 focused=254 passed in8.24s. All16 Python
sources, both shell scripts and the PowerShell stop guard parse. The upload
bundle has69 members,228,325,386 uncompressed bytes and209,386,776 archive bytes;
all scientific input/source identities are verified.

Failures are retained:001/002 expose the native captured-mask discrepancy;
003 passes the first smoke then its controller rejects the underscore in a
Ph+head attempt name;004 fully qualifies the first checkpoint before the same
controller-name issue. Both issues are fixed;005 completes all four full
qualifications and006 completes the final smokes. See the approved
[launch definition](LAUNCH.md) and [local checks](prelaunch/checks.json).

## Verified cloud execution and result

Cloud cohort `001` completed all 48 component checks, four eight-block smokes,
and twelve full processes (three per checkpoint). All 12,168 full-graph logit
comparisons and 4,608 real-operand standalone head comparisons match exactly.
Pooled loss deltas are zero. All processes used the same physical GPU UUID and
the byte-verified source at commit `72ff70aba176da41a5af022d112f255d34c4222e`.

The custom sparse head saves 11.43-16.79% versus its own no-skip control, but
is 35.37-44.55% slower than dense cuBLAS. Full-forward latency increases
7.19-8.86%, consistently across all three processes. The custom no-skip head
already costs about 63% more than cuBLAS. Empty 8x16 gate tiles reach
17.34-27.42%, but only 7.07-15.92% of the paired 16x16 consumer tiles are empty.
Thus skipping saves work without recovering the baseline custom execution gap.
The separate CUDA-event replays support the same conclusion.

All 163 scientific files and eight supplemental delivery files were copied
and hash-verified before deletion. [Teardown](ops/teardown-001.json) confirms
no new Pod or network volume remains; existing foreign resources are unchanged.
The scoped two-hour stop guard was disarmed. The initial placement lost a
capacity race; the next Secure request succeeded at USD 1.19/hour. A slow
legacy-SCP upload was replaced by Runpodctl/croc, preserving the sealed bytes.
CUDA 12.8 existed outside the initial PATH and was selected by the run wrapper.
Scientific inputs and kernel sources were unchanged throughout execution.

For reproduction: `20_analyze.py --tag 001` reduces verified paired records;
`21_figure.py` creates the initial plot and observation; `22_diagnostic_report.py`
adds pooled statistics and numerical interpretation; `23_figure.py` generates
the final visually checked layout with process means. The sealed prelaunch
scripts remain intact; the two final reporting scripts were added after execution.
The [observation](observations/01-sparse-vocabulary-head.md) retains coverage,
legend, aggregation, limitations and all generating sources. No manuscript edit
or finding promotion is included.
