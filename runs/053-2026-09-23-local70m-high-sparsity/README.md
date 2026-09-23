# Run053: local 70M calibration for high-sparsity kernel development

Status: implementing the local calibration authorized by the user's "Go on"
after WSL setup. The high-kappa-first direction was already approved. This
stage checks resource fit and frozen reference implementations; it does not
declare a new kernel winner or change the manuscript.

## Question and fixed inputs

Can the RTX 5070 Ti Laptop execute the existing full-sequence 70M workload
with headroom, and does the adapted kernel retain its strong high-kappa
response on this development device? Compare Base c00 and existing T2/Ph
checkpoints c24/c25/c26 at kappa .05/.1/.5. Prior random initialization,
seed1234, step712 weights, 1,493,172,224-token training histories, AdamW and
h-only orthogonal-L1 histories are unchanged. No training or new threshold.
T2 is operational HZ: one-sided gates at h and z, equality survives, h replaces
GELU, z is immediately before W_o. Preserve BF16 branches and residual rounding.

BF16, FP32 accumulation, B1/T2048, all50304 logits, TF32 and reduced-precision
BF16 reductions disabled. Runtime seed2801 and timing seed2504. Frozen modes:
native PyTorch, original K050-derived 70M port (`legacy`), optimized adaptation
`opt073`, its same-layout h/z skip-disabled control, and `native_hz` (the
opt073 scaffold with native dense h/z projections). Skip disabling retains
gates and weights. These are calibration references, not a complete search
over the strongest dense kernels.

The immediate smoke uses the first four blocks of the retained training
development cache, two randomized paired timing passes and one fresh process
per checkpoint. It establishes neither validation quality nor significance.
The retained cache contains64 blocks despite its older description saying
"first16"; four explicit block indices are recorded here.

The separate full-reference mode uses all500 validation documents:338 complete
blocks,692224 input tokens,691886 predicted tokens,1444 excluded tail tokens.
It retains64 timing inputs and seven passes, but only one process per cell;
it is still local calibration, not the final replicated manuscript comparison.
Neither stage selects a kernel on validation. Bounds remain per-logit
atol.25+rtol.02, per-block relativeL2<=.02 and pooled loss delta<=.001.

## Attribution and next decisions

The manuscript goal remains both moderate endpoints faster than native Base,
a larger positive .05-to-.1 latency drop than14M, and sparse contribution
beyond optimized dense execution. This run alone cannot establish that claim:
it is a separate device cohort and initially contains no matched14M matrix.
High-kappa acceleration plus a controlled skip benefit is a positive control;
failure at moderate kappa diagnoses this family/device, not impossibility.

Prioritize the high-kappa bottleneck, then reduce thresholds while keeping
.05/.1 in the evaluation loop. Any next candidate is appended with immutable
source/configuration identity before execution. Preserve original references;
do not slow .05 merely to enlarge the latency drop. No TeX changes authorized.

## Diagnostic inventory and local execution

The previously approved inventory persists: all checkpoint/cache identities,
full-validation exact/near-zero counts, RMS/L2, weight norms, row/group/tile
occupancy, h/z logical and executed-work counters, separate profiles, raw
timings, compiler evidence and numerical failures. Source checkpoints and
their existing full-validation diagnostics remain retained. The initial smoke
collects timing, correctness, memory and profiles only; it does not relabel
four blocks as full diagnostics. Training gradient interactions cannot be
reconstructed from these checkpoints.

`01_stage.py` copies an explicit hash-checked inventory into the Linux filesystem
through stdin tar, without mounting Windows drives. No prior run is modified.
`02_calibrate.py` owns each attempt, logs progress and checks GPU/runtime identity.
`03_local.py` launches one hidden persistent Windows WSL client, executes
checkpoints sequentially with a bounded timeout, and retrieves/hash-verifies
artifacts. Existing cloud resources are not touched.

Initial smoke budget: maximum20minutes, estimated5--12minutes including cold
CUDA compilation. Only smoke is launched initially. Monitor every60seconds;
warn/stop on numerical failure, less than1.5GiB free after capture, OOM,
nonfinite loss or timeout. Report measured full-reference ETC after the smoke.
No model, compiler or benchmark process remains running after this stage.

## Verification

Implementation checks and actual local smoke results will be appended after
execution. Environment readiness is recorded separately in
[tools/local_gpu](../../tools/local_gpu/README.md).

## Execution log: infrastructure retry

All242 bootstrap tests passed in24.10seconds before launch. Run-local Python
syntax and explicit smoke/full-validation coverage checks passed. Preparation
commit: `fe8ddcd4`. The first transfer verified1,287 entries, but its enumeration
had omitted vendor headers whose Windows paths exceeded260characters. The
first attempt `smoke-001-c00` completed native graph capture, then stopped
during the legacy backend's dependency verification. No performance conclusion
comes from this failure; all five returned files are hash-verified and retained.

`04_stage_longpaths.py` is the infrastructure correction; `01_stage.py` remains
as executed history. Explicit extended Windows paths recover the omitted files.
The second transfer verifies1,524 files /1,162,863,494bytes, followed by all900
entries in the frozen vendor inventory. `smoke-002` retries identical checkpoints,
gates, numerical bounds, model code and kernel sources. The first transfer
inventory and source state remain in the ignored artifact directory.

Use `python -X utf8 03_local.py status --tag smoke-002` from this directory
for Windows console status, avoiding cp1252 errors from library progress text.

## Completed local calibration and component continuation

The unchanged second smoke completed all four checkpoints in307.1seconds.
All80 graph/block numerical comparisons passed (bitwise logits on these four
training blocks). Peak PyTorch allocation was4.923GiB and peak reservation
5.502GiB with all five graph implementations resident. This establishes local
development fit; the smoke does not establish full-validation quality.
All72 output files /16,764,028bytes are retrieved and hash-verified.
See [the calibration table](results/smoke-002-table.md) and its source-hashed JSON.

The high-kappa adaptation is faster than native in this small smoke, while
the moderate endpoints remain slower. Instrumented profiles identify the
h/z fallback as a major cost at moderate thresholds. Profiling durations
are not benchmark latency. Prior Run042 already tested basic M8/M16 layout
changes, so those are not introduced as a new discovery.

The next bounded implementation step under the approved adaptive local kernel
direction is `wide_sparse.py`: two wider feature-union groups (32/64 rows,
compared with the previous family's maximum16) and two tile-support paths
(16/32 rows,16 features). The latter materializes a gated tensor and tile
support once, then skips activation/weight reads and dot operations for empty
tiles. The former compacts the active feature union and gathers only those
weights. Both rebuild metadata on every invocation and have controls that keep
that producer but force all reduction tiles. No new gate or sparsity is added.

`06_component_operators.py` checks both input widths, zero/short/mixed/dense/
boundary/empty-tile inputs, independent reduction-work counts, count-on/off
agreement and changed-input graph replay. `07_component_screen.py` then uses
training blocks0:16 and16:32 at kappa .5/.05/.1, all12 sites, five randomized
paired passes. These are development/confirmation prefixes, not an untouched
holdout. Each graph contains20 complete operator invocations to amortize WSL
submission overhead; host and CUDA times per invocation are retained. This is
component throughput evidence, not interchangeable with full-model latency.

Controls include native gated PyTorch, fused gate plus native GEMM, dense
Triton dot and the prior Run050 selected site policy. Shortlist only candidates
that pass numerics and beat the best qualified dense/prior control by at least
5% at both moderate kappas in both training prefixes. Retain high-kappa and
skip-control effects separately; integration must establish the high-kappa
benefit and both endpoint targets before promotion. A component win is not a
full-model win. Absence of a winner only rejects these tested variants.

This separate bounded component stage has a20minute maximum and an initial
5--10minute ETC. It stops if its operator checks fail or time expires. Same
checkpoints/cache and diagnostic inventory persist. New training-prefix
activation counts/moments, weight norms, row/group occupancy, reduction-work
counts, raw timings, compiler resources and numerical checks are retained;
full-validation diagnostics remain required for a future frozen candidate.
All242 bootstrap tests passed again in8.14seconds; GPU checks are pending.

## First component result and bounded output-width follow-up

`components-001` completed in330.6seconds. All96 operator cases,32 graph-input
changes and13,824 real-input numerical comparisons passed. All1,310 returned
files /46,426,614bytes are SHA256-verified, including the complete persistent
Triton cache snapshot. See [observation002](observations/002-wider-component-screen.md).
The union path improves all six high-kappa z components, but no candidate
passes the joint moderate-endpoint5% rule. No full-model winner is claimed.

The next four variations keep the union algorithm and increase the output
tile to128/256 at row groups32/64. `wider-output-config.json` and scripts10--13
retain the same controls, coverage, bounds and criterion. The scripts are
immutable copies with filename/config references changed; the derivation
manifest records both source and resulting hashes. The first screen remains
intact. Same training prefixes are explicitly reused for adaptive development.

Local follow-up budget:20minutes maximum, approximately6minutes based on the
first screen, no cloud activity. Operator failures or the timeout stop the
stage. Monitor every60seconds and retrieve/hash-verify all outputs afterward.
All242 bootstrap tests passed in8.05seconds before this follow-up. GPU operator
checks precede its real-input screen; no new full-model deployment is launched.

## Wider-output result and full-model integration

`wider-outputs-001` completed in301.8seconds. All operator and real-input
numerical checks passed. The32-row/256-output union variant passes the joint
moderate-endpoint rule at three h and five z sites. See
[observation003](observations/003-output-width-follow-up.md). All2,334 outputs
are hash-verified. The large-cache pipe stall and its infrastructure-only
retrieval correction are retained in that observation; no measurements changed.

`provenance/local-policy.json` freezes one policy from training data before
validation. `local_policy.py` integrates it without changing gates or residual
rounding. `15_full_model.py` compares native, opt073, the strongest tested local
dense policy, the strongest tested prior/dense policy, the candidate, and its
all-sparse-skip-disabled control. Base retains the native h/z fallback. All
six graph models must retain at least1.5GiB free GPU memory after capture.

`16_model_local.py` first runs four-block smokes for Base/.5/.05/.1, stopping
on failure, then runs all338 validation blocks for each checkpoint. Each final
cell retains64 timing inputs with seven randomized paired passes. Profiles
are separate from timing. `full_diagnostics.py` additionally retains full
candidate activation counts/moments, weight norms, row/group occupancy and
independently checked executed reduction tiles at all eight new sites on all
338 blocks of each T2/Ph checkpoint. Prior sparse sites have occupancy
histograms; counters are software reduction tiles, not hardware DRAM traffic.

This local integration has a20minute global cap and a10--15minute ETC. Monitor
every60seconds; stop on numerical/headroom/nonfinite/timeout failures. The
controller survives terminal disconnects. Retrieve and verify all outputs
afterward. All242 bootstrap tests passed in8.21seconds; target-device component
checks passed before this launch, and full-model smoke remains its first gate.
This is one process per checkpoint, not a replicated final14M/70M comparison.

## Full-model qualification outcome and numerical diagnosis

`model-001` stopped after430.5seconds at the numerical gate. Base, kappa .5
and .05 completed full validation and diagnostics. At .1, all338 inputs were
evaluated: only the new candidate fails, on input78's elementwise logit bound
(maximum absolute difference .5, relativeL2 .00056935). Its pooled loss delta
is -0.0000107655, within the loss bound; that does not override the failed
per-logit requirement. Native, opt073, dense, prior and skip-disabled controls
all pass. All153 available outputs /39,710,437bytes are retrieved/hash-verified.
The failed candidate is not a qualified moderate-endpoint kernel.

Scripts18--19 trace input78 through the unchanged reference and candidate to
distinguish projection roundoff from downstream gate changes. They also collect
the missing full338-block candidate inventory and separate profiles at .1;
these remain diagnostics of a failed implementation. No numerical bound,
checkpoint, threshold or implementation changes. Local diagnosis budget:
10minutes maximum, approximately1--2minutes,60second monitoring. All242 tests
pass in7.80seconds before launch. The original failure and timings remain intact.

The diagnosis completed in17.8seconds and reproduced three logit-bound
violations. The first discrepancy is one BF16 step in z.1, then downstream
gate-membership changes. Full .1 diagnostics and separate profiles are now
retained; all12 files are hash-verified. See
[observation004](observations/004-model-qualification-failure.md).

Scripts21--24 test four aligned K16 tile-skip variants, row groups16/32 and
output widths128/256, with the original controls, bounds and selection rule.
This tests whether preserving feature positions avoids the compaction path's
rounding problem while keeping the output-width improvement. Same training
prefixes, checkpoints and gates; no input-specific workaround. Local screen:
20minute cap, approximately5--6minutes,60second monitoring, operator checks
before real-input timing. Full-model qualification remains necessary.
