# Run069: the head efficiency penalty is partly fixable

The controlled test reduces the no-skip projection penalty from about 65% to 27-28% versus native PyTorch/cuBLAS.
The best sparse implementations beat the matched gated native full forward by 1.86% and 3.05%. These are modest
implementation gains on already head-gated checkpoints; the existing quality penalties remain. The strongest
14M body/frontier was not integrated or remeasured. No new practical quality-latency frontier win is established.

## Qualified final measurements

All rows use the same trained gates and operands. Native executes dense GEMM on gated H_L; custom no-skip also
uses gated H_L and merely disables MMA bypass. It is not an ungated model. h/z gates stay 8x16 at kappa 0;
head-input gates stay 16x16 at kappa 0.1/0.5, with the retained Ph+head training history and no optimizer updates.

| Head kappa | Native projection us | Current sparse us | Best sparse projection us | Best sparse mode | Native full ms | Best full ms | Full latency reduction vs matched native |
|---:|---:|---:|---:|---|---:|---:|---:|
| 0.1 | 192.579 | 199.857 | 169.459 | reuse2_n256/sparse | 0.998643 | 0.980100 | 1.86% |
| 0.5 | 192.982 | 160.785 | 157.091 | prefetch_n128/sparse | 0.998882 | 0.968456 | 3.05% |

Times are host-wall geometric means over three equal-size fresh processes. Each process retains 448 paired
samples/mode/boundary (64 inputs x 7 passes); CUDA-event times and process ranges remain in comparison.json.
The best final sparse row is identified within the screen-selected shortlist; it is not a prespecified universal winner.
Projection, gate/cast/projection and full-forward measurements overlap and cannot be summed. See the complete
[final table](REPORT.md) and [qualified-boundary figure](../figures/01-qualified-boundaries.pdf).

## What the ablations establish

- Current N256 no-skip takes 318.1/318.2 us versus native 192.6/193.0 us. Reuse2 N128 lowers this to 246.8/245.5 us.
  The gap shrinks substantially but remains 27-28%; the h/z-style goal of a near-native no-skip path is not reached.
- At fixed N128 in the exploratory screen, the two-row-tile variant cuts no-skip latency from 377.3 to 250.1 us
  (kappa 0.1) and 373.5 to 246.2 us (kappa 0.5), about 34%. Its intervention includes weight reuse, doubled row work
  per block and half as many blocks at that width. These effects cannot be causally separated by this ablation.
- Width tuning alone helps modestly: legacy N512 is about 4.5% faster than N256 without skipping. N128 alone is slower.
- Direct stores remove the shared-memory epilogue but make no-skip about 51% slower than current N256.
  This refutes the proposed direct-store implementation as a remedy. Source addresses suggest less efficient
  global stores; transactions/stall attribution were not measured.
- Prefetch N256 is slower without skipping. Prefetch N128 barely improves its corresponding legacy screen,
  although its sparse path wins the kappa 0.5 final shortlist by a small margin. It does not solve the dense deficit.
- All inspected custom binaries report STACK0/LOCAL0. New ptxas logs report zero spill stores/loads.
  Register spilling is not supported as the explanation for these binaries. Register/shared allocations vary;
  achieved occupancy and memory stalls remain unknown.

The [all-variant figure](../figures/02-exploratory-screen.pdf) is exploratory: one process per checkpoint,
64 real operands and 448 paired samples/mode. Only the selected final modes qualify over all 338 blocks in three processes.

## Source-level work and retained support

| Head kappa | Empty head tiles / MMA bypass | Reuse2 weight-request reduction vs sparse M16 at same N |
|---:|---:|---:|
| 0.1 | 39.64% | 38.27% |
| 0.5 | 70.47% | 35.67% |

The percentages pool integer counts over the 64 timing operands. Actual untimed Count specializations confirm
MMA bypass. The 16x16 input mask is shared with the trained gate; reuse2 preserves each row tile independently.
Its union mask governs only a shared weight fetch. Weight-request reductions above follow source control flow
and retained certificates; they are not measured DRAM traffic or runtime speedups.

Without skipping, M16 kernels request the full 12.878 MB weight matrix once per 128 row tiles; M32 reuse requests
it once per 64 row pairs: about 1.648 GB versus 0.824 GB of requested B operands per call. At the same N128,
CTA count halves (50,304 to 25,152) and input-request count is unchanged. Comparing reuse2 N128 against current
M16 N256 also changes column width: total CTA count is nearly unchanged and input requests increase.
These are algorithmic requests serviced by caches as well as DRAM; hardware traffic was not measured.

Every implementation still writes 206,045,184 output bytes (196.5 MiB) per call. Removing feature products cannot
remove the required dense logit tensor. The synthetic zero-MMA current-kernel diagnostic still takes about 136 us
host time, with full output/grid work. This is not a true runtime lower bound, a quality result, or an exact
decomposition of compute and stores. No claim that 70% bypass implies 70% latency reduction is justified.

## Forward and kernel step map

| Step | Scope and evidence |
|---|---|
| Transformer body + affine final LayerNorm | Native dense execution with existing h/z gates. Included only in full-forward timing. |
| Head-input gate | FP32 upper-left certificate per 16x16 token/feature tile; strict value > kappa; rejected tile becomes zero. Same trained rule for every mode. |
| Cast / support preparation | FP16 input and FP32 certificate/support handling. Included in gate/cast/projection and full forward, outside projection-only. |
| Projection main loop | Read row-tile masks; load input/weight fragments; execute MMA on every K group or only live groups. Reuse2 shares B loads across independent row tiles. |
| Projection epilogue | Convert FP32 accumulators to FP16 and write every logit; shared/vectorized stores or the direct-store ablation. Included in projection timing. |
| Softmax / loss / token choice | Outside all latency boundaries. Loss is computed separately for numerical qualification. |
| Static weight casts/packing and input staging | Outside measured latency. Retained source operands and packed implementation identities permit replay. |

The main loop and epilogue were not timed independently. Nsight Compute failed with ERR_NVGPUCTRPERM even in
the cloud root process. Therefore this run cannot assign latency percentages to loads, MMA, stores or stalls.
The evidence identifies a beneficial implementation family, not a complete hardware-cycle decomposition.

## Qualification, limits and closeout

All 14,196 mode/block comparisons are bitwise exact against the native anchor. Each of six fresh processes covers
all 500 validation documents through 338 complete 2048-token blocks: 692,224 inputs, 691,886 predictions, 1444 excluded
tail tokens. Unchanged bounds were never relaxed. Qualification B1 losses are 5.560689779 and 5.882653701 for every
mode; source canonical losses 5.560725 and 5.882600 retain their original evaluation provenance and are not replaced.
160 cloud component/replay checks also pass. One GPU UUID, two checkpoints, one training seed and one workload
limit generalization. Process ranges are not confidence intervals. Existing head quality penalties are unchanged.

The most supported next engineering target is more effective row/weight reuse while controlling registers and
occupancy. This run does not test larger row groups or establish that they would produce a large practical gain.
No new training, kernel promotion, larger-model result or manuscript claim is introduced.

74 remote files (199,566,775 bytes) were copied and hash-verified locally before Pod d9ryfnu2zhri2w was terminated.
Account inventory confirms no new billable resource remains; the foreign stopped Pod/shared volume are untouched.
Creation-to-confirmation upper bound implies about USD 0.94 compute plus USD 0.01 storage. Billing records had not
posted at closeout; this is a provisional upper-bound estimate, not a zero cost or final invoice.

Sources: [diagnostic data](diagnostic.json), [process reduction](comparison.json), [verification](verification.json),
[raw endpoints](../artifacts/endpoints/), [screen](../artifacts/screen/),
[compiler/binary resources](../prelaunch/resources-cloud.json), [cloud pipeline log](../artifacts/control/cloud-pipeline.txt),
[profiler failure](../artifacts/profiles/manifest.json), [closeout](../prelaunch/cloud-001/closeout.json).
Generator: [13_diagnose.py](../13_diagnose.py).
