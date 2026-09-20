# Run047: optimized latency of the later 70M T2/Ph kappa=0.5 checkpoint

Status: complete and verified. All nine final processes qualify; all 217
returned files pass size/SHA256 checks and local reduction exactly reproduces
the remote summary. The later endpoint takes 1.214368 ms with opt073 versus
1.695806 ms with the original port, or 1.363778x its same-session native Base.
Pod `e2njhw4wy9d9uo` ran from 22:26:40 to 22:48:34 UTC on 20 September
and was deleted after retrieval. Zero Pods remain; shared storage is unchanged.
Estimated total below USD0.37 (not a settled invoice), within the approved
90-minute/USD2 cap. See [the result and limits](observations/001-final-results.md).
The frozen configuration retains its prelaunch authorization snapshot; the
actual approval, Pod identity and deadline are recorded in `prelaunch/lease-001.json`.

## Question and approved scope

Measure the missing optimized latency of Run046's final T2/Ph kappa=.5 checkpoint
with the frozen Run045 protocol. Repeat Base (c00) and T2/Ph kappa=.1 (c25) as
references for differences between GPU sessions. The new endpoint is c26.
Compare native PyTorch/SDPA, the original Run035 port, and frozen Run042 opt073
in every process. Qualified lower latency supports the optimization's benefit;
slowdown or numerical failure limits it and remains recorded. No new training,
kernel search, threshold tuning or post-hoc clipping occurs.

The user explicitly confirmed this three-checkpoint design, including the same
previously approved diagnostic and checkpoint-retention package. Model/weights,
thresholds, validation inputs, precisions, numerical bounds, timing seeds and
implementations remain fixed. Only implementation and checkpoint vary within
this new session. It does not establish a scale-dependent or quality-matched
speedup, or isolate pure skipping from dense-kernel improvements.

## Checkpoint and data identity

Six-layer Pythia-70M, d512, FFN2048, eight heads, vocabulary50304; pinned config
revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c. All checkpoints came from the
canonical random initialization and MiniPile order, model/data seeds1234,
712 optimizer updates and 1,493,172,224 input tokens. Original training used
AdamW betas(.9,.95), eps1e-8, weight decay.1, clipped task gradient norm1,
learning rate.001 to.0001 and 1% warmup. Those histories are retained;
this inference-only run performs no optimizer steps.

T2 means the HZ topology: one-sided gates at h and z, with h replacing GELU and
z immediately before the attention output projection. Keep x>=kappa and zero
smaller values. OL1 pressure targeted h with lambda=b=1. The missing checkpoint's
content SHA256 is d6a3813f83ce080b07637fe422bb7ab1b23a6793e39c365601301a661851d02f.
Its canonical loss is4.838033648 and model-wide sparsity15.42696957%.
Canonical final-validation loss/logical counts are retained, separately from
the BF16 numerical-qualification loss. No checkpoint is modified.

The full pinned MiniPile validation cache is shared with Run045: all500
documents,338 complete2048-token blocks,692224 input tokens,691886 predicted
tokens; report the excluded1444-token tail. `provenance/inputs.json` retains
source paths, checkpoint/config/content hashes, integer logical counts and
analytic ceilings. The three final models remain local throughout execution;
optimizer states and earlier checkpoints remain in their source runs.

## Matched timing and numerical protocol

One RTX5090, BF16 operands/output with FP32 accumulation, batch1, sequence2048,
causal full forward and all50304 logits. Disable TF32 and BF16 reduced-precision
reduction. Runtime seed2801, timing seed2504; same pinned Python3.12,
Torch2.11.0/CUDA12.8, Transformers5.12.1, NumPy2.5.0.
All three implementations use CUDA graphs and randomized paired timing order.
Compilation, static weight layout, capture and equal input staging are excluded;
every recurring gate, inspection pass, model operation and output head is timed.

Three fresh final processes per checkpoint give nine processes. Each measures
64 fixed validation inputs x7 passes:1344 host timings per backend/checkpoint.
Use their geometric mean; retain raw host/CUDA samples, medians, per-process
ranges and paired ratios. All338 blocks must qualify in each final process
against unmodified eager outputs: finite logits; every absolute error at most
.25+.02*abs(reference); per-sequence relativeL2<=.02; pooled loss delta<=.001.
No best-backend selection or replacement of failures is permitted.

Preflight runs each of the three checkpoints once, on eight blocks and four
timing inputs x2 passes. These are smoke checks, not final measurements.
The final controller keeps unique attempts and cannot overlap another controller.

Replicate1 retains complete338-block activation exact/near-zero counts at
0/.001/.01, RMS/L2, per-layer weight norms, integer logical opportunities,
occupancy/MMA/scalar work counters for both custom implementations, and profiles.
The original training gradient-conflict/OL1 histories remain in the source runs;
they are not reconstructed during inference. No new clipping measurements.

## Execution, retention and manuscript use

Proposed envelope: one Secure RTX5090, at most90 cumulative billable minutes
and USD2 total, including storage/retries; quote and tests go in the launch
proposal. Use30GB container and40GB persistent /workspace, no new network
volume. Logs, manifests, inputs, raw outputs and inventories persist in
/workspace. Runtime/extensions and a verified duplicate of vendor sources use
local container storage to avoid Run045's repeated metadata overhead; immutable
original sources remain on /workspace. This setup occurs before preflight and
does not change or enter the timed workload. A stopped Pod retains /workspace;
the pinned runtime can be rebuilt if recovery requires a restart.

Use a detached pipeline, on-host process deadline, independent workstation
provider-stop guard, and20-minute collection/recovery reserve. Expected ETC is
45--75min including setup, upload, compilation, full checks and verified return.
Monitor read-only every five minutes, sooner near predicted completion or after
a warning: nonfinite/failed qualification, source/runtime drift, less than8GiB
VRAM headroom, less than5GB free storage, ten-minute stale progress, or projected
deadline/cost overrun. Verify returned archive/member sizes and SHA256 locally
before deleting the run-owned Pod; confirm removal and leave shared storage alone.

After verification, a new numbered analysis will add c26 to both panels of
`23-70m-quality-sparsity-native-latency.pdf`, update tables/text and rebuild
the manuscript. The earlier26 measurements remain identified as Run045, and
the new point as this later session. Compare repeated references explicitly;
do not silently rescale new latency, replace old values, pool sessions, or call
all27 checkpoints one matched GPU sweep. Any new native-relative ratio uses
this session's measured native reference and states its denominator.

Scripts01/04/09 prepare frozen inputs and infrastructure;02/03/05 run the
unchanged benchmark through a bounded controller;06 reduces the full cohort;
07 is a real-checkpoint CPU installation smoke;08 packages all terminal evidence.
`prelaunch/` owns transfer, monitoring and provider-stop helpers. Kernel/operator
files and the measurement/diagnostic arithmetic are byte-identical to Run045.

## Local verification before launch

Seven focused protocol tests passed in0.10s: the nine-process matrix, checkpoint
and data identities, unchanged operator/measurement bytes, geometric means,
complete paired full-logit cells, per-block/pooled-loss qualification and rejection
of incomplete cohorts. The full bootstrap suite passed242 tests in8.21s.
Three real CPU checkpoint smokes passed for c00/c25/c26: finite1x4x50304 native
outputs, correct topology and h/z thresholds, and successful installation of both
frozen custom implementations. They do not execute CUDA kernels or establish
performance. All24 root/prelaunch Python files parse; both shell scripts and the
PowerShell provider guard pass syntax checks. The live CLI authentication works.

`01_prepare.py verify` reconciles1467 retained input/source copies and all three
checkpoint identities. `prelaunch/local-checks.json` records exact commands and
resource scope; `prelaunch/launch-proposal.md` records the live quote, bounded
execution and recovery inventory. No GPU has been allocated or measured yet.
