# Run 014 - Pythia-14M paper-scale A7-OL1 full pass

## Status and authorization

Design and end-to-end autonomous execution were explicitly pre-approved by the
user on 2026-08-31, including implementation, tests, billable RunPod preflight,
five-GPU scientific launch, monitoring, artifact retrieval, teardown, and
closeout. The evidence gates below therefore do not pause for additional
permission. This folder is the next free run number recorded by
`research/INDEX.md`.

Execution and cloud closeout are complete. Five condition-parallel scientific
attempts completed, were verified remotely, transferred with matching archive
hashes, safely extracted, and independently verified locally. The cohort
verifier reports valid evidence for all 3,560 optimizer steps and 20 complete
validation passes. All RunPod Pods were deleted; the pre-existing volume is
unchanged.

Before launch, the focused Run 014 suite passed 8/8, the affected Run
010/012/013/014 suite passed 33/33, the full bootstrap suite passed 174/174,
and all Python files compiled. Bash is not installed on the Windows controller,
so `00_setup_remote.sh` received its syntax/execution check on the Linux
preflight Pod.

## Question and hypothesis

This run completes the paper-control `A7-OL1` cell and paired comparison P08:
does operational OL1 pressure at all seven active A7 sites improve the
paper-scale A7 threshold quality--logical-opportunity frontier relative to Run
013 at matched `kappa`?

At a matched threshold, support means lower complete-validation loss at
comparable `R_model`, higher `R_model` at comparable loss, or a new
nondominated point. The hypothesis is weakened or refuted if all A7-OL1 rows
are dominated by Run 013, if selected-site sparsity and `R_model` do not move,
if numerical stability degrades, or if strong gates remove enough pressure
gradient support that OL1 becomes ineffective.

## Conditions and matched comparison

Five independent conditions use operational topology
`A7-Z-POST = {a,m,h,q_post,k_post,v,z}` in all six layers. One common `kappa`
is used within each condition. Equality survives both gates.

- `a,m,h,z`: one-sided `x` when `x >= kappa`, otherwise exact zero.
- `q_post,k_post,v`: symmetric `x` when `abs(x) >= kappa`, otherwise exact
  zero.
- `q_post` and `k_post` are the operands after partial RoPE.
- `z` is concatenated PV context immediately before `W_o`.
- OL1 targets the post-gate outputs at all seven sites. Its scalar is the
  unweighted mean of 42 tensor means: seven sites by six layers.

| Order | Condition | `kappa` | Pressure |
| ---: | --- | ---: | --- |
| 1 | `a7-ol1-kappa-0` | 0 | OL1, `lambda=1`, budget `1` |
| 2 | `a7-ol1-kappa-0p01` | 0.01 | OL1, `lambda=1`, budget `1` |
| 3 | `a7-ol1-kappa-0p05` | 0.05 | OL1, `lambda=1`, budget `1` |
| 4 | `a7-ol1-kappa-0p1` | 0.1 | OL1, `lambda=1`, budget `1` |
| 5 | `a7-ol1-kappa-0p5` | 0.5 | OL1, `lambda=1`, budget `1` |

Run 013 supplies the five primary no-pressure comparators. Initialization,
realized data order, global batch, horizon, optimizer, LR schedule, precision,
validation cache, diagnostic implementation, checkpoint boundaries, topology,
gate families, and threshold are matched. The only scientific delta is adding
post-gate OL1 at all seven active sites with fixed weight and trust budget.
Run 010 is pilot precedent only: it used seed 0, BF16, global batch 64, 581
boundaries, and another decomposition, so it is not pooled with this run.

## Verified result

| `kappa` | A7-OL1 loss | A7-OL1 `R_model` | A7 loss | A7 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.480184 | 7.0542% | 5.468401 | 7.2177% | +0.011783 | -0.1634 pp |
| 0.01 | 5.475797 | 7.7234% | 5.458822 | 7.6176% | +0.016975 | +0.1057 pp |
| 0.05 | 5.462811 | 9.8630% | 5.437888 | 9.1269% | +0.024923 | +0.7361 pp |
| 0.1 | 5.429497 | 11.7968% | 5.428681 | 10.4250% | +0.000816 | +1.3717 pp |
| 0.5 | 5.829390 | 27.4827% | 5.702923 | 15.3868% | +0.126466 | +12.0959 pp |

The hypothesis is supported only in a threshold-dependent form. At
`kappa=0.1`, OL1 adds 1.3717 percentage points of logical opportunity for
`+0.000816` validation loss. At `kappa=0.5`, it adds 12.0959 points for a
larger `+0.126466` loss. Those two A7-OL1 endpoints are new nondominated points
among the ten matched Run 013/014 endpoints. The zero-threshold OL1 row instead
loses 0.1634 points and has higher loss, refuting a blanket claim that OL1
improves every A7 threshold.

All five conditions realized the required 42 pressure tensors at every
boundary, completed without overflow or skipped updates, and respected the
OL1 trust budget. The full count-reconciled result, boundary summary, caveats,
and evidence sources are in
`observations/O001-paper-scale-a7-ol1-versus-a7.md`. This candidate observation
is not a promoted finding or manuscript claim.

## Model, initialization, data, budget, and optimizer

- Pythia-14M architecture config revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`, constructed from config with
  random weights; released weights are not loaded.
- Pythia `small_init` ordinary weights and `wang_init` residual-output
  projections; model seed `1234` and data-order seed `1234`.
- MiniPile revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`, Pythia
  tokenizer at the model revision, append EOS, no encode-time special tokens.
- Training cache: 1,000,000 documents, 1,491,711,416 tokens, 728,374 complete
  blocks, 1,464-token tail, SHA-256
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.
- Sequence length 2,048; global batch 1,024 as MB32/GAS32; 712 optimizer
  boundaries, 729,088 scheduled blocks, and 1,493,172,224 input tokens per
  condition. The cohort totals 3,560 boundaries and 7,465,861,120 tokens.
- FP32 parameters and AdamW state; dynamic FP16 CUDA autocast; Flash SDPA;
  zero dropout; no activation checkpointing.
- Fused AdamW with betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, and
  no decay on biases or LayerNorm. Peak/minimum LR are `1e-3`/`1e-4` with 1%
  warmup and GPT-NeoX-v1 pre-step cosine semantics.
- The globally clipped task gradient alone updates AdamW and its moments. OL1
  preconditions the unclipped pressure gradient with the task second moment,
  removes only a globally conflicting component, caps the weighted
  pressure/task direction ratio at `1.0`, and applies the correction after
  AdamW. A nonfinite component atomically skips the entire boundary; valid
  completion requires zero skipped boundaries.

Expected initial-parameter SHA-256 is
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`;
expected schedule SHA-256 is
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

## Validation, diagnostics, and post-hoc retention

Every ordinary or diagnostic pass covers all 500 validation documents, all
338 complete 2,048-token sequences, and 692,224 input tokens. The 1,444-token
tail is excluded and reported. Ordinary validation runs after boundary 1 and
from the reloaded final checkpoint; activation and eager logical-product passes
use the same complete workload.

Every boundary retains task/pressure losses; raw task/pressure gradient norms,
dot, cosine, and conflict; task clipping; loss-scale and overflow state; OL1
adaptive directions, projection, raw ratio, trust scale, and final ratio; LR,
throughput, elapsed time, and peak CUDA memory. The boundary validates the
realized 42-name capture set after every microbatch and records its count and
SHA-256. This closes the gap between a declared pressure set and the objective
actually differentiated.

The reloaded final checkpoint records count-first exact-zero and near-zero
counts at `0`, `1e-3`, and `1e-2`, sums, RMS/L2 moments, and nonfinite counts
for `a,m,h,q_post,k_post,v,z,attention_output`; all named parameter norms
including bias and normalization; and all six actual-operand logical-product
families. The A7-Z-POST analytic ceiling is retained as integer counts:
`5,638,717,440 / 18,825,609,216 = 29.9523769738%`.

Model snapshots are retained at boundaries
`0,1,2,4,8,16,32,64,128,256,512,712`; optimizer, scaler, and RNG recovery
state are retained at `256,512,712`, including the final checkpoint. No
clipping frontier, histogram set, or qualitative predictions are included.
Gradient interaction is collected now because it cannot be reconstructed from
a checkpoint. `R_block`, `R_model`, and `R_model_max` are logical-product
opportunities, not removed FLOPs or measured speedup.

## Material prior-run discrepancy

Inspection during design found that Run 012 declares pressure at `a,m,h,z` but
its reused training loop calls `ActivationCapture(model, ["h"])`. Its config,
manifest, and verifier therefore overstate the realized pressure set. Run 014
does not reuse that behavior: its adapter captures all seven configured sites,
and the optimizer boundary plus verifier require the 42 realized tensor names.
This discrepancy does not invalidate the direct Run 013 versus Run 014 P08
comparison, but it limits future P09/P10 use of Run 012 until separately
audited or rerun.

## Manuscript relationship and interpretation limits

This executes the paper-control Pythia-14M `A7-OL1` cell, P08, the missing
pressure half of P12, and the A7 endpoint needed for P16. It uses the accepted
operational mapping: one-sided `a,m,h,z`, symmetric post-RoPE Q/K/V, and OL1 at
all seven post-gate outputs. The operational baseline clips the task gradient
before AdamW/OL1 geometry, and the positive trust budget is mandatory. No TeX
or finding is changed automatically; a cross-run synthesis belongs in a later
numbered analysis and a manuscript claim still requires user approval.

Limits are one seed, one 14M scale, one MiniPile pass, and joint pressure at
seven sites. The run cannot attribute an effect to one site. Independently
scheduled GPUs are matched but not assumed bitwise identical. Strong gates
alter both forward values and gradient support. Logical opportunity is not a
hardware-speed claim.

## RunPod execution, live price, transfer, monitoring, and teardown

The 2026-08-31 preflight inventory reports zero Pods and one pre-existing
retained 100 GB Standard volume `9luykg5yc3` in `EUR-IS-1`. The volume already
bills independently at about `$0.01/hour`; Run 014 will not create, resize, or
delete it. Secure `NVIDIA A100-SXM4-80GB` with CUDA 12.8 has low live stock at
`$1.59/GPU-hour`; A100 PCIe CUDA-12.8 stock is unavailable.

One guarded Secure A100 SXM 80 GB Pod first runs five exact MB32/GAS32
boundaries at both threshold extremes. It must verify source, cache,
initialization, schedule, runtime, Flash attention, finite non-skipped
boundaries, the 42-tensor pressure identity, OL1 trust ratios, and at least 10%
device-memory headroom. Its absolute guard is 1.5 hours. After a pass, five
fresh one-GPU Pods run concurrently, one condition per GPU, each with a
2.5-hour absolute guard. This is condition parallelism, not DDP, and preserves
the scientific global batch.

The maximum compute envelope is `(1.5 + 5*2.5) * $1.59 = $22.26`; temporary
Pod storage allowance is `$0.30`, for a rounded `$22.56` Run 014 maximum. The
existing volume cost is unchanged and excluded. The pinned image is
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Each Pod has 30 GB container disk and a 25 GB Pod volume mounted at
`/workspace`. No service port is exposed; SSH is the control channel.

Source, exact caches, environments, logs, attempts, and checkpoints remain
under `/workspace`. A compatible `EUR-IS-1` preflight may read the retained
cache only after metadata/byte/hash verification; otherwise each worker gets a
hash-checked cache copy. Scientific jobs launch with `setsid` and persistent
PID/stdout/stderr records.

Monitoring is read-only every five minutes, shortened to the refreshed
completion window. Every update reports step, input tokens, task and pressure
loss, throughput, ETC, loss scale/overflow, OL1 ratio, GPU memory/utilization,
disk, process state, and event age. Warnings are a missing process, ten-minute
stale event, nonfinite or skipped boundary, capture-identity mismatch, trust
ratio over budget, memory beyond preflight, less than 5 GB free disk,
incomplete validation, or projected deadline overrun.

Each terminal worker is verified remotely, inventoried by relative path, byte
count, and SHA-256, archived, copied locally, re-hashed, safely extracted, and
independently verified before its exact Pod is deleted. The cohort verifier
runs only after all five attempts are local. Closeout re-lists all Pods and
volumes, confirms zero unintended compute, records the unchanged retained
volume, and separates lagging posted billing from authoritative resource
teardown.

## Actual RunPod closeout

The exact A100 preflight passed at 67,851,255,808 bytes reserved on an
85,093,777,408-byte device and verified the 42-tensor capture identity. One
empty Pod was deleted when source-transfer approval paused provisioning. A
second preflight Pod completed and was deleted after its evidence was retrieved.
One planned `kappa=0` scientific Pod was deleted before any scientific attempt
because dependency installation remained in one phase for more than 27
minutes; its replacement completed the unchanged condition. This was an
infrastructure retry, not a new scientific run.

All five accepted attempt archives are 1,014,855,680 bytes. Their remote and
local SHA-256 values match, their member paths were safe, their internal
transfer/checkpoint inventories reconcile, and both remote and local
standalone verification passed. `artifacts/verification.json` then verified
the complete cohort. `prelaunch/scientific-transfer-closeout.json` records the
per-Pod, archive, log, readiness, billing, and teardown provenance.

At the final 2026-08-31 14:19 UTC audit, RunPod listed zero Pods and the one
unchanged pre-existing 100 GB Standard volume `9luykg5yc3`. Posted billing was
settled at `$14.1285` for the same eight Run 014 Pod identities: `$14.0587`
GPU plus `$0.06985` temporary Pod disk. This is below the approved `$22.56`
ceiling. There is no continuing compute charge; the pre-existing volume
continues its independent approximately `$0.01/hour` charge.

Closeout verification passed the focused Run 014 suite 8/8 and the complete
bootstrap suite 174/174. The first full-suite invocation encountered seven
Windows fixture-setup errors because the shared `.pytest_tmp/default` directory
could not be cleaned; it had no assertion failures. Repeating the same 174
tests with a fresh run-specific `--basetemp` passed completely.

## 2026-09-01 billing refresh

The Status Report Number 2 RunPod REST v2 audit reconciled the eight Run 014
Pod IDs to `$19.3236626610` GPU plus `$0.0956597279` temporary Pod-disk spend,
`$19.4193223889` total. This current Pod-ID total supersedes the earlier value
described as settled, because additional hourly buckets posted afterward. It
excludes the separately retained network volume.
