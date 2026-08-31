# Run 014 - Pythia-14M paper-scale A7-OL1 full pass

## Status and authorization

Design and end-to-end autonomous execution were explicitly pre-approved by the
user on 2026-08-31, including implementation, tests, billable RunPod preflight,
five-GPU scientific launch, monitoring, artifact retrieval, teardown, and
closeout. The evidence gates below therefore do not pause for additional
permission. This folder is the next free run number recorded by
`research/INDEX.md`.

Implementation and non-billable verification are complete. The focused Run
014 suite passes 8/8, the affected Run 010/012/013/014 suite passes 33/33,
the full bootstrap suite passes 174/174, and all Python files compile. Bash is
not installed on the Windows controller, so `00_setup_remote.sh` receives its
syntax/execution check on the Linux preflight Pod. No scientific attempt has
started yet.

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
