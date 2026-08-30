# Run 012 - Pythia-14M paper-scale A4-OL1 full pass

## Status and approval

The user requested this run on 2026-08-30 and explicitly pre-approved the
scientific design, implementation, RunPod preflight, scientific launch,
monitoring, artifact retrieval, and teardown without waiting at the usual two
confirmation gates. The design below is therefore locked before implementation
and billable execution, but approval is not evidence that any condition ran.

## Question and hypothesis

This run asks whether operational OL1 pressure at every active A4 site changes
the paper-scale A4 threshold quality--logical-opportunity frontier relative to
Run 011's threshold-only conditions at the same `kappa`.

At a matched threshold, support means lower complete-validation loss at
comparable `R_model`, higher `R_model` at comparable loss, or a newly
nondominated point. The hypothesis is weakened or refuted if all A4-OL1 rows are
dominated by Run 011, if selected-site sparsity and `R_model` do not move, if
quality or numerical stability degrades materially, or if thresholding removes
so much pressure-gradient support that OL1 becomes ineffective.

## Conditions and matched comparison

Five independent conditions use operational topology `A4-Z = {a,m,h,z}` in all
six layers. Every site applies the same one-sided gate, keeping `x` when
`x >= kappa` and returning exact zero otherwise; equality survives. OL1 targets
the gated outputs at all four active sites, using the unweighted mean of the 24
site-layer tensor means.

| Order | Condition | `kappa` | Pressure |
| ---: | --- | ---: | --- |
| 1 | `a4z-ol1-kappa-0` | 0 | OL1, `lambda=1`, budget `1` |
| 2 | `a4z-ol1-kappa-0p01` | 0.01 | OL1, `lambda=1`, budget `1` |
| 3 | `a4z-ol1-kappa-0p05` | 0.05 | OL1, `lambda=1`, budget `1` |
| 4 | `a4z-ol1-kappa-0p1` | 0.1 | OL1, `lambda=1`, budget `1` |
| 5 | `a4z-ol1-kappa-0p5` | 0.5 | OL1, `lambda=1`, budget `1` |

The primary comparator is Run 011 at each matched threshold. Initialization,
realized data order, global batch, horizon, optimizer, LR schedule, precision,
validation cache, diagnostic implementation, checkpoint boundaries, active
sites, gate operator, and threshold remain matched. The only scientific delta
is adding post-gate OL1 pressure at `a,m,h,z` with fixed weight and trust budget.
Run 007 is a pilot precedent only: it used seed 0, BF16, global batch 64, 581
updates, and a different batch decomposition, so its measurements are not
pooled with this run.

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
- FP32 parameters and optimizer state; dynamic FP16 CUDA autocast; Flash SDPA;
  zero dropout; no activation checkpointing.
- Fused AdamW with betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, and
  zero decay for biases and LayerNorm. Peak/minimum LR are `1e-3`/`1e-4` with
  one-percent warmup and GPT-NeoX-v1 pre-step cosine semantics.
- The task gradient is globally clipped to norm `1.0` and alone updates AdamW
  and its moments. OL1 preconditions the unclipped pressure gradient with the
  task second moment, removes only a globally conflicting component, caps the
  weighted pressure/task direction ratio at `1.0`, and applies its correction
  after AdamW. A nonfinite component atomically skips the entire boundary;
  valid completion requires zero skipped boundaries.

The expected initial-parameter SHA-256 is
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57` and
the expected schedule SHA-256 is
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

## Validation, diagnostics, and post-hoc retention

Every ordinary or diagnostic validation pass covers all 500 documents, all 338
complete 2,048-token sequences, and 692,224 input tokens. The 1,444-token tail
is excluded and reported. Ordinary validation runs after boundary 1 and from
the reloaded final checkpoint; activation and eager logical-product passes use
the same complete workload.

Every boundary retains task/pressure losses; raw task/pressure gradient norms,
dot, cosine, and conflict; task clipping; dynamic-loss-scale and overflow
state; OL1 adaptive directions, projection, raw ratio, trust scale, and final
ratio; LR, throughput, elapsed time, and peak CUDA memory. These interaction
metrics are collected during training because checkpoints cannot reconstruct
them.

The reloaded final checkpoint records count-first exact-zero and near-zero
counts at `0`, `1e-3`, and `1e-2`, sums, RMS/L2 moments, and nonfinite counts for
`a,m,h,q_post,k_post,v,z,attention_output`; all named parameter norms including
bias and normalization; and all six actual-operand logical-product families.
The A4-Z analytic ceiling is retained as integer counts:
`2,415,919,104 / 18,825,609,216 = 12.8331523101%`.

Model snapshots are retained at boundaries
`0,1,2,4,8,16,32,64,128,256,512,712`; optimizer, scaler, and RNG recovery state
are retained at `256,512,712`, including the final checkpoint. No clipping
frontier, histograms, or qualitative predictions are included. This is the
confirmed pre-launch post-hoc inventory.

`R_block`, `R_model`, and `R_model_max` are logical-product opportunities, not
removed FLOPs or measured speedup.

## Manuscript relationship and interpretation limits

The run executes the experiment-control `A4-OL1` cell and paired comparison
P06. As already operationalized by the control, manuscript `A4` maps to
`A4-Z`, including the pre-`W_o` context site `z`; the pressure suffix is not a
topology. It tests the proposed architecture-wide gate plus conflict-aware
pressure direction, and may support a run-local observation or later paired
analysis, but no result is promoted to a manuscript claim automatically.

Limits are one seed, one 14M scale, one MiniPile pass, and joint intervention at
four sites. It cannot attribute an effect to one site. Independently scheduled
GPUs are matched but not bitwise identical. Strong thresholds can remove input
gradient support. Logical opportunity is not a hardware-speed claim.

## RunPod execution and cost guard

The cohort uses condition-level parallelism: five independent Secure one-GPU
Pods, not DDP. The target is `NVIDIA A100-SXM4-80GB` because the matched
MB32/GAS32 paths reserve about 57 GiB and failed on 24 GB hardware. One seed
worker first runs five exact target boundaries at both `kappa=0` and `0.5`.
It must match the initialization, schedule, cache, config, and code identities;
use Flash SDPA; remain finite with no skipped updates; satisfy every OL1 trust
ratio; and reserve less than 90 percent of device memory. After a pass it becomes
the `kappa=0.5` scientific worker while four additional workers stage in
parallel.

The live 2026-08-30 audit found zero Pods, one pre-existing retained 100 GB
Standard volume `9luykg5yc3` in `EUR-IS-1`, Secure A100 SXM availability `LOW`
across `EUR-IS-1`, `US-KS-2`, `US-MD-1`, and `US-WA-1`, and a live price of
`$1.59/GPU-hour`. The seed has a four-hour absolute termination guard covering
preflight and scientific work; each additional worker has a 2.5-hour guard.
The maximum GPU envelope is `4 * $1.59 + 4 * 2.5 * $1.59 = $22.26`; temporary
Pod storage allowance is `$0.30`, for a rounded `$22.56` all-in maximum. The
existing volume's independent `$7/month` cost is unchanged and is not charged
to this run.

If the seed schedules in `EUR-IS-1`, the retained volume may provide the exact
cache without changing it. Otherwise one local 5.97 GB cache copy is staged and
hash-verified. Four isolated Pod volumes receive concurrent copies from the
verified seed using an ephemeral transfer key removed after use. Source is a
clean Git bundle of the launch commit. Checkouts, environments, caches, logs,
attempts, and checkpoints remain below `/workspace`.

Historical Run 009 OL1 and Run 011 A4 evidence projects roughly 1.8 hours for
the slowest scientific worker plus setup/retrieval; the exact preflight replaces
the recurring-step component before the cohort starts. Monitoring uses five-
minute read-only snapshots and reports step, input tokens, task and pressure
loss, throughput, refreshed ETC, loss scale/overflow, OL1 trust ratio, GPU
memory/utilization, disk, process health, and event age. Warnings are a missing
process, ten-minute stale event, nonfinite or skipped boundary, trust ratio over
budget, memory beyond the preflight envelope, less than 5 GB free disk,
incomplete validation, or projected deadline overrun.

Each terminal worker is remotely verified, inventoried, archived, copied,
hash-checked, extracted, and independently verified locally before its Pod is
deleted. The cohort verifier runs only after all five attempts are local. Final
closeout re-lists Pods and volumes, verifies zero unintended billable compute,
and records lagging posted billing separately from resource teardown.

## Approval record

- 2026-08-30: the user explicitly authorized autonomous execution through
  completion, local retrieval, and RunPod teardown, and stated that all usual
  design and launch confirmations were pre-approved.
