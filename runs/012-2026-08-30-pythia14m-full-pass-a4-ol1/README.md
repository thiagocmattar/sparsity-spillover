# Run 012 - Pythia-14M paper-scale A4-OL1 full pass

## Status and approval

Completed, retrieved, and verified on 2026-08-30. All five conditions reached
712 optimizer boundaries, their complete declared artifacts and final
checkpoints are stored locally, the cohort verifier passed, and the final
RunPod audit reported zero Pods.

The user requested this run on 2026-08-30 and explicitly pre-approved the
scientific design, implementation, RunPod preflight, scientific launch,
monitoring, artifact retrieval, and teardown without waiting at the usual two
confirmation gates. The design below was therefore locked before implementation
and billable execution; approval itself was not treated as run evidence.

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
hash-verified. Source is a clean Git bundle of the launch commit. Checkouts,
environments, caches, logs, attempts, and checkpoints remain below `/workspace`.
The planned seed-to-worker transfer would have required placing a private
ephemeral key on the seed; the credential safety gate rejected that action, so
the actual fanout used concurrent resumable local-to-worker byte streams. All
copies were accepted only after their complete SHA-256 values matched.

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

## Live execution record

- Launch source is clean commit
  `23872b9a89bf84c8a786965f50d57a21b299f4dd`; the self-contained Git bundle
  SHA-256 is
  `841fc8473b1ddd29385d5f4e6f78067411b1cf2758ce005f1f9a397d153e05bc`.
- `EUR-IS-1` had no matching capacity at allocation time. The seed scheduled in
  `US-MD-1`; the other workers scheduled in `US-MD-1`, `US-KS-2`, and
  `US-WA-1`. The five Pod IDs and absolute termination deadlines are retained
  in `prelaunch/launch-plan.json`.
- The seed preflight passed on the exact cache and runtime at 2026-08-30
  20:19:45 UTC. Mean optimizer-boundary times were 6.0522 seconds at
  `kappa=0` and 6.0255 seconds at `kappa=0.5`; peak reservation was
  61.9023 GiB against the 71.3248 GiB gate.
- Every worker independently matched the train, validation, and metadata
  hashes before launch. The `US-KS-2` worker spent 9.5 minutes in network-volume
  `python -m venv`; that setup process was stopped before scientific execution
  and retried with the identical pinned packages on container disk. Its source,
  cache, condition, and scientific attempt were unchanged. The retry script and
  both setup records remain under `prelaunch/`.
- The `kappa=0.5` attempt launched first after preflight. The remaining four
  attempts launched independently as their cache, runtime, and clean-tree gates
  passed. All five had emitted finite, non-skipped training events by 2026-08-30
  20:46:12 UTC.

## Verified result

The cohort verifier accepted five completed conditions, 3,560 optimizer
boundaries, 7,465,861,120 training input tokens, and 20 complete validation
passes. Every validation pass covered all 338 complete blocks and 692,224 input
tokens, with the declared 1,444-token tail excluded. Initialization, schedule,
source, checkpoint, diagnostic, and transfer identities all passed.

| `kappa` | validation loss | `R_model` | loss delta versus matched A4 | `R_model` delta versus matched A4 |
|---:|---:|---:|---:|---:|
| 0 | 5.215749 | 8.4094% | -0.254749 | +1.1974 pp |
| 0.01 | 5.204504 | 8.5858% | -0.261996 | +1.1721 pp |
| 0.05 | 5.195590 | 9.0356% | -0.238520 | +0.8297 pp |
| 0.1 | 5.228687 | 9.3435% | -0.190955 | +0.3905 pp |
| 0.5 | 5.722666 | 10.2274% | +0.062986 | +0.0119 pp |

At matched `kappa <= 0.1`, A4-OL1 lowers complete-validation loss and
increases logical-product opportunity relative to Run 011 A4. The gain
contracts as `kappa` increases and reverses in quality at `kappa=0.5`,
where the opportunity increment is negligible. Run 012's lowest loss occurs at
`kappa=0.05`; `kappa=0.1` offers a higher-opportunity tradeoff. This is
descriptive one-seed, one-scale evidence, and `R_model` is not measured
speedup. The complete result, caveats, and provenance are in
`observations/001-a4-ol1-matched-outcome.md`.

## Retrieval and RunPod closeout

Each terminal attempt first passed `03_verify.py --condition` remotely. Its
single-root tar archive was then copied locally, matched against the remote
SHA-256, checked for path safety, extracted, reconciled against the internal
transfer inventory, and passed the same standalone verifier before that exact
Pod was deleted. The five archives totaled 5,073,725,440 bytes; all temporary
tar files were removed only after extraction, while the five extracted attempt
trees and final checkpoints remain under `artifacts/attempts/`.

The final cohort invocation of `03_verify.py` wrote
`artifacts/verification.json` with status `verified`. The 2026-08-30
22:14:34 UTC RunPod audit found zero Pods and one unchanged pre-existing
100 GB standard volume, `9luykg5yc3` in `EUR-IS-1`. Run 012 created no
retained resource. Lifecycle duration at the live $1.59/GPU-hour price implies
about $14.77 of GPU compute, below the $22.56 run envelope. Posted RunPod Pod
billing was still incomplete at closeout ($7.0914 attributable subtotal), so
resource absence rather than the lagging ledger is the teardown authority.
Exact retrieval, resource, and billing records are in
`prelaunch/scientific-closeout.json`.
