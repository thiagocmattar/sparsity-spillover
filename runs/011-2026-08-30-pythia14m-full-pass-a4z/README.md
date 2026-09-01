# Run 011 - Pythia-14M paper-scale A4-Z threshold dose response

## Status

Completed with valid evidence on 2026-08-30. The approved preflight and all five
condition-parallel A100 workers completed, their packets were hash-verified and
accepted locally, and the cohort verifier passed. All Run 011 Pods were deleted;
the one pre-existing retained network volume remains unchanged. No finding or
manuscript claim has been promoted from these results.

## Maintained execution checklist

- [x] 1. Interpret and confirm the scientific design.
- [x] 2. Create Run 011 and implement its frozen config, entrypoints,
  diagnostics, verification, and RunPod playbook.
- [x] 3. Pass focused, affected-run, and repository-wide tests; reconcile local
  fit, historical ETC, live RunPod inventory, capacity, and price.
- [x] 4. Run one guarded Secure A100 SXM 80 GB endpoint preflight at the two
  threshold extremes; retrieve and locally hash-verify its packet; stop the Pod.
- [x] 5. Present the measured five-worker ETC, cost, cache-distribution choice,
  and post-hoc inventory; obtain explicit scientific-launch approval.
- [x] 6. Start five concurrent one-condition workers.
- [x] 7. Monitor, retrieve, hash-check, and verify all five attempts and the
  cohort.
- [x] 8. Terminate the exact Pods, reconcile billing/resources, and consolidate
  only user-approved observations or findings.

## Question and hypothesis

This run asks how a common one-sided threshold at operational topology
`A4-Z = {a,m,h,z}` changes complete-validation quality, sitewise activation
sparsity, and model-wide logical-product opportunity during one paper-scale
Pythia-14M MiniPile pass.

Increasing `kappa` should increase selected-site exact zeros and may increase
`R_block` and `R_model`. A useful response requires at least one nonzero
threshold to offer a nondominated validation-loss versus logical-opportunity
tradeoff relative to the within-run `kappa=0` reference. Comparing
`A4(kappa=0)` with Run 004's matched `A1-H` ReLU control tests the topology
expansion separately from positive threshold magnitude.

## Conditions and matched design

Five independent conditions use the same fixed gate in all six layers:

| Order | Condition | Gate |
| ---: | --- | --- |
| 1 | `kappa=0` | `x` when `x >= 0`, otherwise zero |
| 2 | `kappa=0.01` | `x` when `x >= 0.01`, otherwise zero |
| 3 | `kappa=0.05` | `x` when `x >= 0.05`, otherwise zero |
| 4 | `kappa=0.1` | `x` when `x >= 0.1`, otherwise zero |
| 5 | `kappa=0.5` | `x` when `x >= 0.5`, otherwise zero |

Equality survives. There is no L1 or OL1 pressure. The conditions share the
pinned architecture, random seed-1234 initialization, seed-1234 realized block
order, optimizer, schedule, validation coverage, diagnostics, and checkpoint
cadence. Released Pythia weights are never loaded.

Run 006 is the local A4-Z pilot and implementation precedent. It used seed 0,
BF16, global batch 64, and 581 updates, so its outcomes are not pooled with this
paper-scale evidence.

## Paper-scale Pythia recipe

- Pythia-14M architecture revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`, constructed from config.
- Pythia `small_init` and residual-output `wang_init`.
- MiniPile train cache: 1,491,711,416 tokens, 728,374 complete blocks,
  1,464-token tail, SHA-256
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.
- Sequence length 2,048; global batch 1,024 as MB32/GAS32.
- 712 optimizer boundaries and 1,493,172,224 input tokens per condition;
  7,465,861,120 tokens across the cohort.
- AdamW `(beta1,beta2)=(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, with
  bias and LayerNorm excluded from decay.
- Global gradient clipping at norm `1.0`.
- Peak/minimum LR `1e-3`/`1e-4`, one-percent warmup, GPT-NeoX-v1 pre-step
  cosine semantics.
- FP32 parameters and optimizer state, dynamic FP16 autocast, flash SDPA,
  zero dropout, and no activation checkpointing.

The expected initial-parameter SHA-256 is
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
The expected schedule SHA-256 is
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

## Validation, diagnostics, and retention

Every ordinary or diagnostic validation covers all 500 documents, all 338
complete blocks, and 692,224 input tokens. The 1,444-token tail is excluded and
reported. Ordinary validation runs after boundary 1 and from the reloaded final
checkpoint. Separate final activation and eager logical-product passes use the
same complete workload.

The final checkpoint records count-first exact-zero and near-zero counts at
`0`, `1e-3`, and `1e-2`, plus RMS/L2 moments for `a`, `m`, `h`, `q_post`,
`k_post`, `v`, `z`, and post-`W_o` attention output. It also records all named
parameter norms and the six actual-operand logical-product families.
`R_block`, `R_model`, and `R_model_max` are logical opportunities, not speedup.

For A4-Z at 14M and `T=2,048`, the analytic ceiling is
`2,415,919,104 / 18,825,609,216 = 12.8331523101%`.

Model snapshots are retained at boundaries
`0,1,2,4,8,16,32,64,128,256,512,712`. Optimizer, scaler, and RNG recovery state
are retained at `256,512,712`, including the final checkpoint. No clipping
frontier, histograms, qualitative predictions, or pressure-gradient metrics are
included.

## RunPod execution interpretation

Five independent Secure one-GPU Pods are proposed, one condition per A100 SXM4
80 GB GPU. This is condition-level parallelism, not DDP: all five conditions can
train concurrently without changing MB32/GAS32 or adding cross-GPU collectives.
Run 004 measured roughly 57 GiB reserved for the closest full-pass task-only
path and failed on a 24 GB RTX 4090.

Before scientific launch, one guarded A100 must run five exact MB32/GAS32
boundaries at both `kappa=0` and `kappa=0.5`. It must reproduce the initialization
and schedule identities, remain finite without skipped updates, and keep peak
reserved memory below 90 percent of device memory. A successful preflight Pod
may become one scientific worker after a separate launch decision. The
preflight Pod has a 1.5-hour absolute termination guard; every scientific
worker receives a fresh 2.5-hour guard.

The existing 100 GB network volume `9luykg5yc3` remains an independent retained
resource. It is used only if live A100 placement naturally lands in `EUR-IS-1`;
otherwise each worker gets a verified cache on its own `/workspace` Pod volume.
No new network volume is part of this design.

## Implementation and preflight launch packet

The run owns `config.yaml`, numbered setup/smoke/train/verify/monitor/preflight
entrypoints, A4-Z-specific config and diagnostics wrappers, standalone and
cohort verification, and the staged deployment playbook. Frozen Run 004
training machinery is reused under a private module identity, and every reused
file participates in the run-code hash. The approved identities are:

- config SHA-256
  `6e65db62bd3a5cc4673b44d3bd0e09ef963663460ccf14e874ed6b458913a770`;
- run-code content SHA-256
  `c253d6cba5511e7e43f3599a5776b846b4a8b8048a32c23fcfc66317d8b1bf09`.

Local verification passed 7 focused Run 011 tests, 58 affected-run tests, and
all 150 repository tests. The tests cover the five-condition mapping, gate
equality semantics, configuration and code identities, A4-Z hook placement,
the exact analytic ceiling, endpoint-smoke requirements, and standalone/cohort
artifact rejection paths. Python compilation also passed.

The closest Run 004 GeLU/ReLU full-pass controls reserved 56.95--57.29 GiB and
took 62.95--65.96 minutes of scientific-process wall time. A 24 GB local RTX
4090 already failed this shape, so the representative workload does not fit
locally. Combining 1,424 measured optimizer-boundary samples and eight complete
validation/diagnostic samples with conservative provision, setup, and transfer
allowances gives 1.652 hours median and 1.674 hours p90 from first provisioning
through local retrieval when all five workers proceed concurrently. This is a
historical estimate, not a target-GPU measurement; the endpoint preflight will
replace its step-time component.

At the 2026-08-30 preflight audit, RunPod reported zero Pods and one pre-existing
retained 100 GB volume. Secure `NVIDIA A100-SXM4-80GB` capacity was `MEDIUM`,
CUDA 12.8 was available, and the live price was `$1.59/GPU-hour`. The cheaper
A100 PCIe offer did not advertise CUDA 12.8 capacity and is not an approved
substitute. No A100 capacity was advertised in the retained volume's
`EUR-IS-1` data center, so the proposed preflight uses a 25 GB per-Pod
`/workspace` volume and a hash-verified local cache copy.

The approved preflight created exactly one Secure A100 SXM 80 GB Pod with the
pinned image, 30 GB container disk, 25 GB `/workspace` volume, SSH, and a
1.5-hour absolute termination guard. It completed and was stopped after 1,770
runtime seconds. Estimated compute is `$0.78175` before posted-billing
reconciliation, below the `$2.60` envelope. The automatic deletion deadline is
`2026-08-30T18:04:31Z`; scientific execution was not authorized or started.

Both endpoint conditions passed five exact MB32/GAS32 boundaries with no
overflow or skipped update. Median steady boundary time was 4.872 seconds
(about 430,490 input tokens/second); peak reserved memory was 57.242 GiB, leaving
27.77% headroom. Initialization, schedule, cache, config, and code identities
matched, and flash attention was active. The retrieved packet is under
`prelaunch/preflight-attempt-001/`.

The first cache setup attempted a fresh pinned Hugging Face build. At 75% it
crossed the free-disk warning because the source cache and growing token file
coexisted. The non-evidence builder was stopped, its exact partial file and
re-downloadable cache were removed, and the already-approved local token cache
was transferred instead. Remote byte counts and hashes matched. That cache is
private to the stopped Pod, not automatically shared with the other workers.
The pre-existing EUR network volume remains the authoritative reusable copy for
future compatible runs; the preflight did not replace or modify it.

For the scientific fleet, the proposed distribution is one verified seed and
four concurrent seed-to-worker copies onto isolated per-Pod volumes, each
accepted only after byte-count and SHA-256 verification. This avoids five
independent tokenizations and preserves writable-workspace isolation and GPU
placement flexibility. Run 004 measured 78--84 seconds for the same concurrent
distribution pattern. If the stopped preflight Pod expires before approval, the
seed will come from the retained EUR volume when compatible capacity exists, or
from one pinned, verified rebuild; no new network volume is proposed.

For planning, the measured preflight updated the concurrent five-worker
projection to 1.676 hours median and 1.790 hours p90 from provisioning through
local retrieval, or `$13.33` median and `$14.23` p90 compute at the current
price. Five scientific Pods at the current ceiling cost at most
`5 * 2.5 * $1.59 = $19.875` compute; `$0.25` storage allowance gives a rounded
`$20.15` scientific envelope. The separate maximum for both stages would be
`$22.75`. The user approved that scientific envelope before provisioning.

The pre-launch audit reported `$1.59/GPU-hour` and `MEDIUM` overall Secure A100
SXM availability, but only `LOW` availability in each advertised data center,
including `EUR-IS-1`.

The preflight retrieval inventory is the JSON result, stdout/stderr log,
environment freeze, source/config identities, cache identities, and an archive
inventory with byte counts and SHA-256 hashes. Scientific retrieval additionally
includes each attempt's metrics/events, all declared model and recovery
checkpoints, activation/weight/logical diagnostics, manifest, config, and
transfer inventory; Run 004 indicates approximately 1.014 GB per condition, or
5.07 GB across the cohort.

Monitoring uses five-minute intervals and reports step, input tokens, task loss,
throughput, refreshed ETC, loss scale/overflow, GPU memory/utilization, disk,
process health, and event age. Warning conditions are a missing process, a
ten-minute stale event, any nonfinite value or skipped boundary, reserved memory
outside the preflight envelope, less than 5 GB free disk, incomplete validation,
or projected termination-deadline overrun.

The confirmed post-hoc inventory retains exact and near-zero counts, activation
RMS/L2 moments, per-parameter weight norms, six logical-product counter families,
and the final checkpoint with optimizer/scaler/RNG state. Gradient conflict is
inapplicable without a pressure objective; a clipping frontier, histograms, and
predictions remain excluded. This inventory was explicitly reconfirmed before
scientific launch because gradient-time quantities cannot be reconstructed
later.

## Scientific execution and closeout

Five fresh Secure A100 SXM 80 GB Pods were provisioned at approximately
17:16 UTC. Each passed the full remote readiness check before training. One
worker received the exact local cache and then seeded four concurrent direct
copies; every worker matched the declared train and validation byte counts and
SHA-256 identities before launch. The first fanout command found that the four
empty destination cache directories had not yet been created, so it failed
before transferring data. Creating those directories and repeating the same
hash-checked copy was an infrastructure retry and did not change any scientific
input. The five processes launched between 17:43:50 and 17:45:21 UTC.

All conditions completed 712 optimizer boundaries and four complete validation
passes. The locally verified endpoint summary is:

| `kappa` | Final train loss | Final validation loss | `R_block` | `R_model` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.553432 | 5.470497 | 24.0783% | 7.2120% |
| 0.01 | 5.544739 | 5.466500 | 24.7516% | 7.4137% |
| 0.05 | 5.508773 | 5.434110 | 27.3965% | 8.2059% |
| 0.1 | 5.493523 | 5.419642 | 29.8908% | 8.9530% |
| 0.5 | 5.744903 | 5.659680 | 34.1059% | 10.2155% |

The monotone logical-opportunity response supports the predicted direction:
every positive threshold increases selected-site exact-zero mass and
`R_model`. Quality improves through `kappa=0.1`, which is better than the
within-run `kappa=0` reference on both final validation loss and logical
opportunity; `kappa=0.5` buys the most opportunity but reverses the quality
gain. The A4-Z `kappa=0` validation loss is 0.200851 higher than Run 004's
matched A1-H ReLU control. These are descriptive one-seed, one-scale outcomes,
not a promoted finding.

Each 1,014,016,000-byte transfer archive matched its remote SHA-256, contained
only its declared attempt directory, and passed the local standalone verifier
before its Pod was deleted. The cohort verifier then accepted all five
attempts: 3,560 completed optimizer steps, 20 complete validation passes,
7,465,861,120 training input tokens, common initialization/schedule/code
identities, exact checkpoint inventories, and all approved diagnostics. The
archive and support-file hashes, Pod mapping, and teardown state are recorded
in `prelaunch/scientific-transfer-closeout.json`; the accepted metrics are in
`artifacts/verification.json`.

The five scientific Pods and the stopped preflight Pod are deleted, and the
live Pod inventory is empty. The preflight deletion guard did not remove its
already-stopped Pod, so closeout explicitly deleted it after confirming its
packet was local. No network volume was created or changed. Pre-existing volume
`9luykg5yc3` remains intentionally retained at its independent `$7/month`
charge. The 19:39 UTC post-teardown billing query showed `$11.469429` posted
across the six Run 011 Pods, but RunPod had not yet posted the final partial-hour
charges; the absence of billable Pods and the approved `$22.75` all-in ceiling
bound the lagging remainder. The post-teardown billing state is recorded in
`prelaunch/launch-plan.json`.

## Verification and interpretation limits

Each worker has a standalone verification path so its artifacts can be copied,
hash-checked, and accepted before that Pod is deleted. The cohort verifier then
requires all five attempts, one common initialization/schedule/code identity,
20 complete validation passes, exact checkpoint inventories, the A4-Z ceiling,
and the valid hash-locked Run 004 `relu-control` comparator.

The run has one seed and one small model scale. Joint gating cannot attribute an
effect to one site. Thresholded training changes forward values and gradient
support. Independently scheduled GPUs are scientifically matched but not
bitwise. No sparse kernel is measured, and no observation becomes a manuscript
claim without a separate user-approved finding.

## Approval record

- 2026-08-30: the user approved all five kappas, operational A4-Z, the complete
  paper-scale contract, five condition-parallel A100 workers, reuse of Run 004's
  A1-H comparator, checkpoint retention, and omission of a clipping frontier.
- 2026-08-30: the user explicitly replied `Approved for preflight` to authorize
  exactly one non-evidence Secure A100 SXM 80 GB Pod under the `$2.60`
  incremental envelope and 1.5-hour absolute termination guard. This approval
  does not authorize scientific training.
- 2026-08-30: after the passing preflight packet, measured ETC/cost, cache plan,
  transfer/teardown plan, monitoring warnings, and post-hoc inventory were
  presented, the user explicitly replied `Approved`. This authorizes five
  condition-parallel Secure A100 SXM 80 GB workers under the `$20.15`
  scientific envelope and 2.5-hour per-worker termination guard.

## Subsequent consolidation

- 2026-08-30: the user approved tentative Finding F001 from the five matched
  Run 011 A4 and Run 012 A4-OL1 pairs. Analysis 007 owns the count-reconciled
  synthesis; this run remains the immutable A4 evidence source. This later
  approval supersedes the no-finding status recorded at run closeout above.

- 2026-08-31: implementation audit superseded that interpretation. Run 012
  realized A4-Z + OL1@h, not four-site A4-OL1; F001 is discarded. Corrected
  Run 015 and Analysis 009 now provide the eligible four-site comparison.

## 2026-09-01 billing refresh

The Status Report Number 2 RunPod REST v2 audit reconciled the six Run 011 Pod
IDs to `$16.0076928766` GPU plus `$0.0985532458` temporary Pod-disk spend,
`$16.1062461223` total. This current Pod-ID total supersedes the lagging posted
snapshot above and excludes the separately retained network volume.
