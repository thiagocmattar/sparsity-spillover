# Run 013 - Pythia-14M paper-scale A7-Z-POST mixed threshold dose response

## Status

Completed with valid evidence on 2026-08-31. The guarded preflight and all five
condition-parallel A100 workers passed; every packet was hash-matched, retrieved,
and verified locally before its Pod was deleted. RunPod now reports zero Pods.
The pre-existing 100 GB shared volume remains intentionally retained, and no
new volume was created. The result is a candidate observation, not a promoted
finding or manuscript claim.

## Maintained execution checklist

- [x] 1. Reconcile the paper-control A7 cell, operational contracts, Run 008
  pilot, and matched full-pass Run 011 comparator.
- [x] 2. Record the complete design and the user's combined design/launch
  authorization.
- [x] 3. Implement and pass focused, affected-run, and full-repository tests.
- [x] 4. Run one guarded Secure A100 SXM 80 GB endpoint preflight at `kappa=0`
  and `kappa=0.5`, including cache, initialization, schedule, runtime, flash,
  memory, and finite-boundary checks.
- [x] 5. Lock the measured launch packet: live price/capacity, ETC, maximum
  duration/cost, cache path, transfer inventory, monitoring, and teardown.
- [x] 6. Start five concurrent one-condition workers.
- [x] 7. Monitor, retrieve, hash-check, and verify all five attempts and the
  cohort.
- [x] 8. Delete every Run 013 Pod, audit all billable resources, and update the
  paper-control execution state without promoting a finding.

## Question and hypothesis

This run asks whether adding symmetric threshold gates at the actual post-RoPE
`q_post`, `k_post`, and `v` attention operands improves the paper-scale
quality--logical-opportunity frontier relative to Run 011's matched one-sided
`A4-Z = {a,m,h,z}` cohort.

At least one positive `kappa` supports the hypothesis if the A7 condition
increases attention-operand exact zeros and `R_model` while yielding a useful
matched validation-loss tradeoff against Run 011 A4. The hypothesis is weakened
if every positive A7 row is unstable or dominated by its matched A4 row. At
`kappa=0`, the added symmetric gates are exact identities in values and
gradients; a material A4/A7 discrepancy there would challenge execution
matching rather than support an attention-gating effect.

## Conditions and matched design

Five independent conditions use one common threshold in all six layers:

| Order | Condition | `a,m,h,z` | `q_post,k_post,v` |
| ---: | --- | --- | --- |
| 1 | `kappa=0` | one-sided | symmetric identity |
| 2 | `kappa=0.01` | one-sided | symmetric |
| 3 | `kappa=0.05` | one-sided | symmetric |
| 4 | `kappa=0.1` | one-sided | symmetric |
| 5 | `kappa=0.5` | one-sided | symmetric |

The one-sided gate keeps `x >= kappa`; the symmetric gate keeps
`abs(x) >= kappa`. Equality survives. Comparisons are detached, surviving
inputs receive identity input gradients, and rejected inputs receive zero
input gradients. The operational topology is
`A7-Z-POST = {a,m,h,q_post,k_post,v,z}`. `q_post` and `k_post` are after
partial RoPE and feed causal QK; `v` feeds PV; `z` is the concatenated PV
context immediately before `W_o`. No L1 or OL1 pressure is used.

Everything else is matched to Runs 011 and 012: the pinned Pythia-14M
architecture constructed from config, seed-1234 random initialization,
seed-1234 realized block order, exact MiniPile caches, optimizer, schedule,
dynamic FP16 contract, validation coverage, diagnostic implementation, and
checkpoint boundaries. Released Pythia weights are never loaded. The five
conditions execute independently on five GPUs; this is condition parallelism,
not DDP, and does not alter the scientific global batch.

## Model, data, optimization, and budget

- Architecture: `EleutherAI/pythia-14m-deduped` at revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`, `(L,d,H,d_h,V) =
  (6,128,4,32,50,304)`, random config construction with Pythia `small_init`
  and residual-output `wang_init`.
- Data: `JeanKaddour/minipile` at
  `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`, append EOS per document, no
  added encode-time special tokens, 2,048-token sequences.
- Train cache: 1,000,000 documents, 1,491,711,416 tokens, 728,374 complete
  blocks, 1,464-token tail, SHA-256
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.
- Budget: 712 optimizer boundaries, one shuffled pass plus 714 wrapped blocks,
  1,493,172,224 input tokens per condition and 7,465,861,120 across the cohort.
- Global batch: 1,024 sequences (`2,097,152` input tokens) as microbatch 32 by
  accumulation 32 on one GPU per condition.
- AdamW: betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, excluding
  bias and LayerNorm from decay; global gradient clipping at norm `1.0`.
- Schedule: GPT-NeoX-v1 pre-step semantics, peak/minimum LR `1e-3`/`1e-4`,
  one-percent linear warmup and cosine decay.
- Precision: FP32 parameters and optimizer state, dynamic FP16 autocast,
  flash SDPA for training, eager attention for logical diagnostics, zero
  dropout, no activation checkpointing.
- Expected initialization SHA-256:
  `ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
- Expected schedule SHA-256:
  `f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

## Complete validation and measurements

Every ordinary and diagnostic pass covers all 500 validation documents, all
338 complete sequences, and 692,224 input tokens. The 1,444-token tail is
excluded and reported. Ordinary validation runs after boundary 1 and from the
reloaded final checkpoint. Separate final activation and eager logical-product
passes use the same complete workload.

The agreed post-hoc inventory is:

- count-first exact-zero and near-zero counts at `0`, `1e-3`, and `1e-2`, plus
  sums, RMS/L2, and finite/nonfinite counts for every layer of `a,m,h,q_post,
  k_post,v,z,attention_output`;
- named per-parameter weight norms and element counts, including bias and
  normalization;
- integer actual-operand counters for fused QKV, causal QK, causal PV, `W_o`,
  MLP W1, and MLP W2, plus `R_block` and `R_model`;
- model snapshots at `0,1,2,4,8,16,32,64,128,256,512,712` and complete
  optimizer/scaler/RNG recovery state at `256,512,712`, retaining the final;
- every-boundary task loss, LR, clipping, dynamic-loss-scale health,
  throughput, wall time, and peak memory.

Gradient conflict and OL1 boundary geometry are inapplicable because there is
no pressure objective. No post-hoc clipping frontier, histogram set, or
qualitative prediction file is requested. The retained final checkpoints and
exact cache/code identities preserve later activation-only diagnostics, while
no uncollected gradient interaction is implied.

For A7-Z-POST at Pythia-14M and `T=2,048`, all six block-operation families are
reachable. Because `v=0` already closes through PV into `W_o`, selecting `z`
adds no distinct all-zero reach. The exact ceiling is
`5,638,717,440 / 18,825,609,216 = 29.95237697384911%`. This ceiling,
`R_block`, and `R_model` are logical-product opportunities, not removed FLOPs
or measured speedup.

## Manuscript and operational relationship

This is the paper-control ladder's Pythia-14M `A7` cell and operationally
executes `A7-Z-POST`. It tests paired comparison P07 (`A4(kappa) -> A7(kappa)`)
and supplies the missing full-pass A7 half of P12/P14. The mixed operator
assignment is already an accepted operational/manuscript crosswalk: one-sided
`a,m,h,z` plus symmetric post-RoPE Q/K/V. No TeX or finding is changed merely
because the run completes; any cross-run synthesis belongs in a subsequent
numbered analysis and any paper claim still requires user approval.

## RunPod execution definition

One guarded Secure A100 SXM 80 GB Pod first executes five exact MB32/GAS32
boundaries at both threshold extremes. It must verify cache, initialization,
schedule, config/code, Python/Torch/Transformers/CUDA, flash attention, finite
non-skipped boundaries, clipping, and at least ten-percent device-memory
headroom. The exact preflight replaces historical timing while Run 011's
57.242 GiB peak reserved memory establishes the initial fit requirement.

After preflight, five Secure one-GPU Pods run concurrently, one condition per
GPU. The target image is pinned by digest; each Pod has 30 GB container disk,
25 GB persistent Pod volume at `/workspace`, SSH, and a 2.5-hour termination
deadline. The preflight has a 1.5-hour deadline. The current `runpodctl 2.12.0`
binary omits its documented termination flags, so creation uses the current
RunPod GraphQL `terminateAfter` field and then verifies the stored deadline;
ordinary lifecycle, SSH, and transfer use the CLI/MCP surfaces.

The pre-existing 100 GB network volume `9luykg5yc3` in `EUR-IS-1` remains
intentionally retained and already bills independently. The preflight will
attach it only when compatible A100 placement exists, and will use its cache
only after exact metadata/byte/hash verification. Otherwise the approved
fallback is one verified cache seed plus concurrent hash-checked copies to
isolated Pod volumes. No new network volume is authorized or needed.

Monitoring is read-only at five-minute intervals, shortened to the refreshed
completion window. Every report includes step, input tokens, finite task loss,
throughput, refreshed ETC, loss-scale/overflow health, GPU memory/utilization,
disk, process state, and last-event age. Warnings are a missing process, a
ten-minute stale event, nonfinite or skipped boundary, reserved memory outside
the preflight envelope, less than 5 GB free, incomplete validation, or a
projected deadline overrun.

Each terminal worker is independently verified remotely, inventoried by byte
count and SHA-256, copied locally, re-hashed, and verified locally before its
Pod is deleted. The cohort verifier runs only after all five accepted attempts
are local. Closeout then re-lists Pods and volumes and records posted billing;
provider billing lag is reported separately from the authoritative zero-Pod
teardown state.

## Scientific execution and closeout

The preflight used one Secure A100 SXM 80 GB Pod in `EUR-IS-1`, attached the
pre-existing shared volume, and passed the exact cache, initialization,
schedule, code, flash-attention, finite-boundary, and memory checks at both
threshold endpoints. Steady boundary medians were about 4.87 seconds at
`kappa=0` and 4.99 seconds at `kappa=0.5`; peak reserved memory was 57.23 GiB,
leaving 27.8% physical-memory headroom. Its packet was retrieved and verified
before the Pod was deleted.

Five fresh Secure A100 SXM workers in `US-MD-1` then received independent,
hash-matched copies of the exact train and validation caches. Their scientific
processes launched between 00:57:44 and 00:57:47 UTC and completed between
02:19:16 and 02:26:49 UTC. Every worker held dynamic loss scale 4,096, recorded
no overflow or skipped optimizer boundary, and completed all 712 steps.

The locally verified endpoint summary is:

| `kappa` | Final train loss | Final validation loss | `R_block` | `R_model` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.546727 | 5.468401 | 24.0971% | 7.2177% |
| 0.01 | 5.540280 | 5.458822 | 25.4325% | 7.6176% |
| 0.05 | 5.513442 | 5.437888 | 30.4715% | 9.1269% |
| 0.1 | 5.502831 | 5.428681 | 34.8053% | 10.4250% |
| 0.5 | 5.786563 | 5.702923 | 51.3709% | 15.3868% |

Against Run 011 A4 at matched `kappa`, the A7-minus-A4 result is:

| `kappa` | Validation-loss delta | `R_model` delta |
| ---: | ---: | ---: |
| 0 | -0.002096 | +0.0056 pp |
| 0.01 | -0.007678 | +0.2039 pp |
| 0.05 | +0.003778 | +0.9210 pp |
| 0.1 | +0.009038 | +1.4720 pp |
| 0.5 | +0.043244 | +5.1713 pp |

The near-null `kappa=0` pair supports execution matching. At `kappa=0.01`, A7
improves both complete-validation loss and logical opportunity relative to A4.
At `kappa >= 0.05`, adding the three symmetric attention-operand gates buys
progressively more `R_model` with a progressively larger quality cost. Within
A7, the lowest final validation loss is 5.428681 at `kappa=0.1`, while the
largest measured `R_model` is 15.3868% at `kappa=0.5`, or 51.37% of the declared
A7 analytic reach ceiling. These are one-seed, one-scale descriptive outcomes;
no measured sparse-kernel speedup is implied.

Every 1.014 GB archive matched its cloud SHA-256 and passed the local standalone
verifier before teardown. The cohort verifier accepted five conditions, 3,560
optimizer steps, 20 complete validation passes, 7,465,861,120 training tokens,
common initialization/schedule/code identities, exact checkpoint inventories,
and all agreed activation, weight, and logical-product diagnostics. Full
archive, readiness, log, Pod, and teardown provenance is recorded in
`prelaunch/scientific-transfer-closeout.json`; accepted endpoint data is in
`artifacts/verification.json`.

All six Run 013 Pods (one preflight and five scientific) are deleted and the
live Pod inventory is empty. No network volume was created or changed. The
pre-existing volume `9luykg5yc3` remains intentionally retained at its
independent `$0.01/hour` charge. Closed-hour Pod billing had posted `$8.113020`
at the 02:33 UTC audit while the final partial hour was still pending. The
account-balance reduction across the run was `$15.832319`, including the
retained volume's baseline charge, below the approved `$22.65` all-in ceiling.

## Support, refutation, and interpretation limits

Support requires a useful matched A7 gain in logical opportunity without
unacceptable validation degradation; refutation includes universal domination
by Run 011 or numerical instability. The one-seed, one-scale result does not
establish scale persistence. The joint Q/K/V addition cannot attribute an
effect to one attention operand. Thresholded training changes both forward
values and gradient support. Independently scheduled GPUs are matched but not
assumed bitwise. Logical opportunity is not a hardware-speed result.

## Approval record

- 2026-08-30: the user explicitly requested the new paper-scale Pythia-14M A7
  run, five-GPU parallel RunPod execution, complete local retrieval, and
  teardown with no continuing Run 013 charge, and waived separate design and
  launch confirmation waits. This authorizes the exact five-kappa operational
  A7 design above, the required preflight, the live guarded cloud envelope once
  measured, retries that do not change scientific inputs, artifact transfer,
  and termination.

## 2026-09-01 billing refresh

The Status Report Number 2 RunPod REST v2 audit reconciled the six Run 013 Pod
IDs to `$15.7277473900` GPU plus `$0.0754050968` temporary Pod-disk spend,
`$15.8031524868` total. This Pod-specific total replaces the lagging closeout
snapshot for cost reporting; it does not use account-balance deltas and excludes
the separately retained network volume.
