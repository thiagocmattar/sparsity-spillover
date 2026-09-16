# Run 032 — Pythia-14M A7 gates with h-only OL1

## Status and authorization

Implementation/preflight preparation. No scientific attempt has started.
On 16 September 2026 the user approved the matched design and existing
checkpoint/diagnostic inventory, then explicitly requested all five thresholds
on five parallel A100s after receiving the live $1.59/GPU-hour quote and the
2–3 hour / $15.90–23.90 compute estimate. This authorizes implementation,
verification, preflight, five-worker execution, monitoring, retrieval, and teardown.
Use at most 15 GPU-hours ($23.85), plus $0.20 temporary-storage allowance.
The first scientific worker also performs preflight; there is no sixth Pod.

## Question and interpretation

Does OL1 only at h recover the high-threshold A7 logical-opportunity benefit
observed with seven-site OL1? Compare against matched retained Run 013 A7 and
Run 014 all-site OL1 endpoints, using a numbered cross-run analysis after
execution. Recovery at comparable validation loss supports h-only sufficiency.
Little recovery or worse quality weakens that hypothesis. Failure does not
isolate Q/K/V necessity: all-site pressure also adds a, m, and z.

This tests the current manuscript's “Paired effect of adding pressure” claim.
No manuscript change or finding promotion is authorized. It remains a single
seed/scale, fixed-budget comparison, not a runtime-speed or causal-path claim.

## Matched scientific design

The five kappa values are 0, 0.01, 0.05, 0.1, and 0.5; prioritize 0.5.
Topology A7-Z-POST gates a,m,h,z one-sided and q_post,k_post,v symmetrically
in all six layers. Equality survives; the detached mask gives identity gradient
only to surviving inputs. The h gate replaces GELU; q/k gates are after RoPE.
OL1 targets only post-gate h at lambda=1, step budget=1. Its scalar is the
unweighted mean of six tensor means. Relative to Run 014's 42-tensor mean,
this changes both target composition and normalization; do not divide by 42.
Every microbatch must realize exactly h.layer_0 through h.layer_5. Each update
records the capture count and SHA-256 as well as the OL1 boundary geometry.

All other scientific config sections equal Run 014 byte-for-value after YAML
resolution: pinned randomly initialized Pythia-14M (six layers, width 128,
FFN 512, four heads), small_init/wang_init, model/data seeds 1234; pinned
MiniPile/Pythia tokenizer, EOS per document, exact verified caches/order;
712 updates, MB32/GAS32, 1,024 sequences/update at T=2,048 and
1,493,172,224 input tokens/condition. The cohort totals 7,465,861,120 tokens.
FP32 parameters and AdamW moments, dynamic FP16, flash SDPA, no dropout or
activation checkpointing. Fused AdamW uses betas (0.9,0.95), eps 1e-8,
weight decay 0.1 excluding bias/LayerNorm, task clipping 1.0, peak/min LR
1e-3/1e-4, 1% warmup and GPT-NeoX-v1 pre-step cosine semantics.
OL1 uses task-only moments, removes a conflicting adaptive component, and
caps the task-relative correction after AdamW. Zero skipped updates required.

Required initialization SHA-256:
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
Required realized-order SHA-256:
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
Train and full-validation cache identities remain those in Run 014 and
research/DATA.md; preflight recomputes their byte hashes.

## Evaluation and retained measurements

Ordinary validation after step 1 and from the reloaded final checkpoint;
separate complete final activation and eager logical passes. Every pass covers
all 500 documents, 338 complete blocks and 692,224 input tokens, with the
1,444-token excluded tail reported. Exact/near-zero integer counts at 0,
0.001,0.01, activation moments/RMS/L2 and finite counts cover a,m,h,q_post,
k_post,v,z,attention_output in every layer. Retain all named parameter norms,
including bias/normalization, and integer six-operation counters, R_block,
R_model, and A7 reach 5,638,717,440 / 18,825,609,216 (fraction units).

Every boundary retains task/pressure losses and gradient norms/dot/cosine,
clipping, overflow/loss scale, OL1 directions/projection/raw ratio/trust scale/
final ratio, LR, elapsed time, throughput and peak memory. Retain model
checkpoints at 0,1,2,4,8,16,32,64,128,256,512,712 and optimizer/scaler/RNG
recovery at 256,512,712. Final recovery is mandatory. No new clipping sweep,
histogram pass, or qualitative predictions; checkpoint/cache identity permits
later activation diagnostics. Gradient interaction must be collected now.

## Execution and monitoring

Local GPU: RTX 5070 Ti Laptop, 12 GB; unchanged MB32 does not fit the prior
57–63 GiB reservation. Use five Secure A100 SXM 80 GB workers (one condition
each, no DDP), at the live quoted $1.59/hour. Pinned image digest and packages
are recorded in config.yaml/00_setup_remote.sh. Each Pod has 30 GB container
disk and 25 GB persistent /workspace volume. Leave the existing 100 GB network
volume unchanged; no new network volume is requested.

Initial total ETC is 2–3 hours, based on prior A100 SXM records; refresh with
exact A100 MB32/GAS32 preflight. Require runtime/cache/order/initialization
identity, flash attention, five finite updates at kappa=0 and 0.5, exact h
capture and 10% memory headroom before science. Reuse that Pod for kappa=0.5.
Remaining workers independently verify identical runtime, caches, source and
initialization before launch. Persistent detached processes write logs and
artifacts below /workspace and survive SSH disconnection.

Read-only monitoring every five minutes (earlier for the completion window or
a warning). Report step/tokens, task/pressure loss, throughput, ETC, memory,
loss scale, trust ratio, process and event age. Warn on missing process,
10-minute stale event, nonfinite/skipped update, capture mismatch, excess
trust ratio, memory outside preflight, disk below 5 GB, incomplete validation,
or projected budget/deadline overrun. A scoped independent 3-hour stop guard
preserves Pod storage for retrieval if the controller fails. Normal cleanup
deletes each Pod only after verified local retrieval; an emergency stop is
not teardown and any surviving disk must be reported and removed after recovery.

Transfer approximately 1.015 GB per condition: manifests/configs, full events,
metrics, four diagnostic files, all model/recovery checkpoints, environment,
preflight and process logs. Verify archive and internal byte/SHA-256 inventory,
then run the standalone verifier locally before deleting that Pod. Final cohort
verification and an account Pod/endpoint/volume audit complete closeout.

## Verification record

Focused tests passed 11/11 and the complete bootstrap plus new suite passed
253/253. The first full invocation had 11 fixture setup errors because its
pytest parent directory did not exist; creating that directory resolved all
errors without changing scientific code. Python compilation passed. Remote
preflight is pending. The CLI update reported success but left no binary, so
transport pins the working runpodctl 2.12.0 with current help checked.

Historical step_wall_seconds excludes batch assembly/staging. Retain that field
for matching and use differences in end-to-end elapsed_seconds for monitoring
and ETC. Source/checkpoint/diagnostic semantics stay unchanged. Remote Git
info/exclude ignores new run output directories; source files remain tracked,
and code hashes independently verify their bytes.

Scientific files
derive from Run 014, with a separate pressure-site constant and six-tensor
capture audit; the frozen Run 004 training/diagnostic recipe is reused unchanged.

## Execution record ? 16 September 2026, 20:49 UTC

All five scientific workers are running concurrently after independent exact
A100 preflights. Every worker passed five MB32/GAS32 boundaries at both kappa
endpoints, matching initialization and data order, six h captures, finite OL1
geometry and 10% memory headroom. Peak preflight reservations were 57.24 and
61.90 GiB. All 50 non-evidence smoke boundaries passed. Evidence is retained
under `prelaunch/cloud/` and summarized in `prelaunch/cohort-preflight.json`.

Actual source snapshot commit: `240eb34490b40056a3d1dfe86ad0cc4de2f882c7`,
created from main source commit `2a546dc01c3475ec5c0e60d5193523ccdb7ded15`.
This is an actual newly committed deployment subset, not a reconstructed
historical commit. Its complete file inventory and bundle checksum are in
`prelaunch/source-receipt.json`. Scientific source content SHA-256 is
`0e79e9dd83ed4f25f05290f6a2e69f32020d60371b9d4b433df49b2e6a9ac668`.
All 36 text entries normalize CRLF to LF before hashing; this preserves the
same source identity across the Windows checkout and Linux Git blobs.

Infrastructure retries before the affected scientific launches are documented
in `prelaunch/infrastructure/`: smaller source transport, direct local-disk
Python/cache paths, and direct token-file transfer after a CLI ZIP collision.
The preserved train and validation cache hashes match Runs 013/014 exactly.
The transient SSH-address lookup at 20:46 UTC recovered on the next lookup;
no worker restart was required.

Scientific attempts started at 20:39:50 UTC (kappa=0.5), 20:41:50 (0), and
20:47:21?22 (0.01, 0.05, 0.1). The 20:48 UTC snapshot has all five live, zero
skipped updates, six pressure tensors, and 57.23?57.26 GiB training reservation.
Normal monitoring is every five minutes, using end-to-end event differences.
The three-hour creation-time stop guards remain armed; no extension is assumed.
