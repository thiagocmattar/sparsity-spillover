# Run041: Pythia-14M h/z thresholds with h-only OL1

Design and launch approved on 20 September 2026. The user confirmed one-sided
h/z gates and h-only orthogonal L1, and explicitly excluded post-hoc clipping.
Exactly four conditions: kappa 0, 0.01, 0.05, 0.1; lambda=1 and trust budget=1.
Status: implementation and local verification in progress; no result yet.

## Scientific contract

Measure validation loss, count-pooled logical sparsity and final K050 latency
for the h/z-only topology. The h gate replaces GELU; z is immediately before
attention W_o. Equality survives. Kappa=0 is ReLU at both sites. Pressure is
the unweighted mean of six post-gate h tensor means. No a/m/q/k/v gate exists.
This extends the manuscript's site-placement and specialized-runtime question.
The four conditions vary only kappa. No manuscript edit or finding promotion.
Faster numerically qualified execution supports a workload-specific runtime
benefit; slowdown or numerical failure limits it. One seed and budget cannot
establish universal superiority or isolate h versus z effects.

Use the unchanged Run032 model/data/optimization recipe: pinned random Pythia-14M,
six layers, width128, FFN512, four heads; small_init/wang_init; model/data seeds1234.
Expected initialization ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57.
Expected realized order f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa.
712 updates, microbatch32 x accumulation32, length2048, 1493172224 input tokens
per condition. Pinned MiniPile and tokenizer revisions and cache hashes follow
config.yaml and research/DATA.md. The full-pass schedule wraps714 blocks.
Fused AdamW betas(.9,.95), eps1e-8, WD.1 excluding bias/LayerNorm, task clipping1;
LR.001 to .0001, 1% warmup, original pre-step cosine. Dynamic FP16, FP32 parameters
and moments, flash SDPA, no dropout or activation checkpointing. No skipped updates.

Complete validation after step1 and from the final checkpoint, separate final
activation and eager logical passes:500 documents,338 blocks,692224 input tokens,
1444 excluded tail. All eight existing diagnostic sites retain per-layer integer
exact/near-zero counts at0/.001/.01 and activation moments/RMS/L2. All named
parameter norms, six-operation product counters, R_block/R_model, and analytic
HZ reach1006632960/18825609216 (fraction0.053471467958893806) are retained.
Gradient conflict and complete OL1 geometry are recorded at every boundary.
Model checkpoints:0,1,2,4,8,16,32,64,128,256,512,712. Optimizer/scaler/RNG:256,512,712.

## Implementation and execution

Run-local HZ registration leaves historical source registries unchanged. The
existing tested Run004 trainer and diagnostics are reused with Run032's audited
h-only OL1 boundary. New run code verifies gate placement, pressure identity,
serialization, exact count ceiling and complete artifacts. No clipping sweep.

One Secure four-H100-SXM80GB Pod, four independent CUDA_VISIBLE_DEVICES workers,
no distributed gradients. Current quote13.96 USD/hour total. Pinned image digest
in config.yaml,40GB container and50GB persistent /workspace. Existing network
volume and Run040 Pod remain separate. Stop deadline3h from creation; total incremental
cap45 USD including a later RTX5090 at0.99/hour for at most90min and temporary disk.
Estimated completion1-3h including setup, training, diagnostics and transfer;
refresh from end-to-end real-data calibration. The unchanged local MB32 workload
does not fit the local12GB GPU; historical A100 reservation57-62GiB motivates80GB.
The local guard requests provider stop; the remote guard ends workload processes.
Neither deletes unretrieved data.
Detached jobs write persistent logs. Monitor training every5min, diagnostics60s;
report progress, loss, throughput and ETC. Warn on nonfinite/skipped updates,
capture mismatch,10min stale event, insufficient10% VRAM headroom, disk below5GB,
or projected deadline/cost overrun. Retrieve immediately after completion.

Transfer roughly4.1GB of all48 model checkpoints,12 recovery states, manifests,
events, diagnostics and logs, plus latency raw pairs/qualification/occupancy/source
identities. Validate SHA256 and byte inventories locally before Pod deletion.
At the deadline stop rather than delete; report any continuing storage.

## Final latency protocol

Frozen K050 on an uncontended RTX5090, BF16, batch1,2048 tokens, full50304 logits,
uncached inference, CUDA graphs. Same-checkpoint native SDPA graph denominator;
unmodified eager anchors correctness.64 fixed timing inputs,7 paired passes,
3 fresh processes for each checkpoint (12 total). Full338-block qualification:
logit atol.25/rtol.02, relativeL2<=.02, absolute pooled loss difference<=.001.
Raw samples and every failure remain visible. No new kernel search or trained control.
Kernel arithmetic sources stay frozen; the installation layer registers HZ.
Canonical training FP16 counters and BF16 runtime diagnostics are separate.

## Verification record

13 focused tests passed; full bootstrap suite242 passed (7.61s). Remote preflight
will execute six real-data boundaries per condition (one warmup, five timed),
full validation and checkpoint serialization; the kappa.1 worker also calibrates
complete activation/logical diagnostics. Local CPU checks do not qualify CUDA.
The K050 compatibility adapter extends only the topology allowlist to HZ; all
1374 archived files were SHA256 verified and remain byte-identical.

An initial recursive bytecode compilation hit Windows path limits inside the
frozen archive. All Python source syntax was then checked in memory successfully.

### Launch infrastructure note

The account API credential is never uploaded to a Pod. Automatic approval review
rejected that proposed stop-guard transport before any credential transfer.
The pinned official image has runpodctl 1.14.15 but no Pod-issued key/config.
The detached workstation guard therefore performs the scoped provider stop at
15:21:20 UTC; a separate credential-free remote guard ends the workload process
group at the same deadline. The remote guard alone cannot stop billing. Source
snapshot 002 captures this infrastructure-only correction before training.
