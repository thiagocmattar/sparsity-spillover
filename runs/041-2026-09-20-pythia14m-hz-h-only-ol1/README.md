# Run041: Pythia-14M h/z thresholds with h-only OL1

Design and launch approved on 20 September 2026. The user confirmed one-sided
h/z gates and h-only orthogonal L1, and explicitly excluded post-hoc clipping.
Exactly four conditions: kappa 0, 0.01, 0.05, 0.1; lambda=1 and trust budget=1.
Status: complete. All four parallel trainings and all 12 final latency processes
passed verification; all retained artifacts are local and both Run041 Pods are
deleted. Results: [observation 001](observations/001-final-results.md).

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

Initial verification:13 focused tests passed; full bootstrap suite242 passed
(7.61s). The remote preflight was specified to execute six real-data boundaries
per condition (one warmup, five timed),
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

### Preflight retry 002: preserve the approved initializer

Cache construction verified both complete hashes. All four first preflights
stopped before any optimizer boundary: native H100 initialization produced
`25f2a630ae0abe9b37938c3c12d615af01a09be44210a0d4017549d8a159d2fc`
instead of the approved historical random parameter hash. The corrected
initializer replays only the retained Run032 **step-zero** tensor bytes, with
strict safetensors and parameter hashes, into the newly constructed HZ model.
It copies no historical topology, optimizer state, or trained/released weights.
The original small_init/Wang recipe metadata and explicit random-snapshot
provenance are retained. The old trainer's `loaded_checkpoint_weights=false`
field denotes absence of released/trained checkpoint weights; the separate
`random_initialization_replay` record identifies this byte-exact random replay.
Scientific inputs (including the agreed initialization hash) do not change.
The failed preflights and first deployment remain preserved as infrastructure
attempt 001; a clean deployment snapshot captures the correction before training.

Retry verification: 14 focused tests and 242 bootstrap tests passed. The first
retry test invocation hit a pre-existing Windows temporary-directory permission
error; rerunning with fresh, run-specific temporary directories passed.

### Scientific execution started

Deployment `b6cc11bf5ad725d9274963c82bb92f41a2359332` passed all four
real-data preflights on Pod `mqykdkc8coy2ny`. Five timed boundaries after one
warmup took median 3.381-3.393 seconds, approximately 618k-620k input tokens/s
per worker. Reserved memory was 61,412,999,168 bytes of 85,017,493,504 (72.24%).
The full validation, checkpoint save, and kappa=0.1 activation/logical passes
completed. All workers restored the approved initializer and data-order hashes;
no preflight optimizer step was skipped. The four independent scientific
workers launched automatically after the forecast fit the unchanged deadline.

At 12:52 UTC, workers were at step 20-21/712 with losses 8.80-8.89 and no
overflows. End-to-end throughput was approximately 611k-613k tokens/s per GPU,
projecting about 40 minutes of remaining training. Artifact sealing is queued
immediately after full training verification. The initialization upload's slow
SFTP path was replaced by verified, resumable SSH streaming; the remaining
28.9MB plus checks completed in 8.92 seconds. Source and artifact identities
were preserved throughout this transport retry.

### Completion and retrieval

All four workers completed 712/712 updates with zero skipped steps or overflows.
Median throughput was 619k-620k input tokens/s per GPU; each scientific condition
took 2,426-2,430 seconds. Full validation, activation, weight, logical-product and
boundary diagnostics passed local `03_verify.py` after retrieval. All 48 model
checkpoints and 12 optimizer/scaler/RNG recovery states remain local. The training
archive contains 282 files (4,059,415,071 payload bytes); its 3,758,793,441 compressed
bytes and every member passed SHA-256 verification. See
`transfer/training-receipt-001.json` and `prelaunch/retrieval-training-001.json`.

The RTX5090 environment was installed while training finished. Its 58 package
pins exactly match `latency/provenance/pip-freeze.txt`; no frozen K050 arithmetic
source was changed. The first smoke incurred 537 seconds of cold compilation;
the second took 71 seconds. Both qualified. All 12 scientific processes then
completed and qualified, averaging 77 seconds each including process setup and
full validation. Every K050 versus eager-native logit absolute error, relative
L2 error and pooled loss delta was zero in the recorded BF16 qualification.
The final measurements contain 1,344 paired samples per condition, 5,376 total.
The 140-file latency archive (746,160 compressed bytes, 6,715,472 payload bytes)
and each member were hash-verified locally before teardown. See
`latency/artifacts/verification.json` and `latency/transfer/receipt-001.json`.

`17_report.py` reduces these terminal artifacts into `artifacts/summary.json`
and observation 001. Validation losses for kappa 0/.01/.05/.1 are respectively
5.128602/5.130649/5.134379/5.151098. Geometric-mean K050 host latencies are
0.612348/0.608618/0.590124/0.573427 ms, with paired native-graph speedups
1.1670x/1.1755x/1.2125x/1.2464x. No post-hoc clipping was performed.
The inherited verification field mentioning a "three-way reduction" is legacy
wording; this report contains exactly the four approved Run041 conditions.

### Teardown and cost record

Training Pod `mqykdkc8coy2ny` was deleted after local verification (delete request
13:39:05 UTC, response 204). Latency Pod `mw3hubdwa1nflw` was deleted after local
verification (request 14:09:37 UTC, response 204 confirmed 14:09:40 UTC). Both local
deadline guards were closed. A final live Pod list contains neither Run041 Pod.
The separate `run042-sparse-scale-001` Pod and pre-existing shared volume were
left untouched; Run041 created no network volume or serverless endpoint.

Creation-to-deletion-request duration at the quoted GPU rates estimates about
USD18.92 total GPU spend, within the USD45 envelope, plus temporary disk charges.
This is an estimate, not a final invoice: at closeout the billing API had posted
USD10.423605 for training and no latency records yet. Exact responses, times and
estimation inputs are retained in `prelaunch/closeout-training-001.json` and
`prelaunch/closeout-latency-001.json`.

Closeout review parsed all 41 retained Python source files, confirmed all 58
runtime package pins, and reconciled the four summary rows with 5,376 raw pairs.
No dataset, model/recovery binary, archive, cache, credential, or active log is
included in Git. The two raw `nvidia-smi` text captures retain their original
trailing whitespace so their inventory hashes remain valid; code and narrative
pass the whitespace check.
