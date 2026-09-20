# Run046: Pythia-70M T_hz/P_h at kappa=0.5

The user explicitly requested one additional matched 70M RunPod run at kappa=0.5,
as fast as possible. This selects the Run043 design and authorizes its execution
with the same diagnostic package and no post-hoc clipping. Exactly one training
condition is implemented: `hz-h-ol1-kappa-0p5`. Status: complete, retrieved,
verified locally, and both run-owned Pods deleted.

## Question and scientific contract

Complete the missing high-threshold endpoint of Run043's T_hz/P_h grid, called
T2/Ph in the manuscript. Only kappa changes. Higher model-wide logical opportunity
or lower qualified latency at a measured loss cost characterizes the tradeoff;
loss degradation, slowdown or qualification failure limits its usefulness.
This is one seed and one budget, without causal attribution to an individual gate.
The current operational definitions agree with this requested placement. No
manuscript update or research finding promotion is included.

Six-layer Pythia-70M: width512, FFN2048, eight heads, vocabulary50304; architecture
revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c. Replay the canonical random
initialization and RNG used by Run043/034/018, never trained or released weights.
Parameter SHA e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062;
data-order SHA d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e.
Model/data seeds1234. Pinned MiniPile and tokenizer revisions, append-EOS rule,
all one million training documents and exact cache hashes match Run043.

One-sided gates retain x>=0.5, otherwise zero, with a detached comparison.
The h gate replaces GELU; z gates the concatenated context before W_o. No other
gate site. OL1 targets the six post-gate h tensors, averaging their means equally,
with lambda=1 and trust budget=1. Training uses 712 updates, microbatch4 times
accumulation256, global1024, sequence2048: 1,493,172,224 input tokens.
FP32 parameters and moments, dynamic FP16, flash SDPA, no dropout or activation
checkpointing. Fused AdamW (.9,.95), eps1e-8, WD.1 excluding bias/LayerNorm,
task-gradient clipping1, LR.001 to .0001 with 1% warmup and pre-step cosine.

Validation at step1 and final checkpoint uses all500 documents, 338 complete
2048-token blocks, 692224 input tokens, with the1444-token tail excluded.
The final activation and logical passes have the same full coverage. Retain
integer exact/near-zero counts at0/.001/.01, moments/RMS/L2 at all eight diagnostic
sites, all named weight norms, gradient conflict and OL1 geometry at every update,
logical-product counts, R_block/R_model and the analytic HZ ceiling
16,106,127,360 /104,293,466,112 products per B1,T2048 forward. No clipping sweep.
Retain model checkpoints0,1,2,4,8,16,32,64,128,256,512,712 and full recovery
states256,512,712. All12 models and3 optimizer/scaler/RNG states require local
hash verification before teardown, about5.1GB plus diagnostics.

## Execution and retention

One Secure H200 at the live USD4.59/hour quote, preferring US-NC-1, which hosted
a prior matched 70M workload around289k tokens/s. Other H200 hosts ranged about
139k–305k tokens/s; Run043 measured about154k. These are estimates, not a promised
rate. Exact six-boundary preflight includes one warmup, all validation blocks,
both diagnostics and checkpoint serialization; it must pass memory/identity/
overflow checks before a fresh scientific start. The user selected cloud for
fast completion. Local execution is not scheduled.

Pinned Run043 image;40GB container and50GB isolated persistent /workspace.
Four-hour provider-stop deadline, credential-free remote process guard and
separate hidden local provider guard. Expected end-to-end completion2–3.5h,
refined by preflight. Total cap USD25 including latency and temporary storage;
GPU envelope is USD18.36 training plus USD1.485 for up to90min RTX5090 latency.
No new network volume; unrelated resources remain untouched. Provisioning is
detached and logs/checkpoints survive SSH disconnects. Setup and cache preparation
overlap uploads; later latency setup overlaps the training tail.

Monitor training every5min and diagnostics every60s, shortening near completion.
Report step, loss, throughput and refreshed ETC. Warn on nonfinite/skipped updates,
source/cache mismatch, stale10min events, less than10% VRAM headroom, less than5GB
free disk or projected budget/deadline overrun. Download archives and verify all
member SHA256 values, run local scientific verification, then delete Pods and
audit that no run-owned resources remain. A deadline stop retains storage until
verified recovery and deletion.

## Final specialized-kernel measurement

The same frozen Run035 `k050-70m-v2` used in Run043, with its unchanged HZ
compatibility bridge. No kernel search. RTX5090 at USD0.99/hour, BF16, B1,T2048,
uncached full50304 logits. Native same-checkpoint CUDA graph is the denominator;
native eager is the correctness anchor. One eight-block smoke and three fresh
scientific processes,64 fixed inputs times7 paired passes each,1344 pairs total.
Every scientific process qualifies all338 blocks under the original logit bound
abs(error)<=.25+.02*abs(reference), relative L2<=.02 and pooled loss delta<=.001.
Retain raw pairs, all qualifications, complete operand/skip-work diagnostics and
source/runtime identities. These BF16 losses are separate from training validation.

Prelaunch verification, allocation receipts and closeout are recorded below.

Prelaunch checks: seven focused tests passed in4.39s, covering the exact matched
70M recipe, realized data schedule, real70M random parameter identity and h/z
hooks, h-only capture, kappa=.5 equality/gradient behavior, checkpoint roundtrip,
integer ceiling, three-process latency matrix and unchanged frozen kernel bytes.
The full bootstrap suite passed242 tests in7.79s. Commands were
`.venv/Scripts/python.exe -m pytest -q` with `test_run046.py` plus
`latency/test_protocol.py` and `--basetemp tmp/run046-tests-001`, and with `tests`
and `--basetemp tmp/run046-bootstrap-001`. The1374 historical kernel files and
17 port/bridge source identities verified during preparation. These CPU tests
do not substitute for the mandatory exact GPU preflight before training.

The allocated training Pod is `v7py0cfr2jd0ft`, one Secure H200 in US-NC-1 at
USD4.59/hour, created18:42:20UTC on20September2026. Its provider-stop deadline
is22:42:20UTC. The hidden workstation guard is armed. All42 Python files parse;
local free disk is766.8GiB. Prelaunch commit is
`b2f7e1838d4f5cf8eccf2c60c050aa1bba2fc910`; isolated deployment commit is
`e2b73c5b71bd52c3ce0d7565cebe76262a274649`. Source bundle001 is160526 bytes,
SHA256 `e9805bc86820dfc718f1c2e680c46b06cae66b5ab1dbab62d36583731ed1da5b`.
No account credentials are sent to the Pod. The live pre-creation audit found
no existing Pods, only the unrelated shared volume, which remains untouched.

The source bundle passed its remote hash check and the detached pipeline started.
Both canonical random tensor files uploaded in21.8s and passed full SHA256
verification before releasing preflight. The local stop guard PID is recorded
in an ignored file. The source-only latency bootstrap verifies all historical
and port sources; it will install the same runtime and precompile the five
unchanged norm, projection, RoPE, attention and joint builders with the final
benchmark cache/settings during the training tail. No model or scientific
measurement is used by this preparation. Its4,631,715-byte archive receipt and
syntax-checked infrastructure helpers are retained; the scientific training
source and configuration remain unchanged after launch.

Both caches match their pinned full hashes. Preflight passed all six boundaries;
the five timed updates were6.7389/6.7482/6.7487/6.7325/6.7251 seconds, median
6.7389s (about311k input tokens/s). Full validation took0.6264s, checkpoint save
0.4541s, full activation diagnostics2.9362s and logical diagnostics6.6947s. Peak
reserved10,492,051,456 of150,121,545,728 GPU bytes leaves ample headroom. Initial
parameter hash, schedule hash, both kappa=.5 gates and six h-only pressure tensors
matched, with no overflow/skipped boundary. The complete calibration is retained
in `prelaunch/remote-preflight-hz-h-ol1-kappa-0p5.json`.

Fresh scientific training was at3/712 updates at19:00:04UTC: task loss10.411805,
zero overflows, GPU97% busy and9.77GiB peak reserved. Early throughput277k tokens/s
includes step1 validation and checkpoint overhead; the preflight predicts about
80 minutes of steady training. The scientific code/configuration remain frozen.
The local reporting helper derives its runtime conclusion from the actual future
measurement and qualification status, without carrying forward Run043's slowdown
conclusion as a presumption about the new endpoint.

The final-latency Pod `5a44rl7iwfi0f9` is one Secure RTX5090 in EUR-IS-2,
allocated at20:08:21UTC for USD0.99/hour with a21:38:21UTC provider-stop
deadline. Its local guard was armed before deployment. The4,631,715-byte
source-only bootstrap passed its remote hash check. Runtime installation and
all five unchanged kernel builders completed successfully while training was
still active; compilation took231.5521s. This preparation contains no model
measurement. The unrelated Run045 Pod was observed and left untouched.

## Completed result and closeout

All712 updates completed in4862.321s, with no overflow or skipped update. Median
training throughput was309,721 input tokens/s. Final complete validation loss is
4.838033648. Pooled exact-zero rates are99.907744% at h and99.847285% at z;
model-wide logical-product opportunity R_model is15.426970%, against the declared
HZ analytic reach ceiling15.443084%. These logical quantities are not measured
runtime speedups. Full validation uses338 blocks from500 documents, excluding
the1,444-token tail, and no post-hoc clipping was performed.

All three fresh final-checkpoint latency processes qualified on all338 blocks.
Across1,344 paired measurements, the geometric-mean native latency is1.691950ms
and the frozen K050 latency is1.690154ms, yielding1.001063x. Process ratios are
1.003766x,0.997529x and1.001903x: effectively parity under this protocol, with no
clear runtime advantage. The complete operand/skip-work diagnostics from
replicate1 are retained. See [the final observation](observations/001-final-results.md)
and its generating script `17_report.py`; machine-readable results and source
SHA256 values are in `artifacts/summary.json`.

The4,714,244,974-byte training archive and all74 inventoried members passed local
hash verification, followed by `03_verify.py` reporting `verified 1`. All12 model
checkpoints and3 optimizer/scaler/RNG recovery states are retained locally,
totalling5,071,091,284 checkpoint bytes. The328,924-byte latency archive and all77
members passed verification; all3 scientific processes qualified. Archive hashes,
per-file hashes and transfer receipts are retained under `transfer/`,
`latency/transfer/`, and `prelaunch/`. Weights and binary archives are excluded
from Git. The final checkpoint content identity is
`d6a3813f83ce080b07637fe422bb7ab1b23a6793e39c365601301a661851d02f`.
The local verifier's JSON matched the remote result semantically; Windows changed
only its line endings. Original LF artifact bytes were restored and rehashed,
as recorded in `prelaunch/local-verification-check.json`, preserving transfer
provenance. The final report references those retained original bytes.

The H200 Pod was deleted at20:25:36UTC and the RTX5090 Pod at20:25:39UTC on
20September2026, after local verification. Both local deadline guards were
disarmed. A fresh resource audit found no Run046 Pods or endpoints; the unrelated
Run045 Pod and pre-existing100GB shared volume were left untouched. Quoted hourly
rates times elapsed allocation give USD7.898662 training plus USD0.285266 latency,
USD8.183928 total GPU estimate, excluding disk. Posted billing is still partial:
USD3.352652 for training and no latency record at the closeout read, so these are
not final invoice totals. Scoped deletion, billing snapshots and the final
resource audit are recorded in `prelaunch/teardown-*-001.json` and
`prelaunch/final-resource-audit.json`.
