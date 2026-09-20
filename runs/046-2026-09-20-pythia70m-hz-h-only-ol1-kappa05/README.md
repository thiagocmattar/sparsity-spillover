# Run046: Pythia-70M T_hz/P_h at kappa=0.5

The user explicitly requested one additional matched 70M RunPod run at kappa=0.5,
as fast as possible. This selects the Run043 design and authorizes its execution
with the same diagnostic package and no post-hoc clipping. Exactly one training
condition is implemented: `hz-h-ol1-kappa-0p5`. Status: preparing for launch.

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

Prelaunch verification and allocation receipts will be appended below.

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
