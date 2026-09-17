# Run034 — Pythia-70M A4/A7 gates with h-only OL1

## Authorization and question

On 17 September 2026 the user selected the full five-threshold 70M grid from
Analysis023's ETC/cost comparison and explicitly instructed: "Launch all five
kappas for 70M in runpod ... tests run as fast as possible. Manage GPU availability
so we execute it asap." This authorizes preparation, checks, exact calibration,
capacity-managed parallel training, monitoring, retrieval and teardown for these
ten conditions. It does not launch 410M. The explicit instruction, after the
matched-design and resource estimates, supplies the authorization.

Does the favorable h-only pressure regime seen at 14M recur at 70M under A4
and A7 thresholding? Compare these endpoints with matched all-site Run018.
Favorable loss/sparsity operating points support transfer; weakened or reversed
tradeoffs limit it. This is one seed and fixed token budget, not a scaling law,
direct Q/K/V causal test or runtime claim. It addresses the pending 70M h-only
comparison in manuscript/draft/training-results.tex; no TeX edit is authorized.

## Matched scientific definition

A4-OL1(h) and A7-OL1(h), each at kappa = 0, .01, .05, .1, .5. A4 gates a,m,h,z
one-sided; A7 adds symmetric q_post,k_post,v (q/k after RoPE). Pressure is OL1
only at post-gate h, averaged over six layer-tensor means, lambda=1, b=1.
Compared with all-site OL1, both target composition and equal-tensor normalization
change. Gate and pressure placement remain independent.

Match Run018 model/recipe/runtime/data/seeds/training/validation/diagnostic
sections exactly, as checked by the contract test: pinned random Pythia-70M,
seed1234 canonical initialization, identical MiniPile schedule, FP32 parameters
and AdamW states, dynamic FP16/flash SDPA, 712 updates, MB4 x accumulation256,
1024 sequences/update, T2048 and 1,493,172,224 tokens/condition. The optimizer,
learning-rate schedule, clipping, dropout and checkpointing behavior are in
config.yaml and match Run018. No released weights are loaded.

Initial parameter SHA:
`e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`.
Schedule SHA:
`d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`.
The existing model/RNG artifacts are copied, not regenerated. Strict CPU load
and parameter hash precede CUDA transfer and optimizer construction.

Validation at step1 and reloaded final checkpoint covers all 500 documents,
338 complete 2048-token blocks, 692,224 tokens and excluded tail1444. Final
activation and eager logical passes have the same coverage. Retain exact/near
zeros at0/.001/.01, activation RMS/L2/finite counts, all parameter norms, integer
logical counts and analytic reach, plus every boundary's gradient interaction,
clipping, loss scale, capture identity and OL1 geometry. Observed R_model is in
[0,1]; analytic reach is separate and does not bound incidental ungated zeros.

Retain models at0,1,2,4,8,16,32,64,128,256,512,712 and full optimizer/scaler/
Python/NumPy/Torch CPU/CUDA RNG recovery at256,512,712. This matches the recent
14M retention pattern carried into the quoted estimate. No clipping sweep or
kernel benchmark; retained checkpoints/cache identity permit later diagnostics.

## Execution and safety of retained artifacts

Up to ten independent GPUs: H200 preferred, H100 SXM capacity fallback after
exact workload preflight. Multi-GPU Pods can share immutable cache/environment;
each condition owns one GPU, attempt and progress file. No DDP, precision or
batch changes. Use live Community/Secure capacity across regions and record
every allocation and fallback. The existing 100GB network volume is unchanged.

Current H200 rate is USD3.59/4.59 per GPU-hour and H100 USD2.69/3.49.
Balance is about USD523, account limit USD80/hour. Normal expected elapsed time
is2-3.5 hours. Operating allowance is **USD200**, including setup/retries/disk.
Four-hour on-Pod stop guards bound ten all-Secure H200s to USD183.60 compute
before disk. A stop preserves the Pod volume; no automatic deletion may destroy
unverified checkpoints. Stopped disks continue billing and must be reported if
a guard fires. Per-Pod deletion follows verification of all assigned conditions.

Use the Run018 image digest and pinned Python3.12/Torch2.11.0/CUDA12.8/
Transformers5.12.1 runtime. Allocate40GB container disk plus20GB base Pod volume
and20GB per GPU. Source, inputs, outputs and logs persist under /workspace;
the rebuildable environment may use /opt. Seed one verified6.25GB input cache
and initialization set, then distribute within the cloud.

Preflight each assigned GPU: five exact real-cache optimizer boundaries from
canonical initialization, first excluded from ETC; finite updates, exactly six
h captures, 10% memory headroom, full validation/diagnostics and checkpoint write.
Preflight creates no scientific attempt. Start training fresh only after it passes.
All ten conditions are authorized, with no scientific sentinel selection.

Read-only monitoring every five minutes, sooner at a projected completion or
declared warning. Report step, loss, throughput, refreshed ETC, memory and cost.
Warnings: missing process, ten-minute stale event, nonfinite/skipped boundary,
capture/hash/runtime drift, insufficient disk, or ETC/cost beyond the allowance.
Jobs and stop guards must survive SSH/controller disconnect.

Retrieve about5.07GB/condition (50.7GB total), plus diagnostics/logs: every model
and recovery snapshot, manifest/config, events, metrics, diagnostic counts/norms,
calibration, environment and execution records. Hash-verify locally, run the
condition verifier, then delete its Pod once every assigned condition is safe.
Finish with cohort verification and resource/billing audit. Weights/data stay
out of Git. Failed infrastructure attempts get separate retained control records.

## Implementation and checks

Reuse Run004's lifecycle, Run017's 70M model/diagnostics, Run018's canonical
loader and Run032's h-only OL1 boundary. Run034 adds the two-family grid,
full checkpoint inventory, exact calibration and condition/cohort verification.
Tests, remote preflight and execution evidence are retained under prelaunch/.
No scientific input is changed after its first scientific attempt starts.

## Allocation and infrastructure record, 17 September

The scientific implementation was frozen in commit `58e82fd` and passed all
249 local tests before provisioning. Each Pod received the same 116-file
source snapshot, whose receipt is `prelaunch/source-receipt.json`; SHA256 of
the compressed snapshot is
`f8c8c83c6300db11ff2a9fc8e3d1180758a3113038855af35139199758160b92`.
The receipt verifies exact deployed bytes and retains the source commit even
though the compact deployment is not a complete Git checkout.

Ten simultaneous GPU slots were acquired on six Pods: nine H200 SXM and one
H100 SXM. Four H200s use Community capacity; five H200s and the H100 use Secure
capacity. Their combined quoted compute rate is **USD40.80/hour**, plus disk.
See `prelaunch/execution-allocation.json` for condition-to-GPU assignments,
Pod identities, creation times and four-hour stop deadlines. The H100 runs
A4/h-only at kappa=0.5; hardware is recorded, with the scientific definition
unchanged. The other nine conditions use H200.

The original four-H200 seed Pod had slow network/package transfers. Before
any scientific attempt, its setup logs were copied and hash-verified, then
the Pod was deleted. Evidence is retained in
`prelaunch/infrastructure/seed-attempt001-slow-network/`. Capacity retries
were recorded individually; the final allocation replaces those four slots.
The replacement seed received the canonical model in 34.469 seconds via
native SCP. It rebuilds the training cache from the pinned source, verifies
the canonical token-file SHA256, and distributes one hashed input archive.
All six runtimes installed the pinned package set successfully.

`prelaunch/monitor_and_retrieve.py` runs detached on the local machine. It
checks every five minutes (or at the projected completion window), logs step,
task loss, throughput, warnings and refreshed ETC, and retrieves at most two
completed Pods concurrently. It verifies the complete archive hash, extracts
all models/recovery states/diagnostics/logs, and invokes the condition verifier
locally before permitting deletion of an exactly identified owned Pod.
Seven focused retrieval tests passed, including bad hashes, failed scientific
verification, incomplete conditions and mismatched Pod identities. Logs and
receipts are retained under `prelaunch/`; active files are not committed.
The on-Pod and independent local deadline guards remain active throughout.

The local download probe showed that waiting until the end to transfer all
50.7GB would unnecessarily extend retention on billable GPUs. The controller
therefore also recovers immutable checkpoints during training, using the final
checkpoint metadata file as the publication marker. It transfers at most two
streams concurrently, verifies every file's SHA256 before publishing locally,
and keeps an explicit per-file receipt. Terminal archives contain the remaining
logs and metrics; the final scientific verifier still checks the complete
checkpoint inventory before deletion. At 17:40 UTC, four snapshots (1.13GB)
were already verified locally. No training code or scientific input changed.

The slow input link to worker3 used a resumable retry and then four disjoint
byte-range streams. The complete reassembled archive still had to pass its
original SHA256 before extraction. The exact transport script is retained in
`prelaunch/worker3-striped-transfer.py`; this is an infrastructure retry only.

## Confirmed launch status

At **17:45:45 UTC (14:45:45 Sao Paulo), 17 September**, all ten conditions
were training, with steps 3--77 of 712 completed and no current overflow or
process warnings. All ten exact preflights passed and their reports are
retained as `prelaunch/remote-preflight-*.json`. They agree on the canonical
initialization, schedule, pinned runtime and scientific code identity
`f13879dd7bd521f1547b84ead4239fcb68db0ccc8ef135ef196396974fe11a2d`.
The immutable launch snapshot is `prelaunch/confirmed-launch-status.json`.

The slowest recent-step projection was about **3 hours remaining**, targeting
roughly **17:47 Sao Paulo** for the last training completion, with another
15--30 minutes of planning margin for final retrieval/verification. Actual
throughput varies substantially by host; the early per-GPU preflight training
projections span 1.37--3.14 hours. Expected total expense is **USD120--150**,
including setup/retry/retrieval allowance, within the USD200 operating allowance.
These are measured projections, not completion guarantees. The full calculation
and live status paths are retained in `prelaunch/launch-summary.json`.

The detached controller and all six on-Pod plus six independent local stop
guards were checked alive at handoff. Monitoring and checkpoint recovery are
active; the final checkpoints and scientific results are not yet complete.
A process-scoped Windows power request keeps the local recovery host awake
while the controller runs, ending no later than 21:30 UTC. It leaves permanent
power settings and display behavior unchanged.

## Progress and spending follow-up, 18:18 UTC

The user requested continued progress/ETC tracking to avoid unnecessary expense.
All ten conditions remained healthy at steps 143--354 of 712; none of the six
Pods was idle awaiting a completed run. About 14.93GB of checkpoints had already
been hash-verified locally. The longest recent-throughput estimate was 140.5
minutes remaining, or about 17:39 Sao Paulo for the last training completion.
Host throughput fluctuates, so the planning range remains about 2--2.5 hours
of training plus final retrieval. Current account spending was USD40.888/hour
(USD40.80/hour for this run's GPUs, plus storage). The planning total remains
USD120--150 against the USD200 operating allowance.

The separate read-only `prelaunch/track_cost_and_etc.py` now refreshes the
spending/ETC ledger every five minutes, using the primary controller's measured
progress, current provider resource state/rates, and recorded lease times.
It writes `latest-cost-etc.json` and `cost-etc-history.jsonl`, flags stale status,
missing resources, insufficient time before a stop deadline, or a projected
budget overrun. Cost projections include setup/disk/recovery reserves and are
estimates, not a provider invoice. The primary controller continues checkpoint
recovery and verifies locally before deleting each completed Pod independently.
The live reporting path was exercised successfully against all six Pods; no
scientific code or active training process was changed. The immutable follow-up
snapshot is `prelaunch/status-20260917T181832.json`.
