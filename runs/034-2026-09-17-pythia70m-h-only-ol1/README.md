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
