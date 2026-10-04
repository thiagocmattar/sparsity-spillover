# Run062 proposed launch

Design and diagnostic inventory are confirmed. Launch is pending a separate
explicit user confirmation. No Run062 Pod exists and no scientific update has
executed. This packet covers both approved14M T7/Ph tiles at kappa0/.1 and their
four-checkpoint inference follow-up, within a proposed USD30 total envelope.

## Checks and resource fit

Training:26 focused tests and242 bootstrap tests passed. The real14M CPU smoke
passed one T16/MB1 FP32 boundary per condition with identical approved initial
parameters, exact h-only six-tensor capture and finite losses. Tests include
FP32/FP16/BF16 equality, head/batch/tile boundaries, signed interior values,
detached gradients, causal prefixes/logits, exact downstream ports, pressure,
checkpoint/recovery serialization, full-block coverage and integer pooling.
CUDA component checks:32 passed on RTX5070TiLaptop, including all scalar/corner
gate modes, headwise operands and changed-input CUDA graph replay. These are
development checks, not research timings. Four full-model development forwards
completed on two blocks each, using retained scalar weights and explicitly
untrained tiles. c36 and tile kappa .1 passed the bounds; c39 and tile kappa 0
failed the optimized-versus-eager logit bounds. All native graphs passed.
The failed qualification is retained, not counted as a green test.

An operand trace located the first difference in attention: the laptop's native
Flash kernel is unsplit (128x128), while frozen K035 uses the measured RTX5090
two-split KV schedule. First-layer a/m, QKV and post-RoPE gates match exactly;
the h/z consumer matches native projection/addition arithmetic for its operands.
Small attention differences propagate into later threshold decisions. This
explains a local dispatch difference, but does not prove final RTX5090 equality.
Keep the exact agreed bounds; mandatory target-GPU smokes and full338-block
qualification must pass before accepting or plotting an optimized latency.

Unchanged MB32/GAS32/T2048 production training cannot fit the local12GB GPU.
Run060's two-H100 tile T2 preflights reserved61,467,525,120 of85,017,493,504 bytes
(72.3%), with five post-warmup full-update medians3.3822/3.3821s. T7 adds gates
and must be calibrated: both exact conditions run six complete preflight updates,
retain five timed updates, full validation/diagnostics and checkpoint roundtrip.
Reject identity/capture drift, overflow/skips, nonfinite results, >90% memory,
failed diagnostics or median update>6.8s. Never silently reduce the batch.

## Hardware, current prices and envelope

Catalog snapshot in `prelaunch/catalog-and-inventory.json`:
two Secure H100SXM80GB have HIGH global availability, USD3.49/GPU-hour.
H200141GB is USD4.59/GPU-hour, with MEDIUM availability;
its workload advantage is unmeasured. Two B200s have NONE availability.
RTX5090 is USD.99/hour with NONE availability at 17:29 UTC. Re-read stock and actual quotes
immediately before creating each Pod.

Training: one Secure Pod `run062-t7-corner-tile-training-001`, two H100 SXM GPUs,
one independent condition per GPU, no DDP. Keep placement broad across currently
eligible centers; the existing network volume is in EUR-IS-1, outside the current
H100 offer, so do not attach it. If a two-H100 host cannot allocate promptly,
one two-H200 host may be used only within the same two-GPU/duration/total-cost
envelope and unchanged precision/MB/GAS recipe, with an explicit receipt and
fresh exact-condition preflights. No extra concurrent training GPUs.

Training hard running limit:2h from provider start. H100 maximum GPU cost
USD13.96; H200 maximum USD18.36. Pinned image digest in config,40GB container
disk plus50GB persistent Pod volume at `/workspace`, existing registered SSH key.
No new network volume. All source/cache/logs/checkpoints persist on `/workspace`;
long processes are detached. Fetch/rebuild pinned caches once and verify both
canonical hashes before the two workers share them.

Inference: after training artifacts are verified locally and its Pod terminated,
one Secure RTX5090 Pod `run062-t7-corner-tile-latency-001`, one GPU, same pinned
image,30GB container plus40GB Pod volume, hard running limit1h, max GPU USD.99.
All six scientific timing processes are sequential on the same GPU. No accepted
timing runs until both final-checkpoint smokes pass. Failed numerical checks stop
the cohort and are retained. Current availability may delay this phase.

Total proposed hard ceiling:USD30, including GPU/storage/quote variation.
Catalog maximum GPU sums are USD14.95 (H100) orUSD19.35 (H200); allow <=USD.20
estimated short-lived Pod storage. Reconcile actual provider prices in leases;
do not create a resource whose offered cost exceeds the envelope. Both Pods get
a credential-free workload guard and an independent hidden workstation provider
stop guard, armed before setup/upload. At deadline stop and preserve unverified
artifacts. Stopped volumes keep billing; retrieve and verify promptly, then delete
the owned Pod. Normal completion transfers and verifies before deletion.

ETC: training/setup/retrieval90-120min, based on Run060's measured87min total,
16min pinned cache rebuild and ~40min training. T7 timing remains uncalibrated.
Inference setup/compile/qualification/timing/retrieval20-45min from the previous
small RTX5090 cohort, with a1h backstop. Initial total about2-3h after launch,
excluding approval waiting and stock delays. Refresh from actual preflights,
transfer throughput and process progress; stop before an unapproved overrun.

## Transfer and evidence inventory

Training upload: sealed167,468,861-byte Git bundle from commit
`03dff662132c9d2b20e4c513dc34c6b5cda4daa8`,56,279,344-byte
untrained replay; verify both remote hashes and the remote Git commit. Rebuild
the pinned ~6GB train token file on Pod, rather than uploading it over an
uncalibrated workstation link. Preserve cache identities and block-order hash.

Training download: all configs/manifests/events, full loss coverage, activation
and tile diagnostics at0/32/128/712, weight norms, logical counters, every-boundary
gradient/OL1 metrics,24 model checkpoints and6 optimizer/scaler/RNG states,
preflight/runtime/control logs and byte/SHA256 inventories. Expect ~1.9-2.1GB.
Verify the archive, every member and the complete two-condition cohort locally.

Inference upload: immutable frozen consumer/vendor sources (~30MB), four final
checkpoints (~225MB), full validation (~2.8MB), provenance and pinned packages.
Final tile inputs are created only from the verified training finals. Native
development models never enter the scientific bundle.

Inference download: all raw synchronized-host/CUDA samples, input indices,
process results, numerical/full-loss checks, per-layer/site zeros/near-zero/RMS/L2,
empty8x16 tiles for all seven sites, weight norms, logical lower-bound and kernel
work counters, runtime/compile/control logs and hash inventory (previously ~20MB).
All final checkpoints remain local. No clipping sweep was requested. Original
canonical FP16 quality and new BF16 numerical qualification are separate fields.

## Monitoring, results and closeout

During training monitor read-only every5min with idle waits; inference every60s.
Each update: active step/stage, current finite loss, throughput, refreshed ETC,
GPU memory/utilization, disk and event age. Check at the projected completion
window when sooner. Warn on10min stale events, missing process, nonfinite/skipped
boundaries, capture drift, >90% memory, <5GB free disk, failed coverage/qualification,
transfer/hash mismatch or projected deadline/cost overrun.

Produce loss/perplexity, sitewise sparsity and empty-tile comparisons, measured
latencies and a source-linked PDF comparison against the orange T7/Ph series in
Analysis041. Retain same-session paired scalar controls separately from historical
coordinates. Gate producers/fusion differ from the historical orange kernel;
the new exact gate passes are included in the timer. No equal-quality or decoding
claim, manuscript edit or finding promotion is authorized.

After agreed artifacts pass local hashes and scientific verifiers, delete only
Pods created by this launch and confirm their absence. Preserve existing foreign
Pods and the100GB shared network volume. Record provisional GPU/storage costs,
refresh posted billing when available and disclose any retained billable storage.

## Historical packet: executed and closed

The user separately approved this packet. Both training conditions and the
four-checkpoint RTX5090 follow-up completed with unchanged scientific inputs.
All agreed artifacts are verified locally; both owned Pods are deleted and
their absence confirmed. The README's final closeout and
`results/compute-closeout.json` record actual source identities, results,
resource ownership and provisional cost. Earlier future-tense statements in
this file preserve the prelaunch proposal, rather than current execution status.
