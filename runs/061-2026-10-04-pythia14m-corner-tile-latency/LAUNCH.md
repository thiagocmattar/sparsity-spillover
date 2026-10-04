# Run061 launch proposal — approval pending

The user explicitly confirmed the four-checkpoint benchmark design and then
confirmed the proposed diagnostic inventory as sufficient. This proposal
requests launch approval under root AGENTS.md; design approval has not created
billable resources or launched the scientific cohort.

## Implemented workload and tests

Four retained final Pythia-14M T2/Ph models: scalar and causal corner8x16 at
kappa 0/.1, step 712, seed 1234. No retraining or changed weights. Native,
adapted K050 sparse and matched no-h/z-skip graphs execute the exact original
operators. The latter disables h/z tile bypass and short-row substitution;
all other K050 settings stay matched. Recurring gate/support overhead and full
logits remain timed. Equal input staging and all setup/compilation are excluded.

The archived scalar consumers remain untouched. Run-local `joint.cu` adds an
explicit scalar/corner gate mode to K049 arithmetic and scheduling; corner
equality drops the tile, scalar equality survives, and admitted corner tiles
preserve signed interiors. The code, derivation diff, tests, controller,
diagnostics, reduction and PDF/observation script are in this folder.

- Seven focused tests pass: four real CPU checkpoint forwards, original
  model/initialization/schedule identities, gate equality/signed values/causal
  prefixes, eight optimized installation cases with unchanged weight objects,
  paired estimator rejection, full cache/source coverage and count serialization.
- All 242 bootstrap tests pass in 11.03 seconds.
- Sixteen synthetic CUDA checks pass on the existing RTX5070TiLaptop:
  both widths, both gates, both thresholds, changed-input graph replay,
  branch/residual outputs and independent instrumented-counter verification.
- Four real native CUDA checkpoints each pass two B1/T2048 blocks with zero
  observed logit error versus their eager anchors. All four graphs resident
  together peak at 3.426 GiB allocated / 3.912 GiB reserved. These smokes cover
  eight model/block cases, not complete validation or optimized K050 execution.
- All 1,403 frozen source/input identities pass. Python sources compile;
  shell and guard syntax plus sealed archive inventory are checked before commit.

Mandatory post-launch qualification: repeat component checks on the RTX5090,
precompile the actual dependency chain, then run both four-input/eight-block
threshold-pair smokes, covering all native/no-skip/sparse implementations.
Stop if any smoke fails. Each of the six full processes subsequently qualifies
every implementation against its own eager reference on all 338 validation
blocks / 500 documents / 692,224 inputs / 691,886 predictions, with the
1,444-token tail excluded and reported. Bounds remain atol .25, rtol .02,
relative L2 <= .02 and pooled loss delta <= .001.

## Resource choice, price and ETC

Live Secure catalog read on 4 October 2026 at 14:25:51 UTC: RTX5090, 32 GB,
LOW availability in EU-RO-1, USD .99/hour, eligible host CUDA 13.0/13.2.
The refreshed deep probe at 14:36:00 UTC found CUDA13.0 OUT and CUDA13.2 LOW
at USD .99/hour. Use currently available host CUDA13.2 with the pinned
CUDA12.8 application runtime; recheck immediately before creation.
H100 SXM was also available at USD3.49/hour; it changes the measured hardware
and requires fresh qualification of a kernel previously verified on RTX5090.
The RTX5090 choice preserves the approved prior-workload comparison and has
ample headroom; it is not a cheapest-GPU fallback. No concurrent timing
processes will contend for this one device.

The local laptop fits the development smoke, but the approved scientific
cohort uses the RTX5090 to match the previous inference protocol. Six graph
implementations per threshold are conservatively expected to fit within
8–10 GiB on its 32 GB; exact optimized fit is checked in the remote smoke.

Proposed one owned Pod: `run061-corner-tile-latency-001`, Secure, one
`NVIDIA GeForce RTX 5090`, currently available host CUDA13.2. Current eligible
Secure placement is EU-RO-1; do not attach the existing EUR-IS-1 network volume.
Pinned image:
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Python3.12 / Torch2.11.0+cu128 / Transformers5.12.1 / NumPy2.5.0, with the
retained complete package lock and uv0.8.15. Use 40 GB container disk and a
25 GB `/workspace` Pod volume. No new network volume.

Initial end-to-end ETC: **40–75 minutes** from provisioning through verified
retrieval and analysis, not from steady-state kernel time. Approximate budget:
provision/transfer 2–5 min; pinned installation/cold compilation 15–30 min;
both smokes 2–5 min; six full processes and diagnostics 15–25 min; retrieval,
verification and analysis 3–10 min. Previous Run059 full processes took
99–168 seconds with fewer implementation modes. This run adds the no-skip
graphs and full validation, while placing runtime/vendor files on local disk
before first compilation to avoid its earlier slow-storage retries. Cold
compilation/host storage remain the main ETC uncertainty; refresh after smoke.

Hard maximum: **2 GPU hours from provider creation, USD3 total**. GPU ceiling
is USD1.98. At current storage rates, 65 GB at USD.10/GB/month for two hours
is about USD.018 (730-hour month); USD3 leaves a conservative storage margin.
Storage rates and persistence are sourced from
[RunPod storage documentation](https://docs.runpod.io/pods/storage/types).
Re-read stock/price immediately before creation. A price above .99/hour or an
alternative GPU is outside this concrete proposal.

## Transfer, persistence and closeout

Upload only four final inference checkpoint sets (config, model safetensors,
generation config and metadata), their training provenance, full validation
cache, frozen dependencies and run-local code. Optimizer states and all original
models/recovery remain local. The sealed input bundle is approximately 250 MiB;
its exact byte/SHA inventory and receipt live in `bundles/`. The final upload
is `input-004.tar`, 262,471,680 bytes, SHA256
`7cdde1cc6bfc09380c984f26d8e6a19c757e14257e470c34d2613175dd27f3c0`;
all 1,432 members are verified. Earlier unlaunched bundles are retained as
preparation records and are not execution inputs.

Logs, inputs, original sources and all measurements stay under `/workspace`.
The pinned runtime and hash-verified vendor copy are on fast container disk
before compilation. No new results depend on that temporary runtime surviving
a stop; the pinned source/environment identity is retained. Setup, component
checks, smoke and full measurement run detached with persistent logs.

Return every successful/failed attempt, raw JSONL timings and input indices,
numerical/loss checks, per-site/layer counts/norms/tiles, instrumented kernel
work, source/GPU UUID/runtime identities, setup/compile metadata, all logs and
transfer inventory. Expected payload <25 MB; retain the actual size and SHA256.
No new checkpoint is produced by this inference-only workload.

Monitor read-only every60 seconds, or the projected completion window. Report
stage/block/process progress, current loss/error, measured throughput and
refreshed ETC; setup has no current loss. Warn/stop on OOM, nonfinite output,
qualification failure, 900-second leaf timeout, stale measurement progress
over300 seconds, disk shortage or approaching lease deadline. Cold compiler
activity is checked separately from scientific-progress staleness.

Before upload/compute, arm an identity-scoped hidden workstation provider-stop
guard and a credential-free remote process-group deadline. Provider stop at the
absolute two-hour deadline is the billing backstop; it retains the workspace for
retrieval. Normal closeout verifies remote success/failure and every downloaded
archive/member SHA locally, then terminates the owned Pod, confirms its absence,
disarms its guard and audits remaining resources. Final analysis produces one
PDF plus table and observation; sources/results are committed without weights,
caches, archives, credentials or a push.

Current inventory contains only foreign stopped Pod `qa00sm05lmrs7w` and the
pre-existing100GB EUR-IS-1 network volume `9luykg5yc3`. Neither will be changed
or reused. Their continuing storage is not a cost of Run061.

## Approved execution record

The user explicitly approved this packet with "Aprovado" on 4 October 2026.
The live read immediately before creation found RTX5090 Secure stock Low on
host CUDA13.0 and13.2, both USD.99/hour. Pod `a7kx4d05y0spap`, named
`run061-corner-tile-latency-001`, was created at14:40:30.515 UTC with the
approved pinned image, one RTX5090, 40GB container disk and25GB workspace.
The absolute stop deadline is16:40:30.515 UTC. The independent hidden provider
guard logged ARMED before upload. Full deployment/cost/checkout identities
are retained in `prelaunch/lease-001.json`.

The first SSH discovery occurred while the image was provisioning and did not
yet advertise an IP. It failed before upload or scientific execution; wait for
SSH readiness on the same Pod. This is infrastructure readiness, with unchanged
sealed input004 and no additional resource creation. A subsequent live SSH
discovery identified the actual cause: the created Pod did not publish22/tcp,
despite `startSsh:true`. The owned Pod's ports were updated to22/tcp before
upload; this changes transport only. The original deadline and billable cap
remain unchanged.

## Verified closeout

The initial scientific attempt failed only in diagnostic module resolution,
after full numerical qualification. Its first36-member SHA-verified retrieval
is preserved. Infrastructure retry002 preloads the unchanged run-local modules,
repeats both smokes and all six full processes, and uses the same Pod/deadline;
see `infrastructure/retry-002/README.md`. Scientific inputs are unchanged.

All six accepted processes are complete and qualified. Each threshold has
1344 paired observations and full338-block diagnostics. Final transfer002 has
120 files/19,607,531 uncompressed bytes, verified archive and every member SHA.
The reducer confirms same GPU, original inputs, bootstrap identities and full
coverage before accepting estimates. Its table, PDF and observation are local;
the PDF has been rendered and visually checked.

After local verification, Pod termination returned204 and a live Pod listing
confirmed absence at14:58:52 UTC. The hidden provider guard was disarmed.
Elapsed creation-to-termination is1101.485 seconds, estimated GPUUSD.302908
atUSD.99/hour, plus storage. Scoped official billing has no posted records yet;
this is a provisional wall-clock estimate. No owned Run061 resource remains.
The post-closeout inventory also includes two newly appearing foreign running
Pods, `64aa1by0fk9p67` and `8t9ep4a5vf1fxf`, alongside the original stopped
Pod and shared network volume. No foreign resource was modified. Exact safe
inventory and billing metadata are in `prelaunch/closeout-001.json`.
