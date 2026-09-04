# Run 023 deployment playbook

This is an implementation record, not launch authorization. Refresh all
RunPod state immediately before provisioning.

## Proposed first phase

Use one on-demand Hopper Pod for all six sentinels so reported runtime is from
one exact GPU SKU. Prefer H100 SXM 80 GB; try H100 NVL and then H200 if needed.
Use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 60 GB container disk,
and 20 GB Pod volume mounted at `/workspace`. Do not attach or create a
network volume.

The provisional external deletion guard is four hours. It is a spending
ceiling, not an assertion that the workload must finish in four hours. Before
launch, combine the live GPU rate with Pod-disk rates, current balance, and the
continuing cost of any pre-existing resource; report maximum compute, storage,
and total account outflow. The first post-setup timings recalibrate ETC and
cost-to-completion. Stop and retrieve diagnostics if the projection crosses
the approved guard.

The 2026-09-04 16:01 UTC live audit found zero Pods and zero endpoints. The
only continuing resource is the pre-existing, unattached 100 GB standard
network volume at approximately `$0.00972/hour`. Exact CUDA-12.8 Pod capacity
was low for all three community candidates: H100 SXM at `$2.69/hour`, H100 NVL
at `$2.59/hour`, and H200 SXM at `$3.59/hour`. The proposed 80 GB of transient
Pod storage adds approximately `$0.01111/hour`. The corresponding four-hour
total-account ceilings, including the existing volume, are `$10.84`, `$10.44`,
and `$14.44`. Secure H100 SXM and H200 were also low at `$3.49/hour` and
`$4.59/hour`; using Secure Cloud requires an explicitly expanded ceiling.

The pre-calibration ETC is 1.5--2.5 hours: about 20--35 minutes for transfer,
two pinned environments, the official positive control, and compilation, then
roughly 70--115 minutes for six sentinel conditions. This is an engineering
range, not a measured Run-023 calibration. Community incremental cost over
that range is `$4.05--$6.75` on H100 SXM, `$3.90--$6.50` on H100 NVL, or
`$5.40--$9.00` on H200. The first setup/control and condition timings replace
this estimate. The connected RunPod API exposes billing but not account
balance, and the isolated browser session is logged out, so balance remains a
launch-time human check rather than an inferred value. The machine-readable
snapshot is `prelaunch/runpod-audit-20260904.json`.

## Local pre-launch

1. Run `01_static_preflight.py --phase all`.
2. Check that the run patch applies to a clean checkout at the pinned commit.
3. Run the focused Run-023 tests and the complete bootstrap suite.
4. Run `00_prepare_inputs.py --phase sentinel --inventory-only` and inspect
   the allowlist. It must contain six final model-only checkpoints, their
   logical-product JSON files, the public validation cache, and code only.
5. Build `tmp/run023-sentinel-inputs.tar`; record its byte count and SHA-256.
6. Query zero/known Pods, endpoints, volumes, live balance, H100
   SXM/NVL/H200 stock, data center, cloud type, and exact hourly rates.
7. Record the approved GPU, rate, storage, four-hour deadline, maximum cost,
   transfer method, archive hash, and teardown guard in a new attempt directory.

Do not provision unless the visible balance covers the selected four-hour
total-account ceiling with a small margin. Do not infer balance from historical
billing.

## Pod execution

1. Transfer the archive into `/workspace`, verify its SHA-256, then extract
   it as `/workspace/sparsity-spillover`.
2. Start one disconnect-safe worker:

   `bash /workspace/sparsity-spillover/runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/05_start_worker.sh /workspace/sparsity-spillover sentinel`

3. Setup creates two pinned checkouts. The clean checkout runs the official
   SparseLM0.5B control; the second receives the exact Run-023 patch and is
   used only for Pythia.
4. The derived remote preflight must compile and pass the Run-022 N=128
   regression plus every Pythia-70M linear/QK/PV shape before checkpoint
   benchmarking.
5. Expected subsequent stages are six condition loops through source
   validation, dense occupancy validation, sparse validation, full-model
   timing, linear primitives, A7 attention compositions, verification, and
   packaging.

## Monitoring

Wait ten minutes between read-only checks. Each update reports stage,
condition progress, current running validation loss when applicable,
throughput, refreshed ETC, accrued cost, projected remaining cost, balance
required to completion, guard headroom, GPU utilization/memory/power, and log
tail. If the current ETC is under ten minutes, wait to the projected completion
window instead.

Declared early warnings are: no progress for 20 minutes while utilization is
under 5%, nonzero worker exit, source/patch/hash mismatch, compile error, CUDA
OOM, nonfinite output, correctness/loss gate failure, disk exhaustion, balance
below projected need, or ETC beyond the approved deadline.

## Retrieval and teardown

1. Require `status.txt=complete`, `exit-code.txt=0`, and
   `verification.json: passed=true`.
2. Download the result tar and SHA-256 sidecar before stopping compute.
3. Verify the archive locally, extract it under
   `artifacts/attempts/<attempt-id>/incoming/`, and rerun `04_verify.py`.
4. Reconcile the exact GPU, environment freezes, upstream clean/derived
   identities, patch, validation coverage, timing blocks, and all result files.
5. Terminate the Pod only after hashes and local verification pass. Confirm
   zero unintended Pods/endpoints/new volumes and report the final bill.
6. Add an observation Markdown file and any PDF only after scientific review.
   The remainder phase needs a separate decision and launch confirmation.

## Failure handling

- Allocation failure creates an infrastructure-attempt record, then tries the
  next approved Hopper SKU; it does not alter scientific inputs.
- Setup or transfer retries use a new attempt directory within unchanged Run
  023.
- Official-control or derived correctness failure stops before checkpoint
  benchmarking and preserves compiler/preflight evidence.
- Sparse slowdown is retained as evidence. A7 attention non-break-even blocks
  attention promotion but does not invalidate linear results.
- Never terminate billable compute before agreed artifacts verify locally,
  unless the hard approved spending deadline requires stopping; in that case,
  retrieve the available failure evidence first.
