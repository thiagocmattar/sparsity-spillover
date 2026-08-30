# Run 011 staged RunPod playbook

This is a tracked execution sequence, not launch authorization. Stop at each
approval gate.

## Stage 1 - implementation and non-billable verification

1. Lock the five A4-Z conditions, full-pass recipe, Run 004 comparator, and
   one-condition-per-GPU mapping.
2. Implement run-local config, smoke/preflight, worker entrypoints,
   diagnostics, per-worker verification, cohort verification, and monitoring.
3. Run focused tests and the repository-wide bootstrap suite.
4. Reconcile current RunPod Pods, volumes, auth, SSH, live A100 capacity, and
   prices without creating anything.
5. Report tests, historical resource fit, preliminary ETC, exact preflight
   resource, maximum billable duration/cost, transfer inventory, and warnings.
6. Stop for explicit approval of the one-Pod billable preflight.

## Stage 2 - billable A100 preflight

1. Re-query live capacity and price. Create exactly one approved guarded Secure
   A100 80 GB Pod, recording its ID even if readiness waiting fails.
2. Verify SSH, pinned image/runtime, source identity, package lock, caches, and
   disk before running GPU work. Use the existing volume only if placement is
   `EUR-IS-1`; otherwise stage one hash-verified cache onto the Pod volume.
3. Run `05_remote_preflight.py`: five exact MB32/GAS32 boundaries at each of
   `kappa=0` and `kappa=0.5`.
4. Reject any nonfinite loss or gradient, skipped boundary, wrong topology,
   wrong initialization/schedule hash, missing flash kernel, or reserved memory
   above 90 percent of device memory.
5. Retrieve and hash-verify the preflight JSON, log, environment lock, and
   transfer inventory locally.
6. Stop the successful Pod to release GPU compute billing while preserving its
   `/workspace` for at most 24 hours; otherwise retain failure evidence and
   terminate it.
7. Use measured step timings plus matched Run 004 validation, diagnostics,
   checkpoint, setup, and transfer evidence to refresh the five-worker ETC and
   maximum cost.
8. Present the pre-launch post-hoc checklist and stop for explicit scientific
   launch approval.

## Stage 3 - five-worker scientific launch

1. Reconcile all resources and re-query live prices/capacity.
2. Restart the retained preflight Pod and give it a fresh absolute termination
   deadline. Create four more guarded one-GPU Pods within the approved envelope.
3. Prepare the four new workers concurrently. Verify source, environment, cache,
   config, code, initialization, schedule, CUDA, and flash attention on every
   worker before launch.
4. Launch one detached `02_train.py --worker <condition-id>` process per Pod,
   with PID and logs under `/workspace`.
5. Confirm the first finite training event from all five workers before entering
   the normal monitoring interval.

## Monitoring

Poll every five minutes unless refreshed ETC is shorter, in which case wait for
the projected completion window. Every update reports worker step and input
tokens, task loss, throughput, refreshed ETC, loss scale/overflow, GPU memory and
utilization, disk, process health, and last-event age.

Warn on an absent process, a ten-minute stale event, nonfinite value, skipped
boundary, memory above the preflight envelope, less than 5 GB free disk,
incomplete validation, or projected termination-deadline overrun. Monitoring is
read-only and never starts a duplicate worker.

## Retrieval and teardown

For each terminal worker:

1. Run `03_verify.py --condition <condition-id>` remotely.
2. Record the transfer inventory and archive byte count/SHA-256.
3. Copy the complete attempt tree locally and recompute every agreed hash.
4. Run the same per-condition verifier locally.
5. Delete that exact Pod only after local verification passes.

After all five attempts are local, run the cohort `03_verify.py`, re-list Pods
and volumes, report every retained resource and continuing cost, and refresh
posted billing after provider ingestion settles.
