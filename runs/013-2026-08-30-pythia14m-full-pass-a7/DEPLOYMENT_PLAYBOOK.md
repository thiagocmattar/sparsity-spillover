# Run 013 autonomous staged RunPod playbook

The user pre-approved every stage on 2026-08-30. The gates below are evidence
checks, not pauses for further permission.

## Stage 1 - implementation and non-billable verification

1. Lock the five mixed A7 conditions, full-pass recipe, Run 011 matched A4
   comparator, and one-condition-per-GPU map.
2. Implement run-local config, smoke/preflight, worker entrypoints,
   diagnostics, standalone/cohort verification, and monitoring.
3. Pass focused, affected-run, and complete repository tests.
4. Record a clean tracked commit and create a source bundle from that commit.
5. Reconcile RunPod account, Pods, volumes, SSH, live A100 capacity/price, and
   the exact termination mechanism without creating a resource.

## Stage 2 - guarded A100 preflight

1. Re-query capacity and price, then create exactly one Secure A100 SXM 80 GB
   Pod with a 1.5-hour absolute `terminateAfter` deadline.
2. Transfer the clean source bundle, clone it under `/workspace`, verify commit
   and Run 013 config/code identities, and install the pinned environment.
3. If placed in `EUR-IS-1`, inspect the retained volume and bind only a cache
   whose metadata, byte counts, and SHA-256 match. Otherwise stage or rebuild
   one verified cache under `/workspace`.
4. Run `05_remote_preflight.py`: five exact MB32/GAS32 boundaries at each of
   `kappa=0` and `kappa=0.5`.
5. Reject wrong identity/topology/gates, nonfinite loss/gradient, a skipped
   boundary, unavailable flash attention, or reserved memory above 90 percent.
6. Retrieve and hash-verify the preflight JSON, log, environment freeze, and
   inventory. Retain the successful Pod as the first worker only if its
   workspace and deadline can be safely refreshed; otherwise delete it.
7. Lock measured ETC and maximum cost in `prelaunch/launch-plan.json` and
   continue under the user's existing authorization.

## Stage 3 - five condition-parallel workers

1. Reconcile all resources and re-query live price/capacity.
2. Give every worker its own 2.5-hour absolute `terminateAfter` deadline.
3. Prepare workers concurrently. Verify source, environment, cache, config,
   code, initialization, schedule, CUDA, and flash on each worker.
4. Launch one detached `02_train.py --worker <condition-id>` process per Pod,
   with PID and logs below `/workspace`.
5. Confirm one finite training event from every worker before the first normal
   monitoring wait.

## Monitoring

Poll every five minutes unless refreshed ETC is shorter. Report each worker's
step/input tokens, task loss, throughput, ETC, loss scale/overflow, memory and
utilization, disk, process health, and event age. Warn on an absent process,
ten-minute stale event, nonfinite value, skipped boundary, memory beyond the
preflight envelope, less than 5 GB free disk, incomplete validation, or
deadline overrun. Monitoring never launches or mutates a worker.

## Retrieval and teardown

For every terminal worker:

1. Run `03_verify.py --condition <condition-id>` remotely.
2. Record the transfer inventory and archive byte count/SHA-256.
3. Copy the complete attempt tree locally and reconcile every hash.
4. Run the same standalone verifier locally.
5. Delete that exact Pod immediately after acceptance.

After all five attempts are local, run the cohort verifier, re-list Pods and
volumes, record the intentionally retained pre-existing volume and its cost,
and refresh posted Run 013 billing after provider ingestion settles.
