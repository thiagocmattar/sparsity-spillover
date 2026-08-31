# Run 014 autonomous RunPod playbook

The user pre-approved every stage on 2026-08-31. The gates below are evidence
checks: a failed gate stops or retries unchanged infrastructure; it does not
silently change the scientific design.

## Stage 1 - implementation and non-billable verification

1. Lock the five matched A7-OL1 conditions and the Run 013 comparator identity.
2. Implement run-local config, FP16 OL1 boundary, endpoint preflight, worker
   entrypoints, diagnostics, standalone/cohort verification, and monitoring.
3. Run focused, affected-run, and repository-wide tests plus Python compilation.
4. Record live Pods, volume, A100 price/capacity, CLI/auth/SSH readiness, ETC,
   maximum cost, transfer inventory, warnings, and teardown definition.
5. Commit only the launch-ready Run 014 files and create a clean Git bundle.

## Stage 2 - guarded exact preflight

1. Re-query resources, price, and per-data-center stock. Create one Secure A100
   SXM4 80 GB seed Pod with the pinned image, 30 GB container disk, 25 GB
   `/workspace` storage (or the unchanged retained volume in `EUR-IS-1`), SSH,
   and a 1.5-hour absolute termination deadline.
2. Transfer the clean Git bundle, clone it under `/workspace`, verify the commit
   and Run 014 config/code hashes, install the pinned environment, and verify the
   exact train/validation cache byte counts and SHA-256 identities.
3. Run `05_remote_preflight.py`: five exact MB32/GAS32 boundaries at both
   `kappa=0` and `kappa=0.5`, with mixed A7 gates and seven-site OL1.
4. Reject any wrong identity/topology/42-tensor pressure set, absent Flash SDPA, nonfinite
   metric, skipped boundary, trust-ratio violation, or reservation above 90% of
   device memory. Preserve and retrieve failure evidence before deleting a bad
   Pod. An unchanged retry gets its own attempt record and deadline.
5. If it passes, retrieve and verify the preflight packet, then delete the
   preflight Pod before creating the scientific fleet.

## Stage 3 - parallel scientific fleet

1. Create five guarded Secure A100 SXM4 80 GB Pods, allowing any
   approved live data center. Transfer the same source bundle and install the
   pinned environment concurrently.
2. Copy the immutable cache concurrently onto isolated Pod volumes and verify
   bytes and SHA-256 on every worker.
3. Run a lightweight remote readiness check on every worker: exact source,
   environment, CUDA/GPU, cache, and clean-tree identity. The scientific
   entrypoint then asserts config/code, initialization, schedule, and Flash SDPA
   identities before emitting its first event.
4. Launch one detached `02_train.py --worker <condition-id>` process per Pod,
   logging PID/stdout/stderr under `/workspace`. Confirm the first finite train
   event from all five before normal monitoring.

## Monitoring

Poll every five minutes unless refreshed ETC is shorter, then wait for the
projected completion window. Each update reports condition, step/input tokens,
task and pressure loss, throughput, refreshed ETC, loss scale/overflow, OL1 raw
and final ratios/trust scale, GPU memory/utilization, disk, process state, and
event age.

Warnings are: missing process, ten-minute stale event, nonfinite value, skipped
boundary, pressure-capture identity drift, trust ratio above 1, memory beyond the preflight envelope, less than
5 GB free disk, incomplete validation, or projected deadline overrun.
Monitoring is read-only and never starts a duplicate process.

## Retrieval and teardown

For each terminal worker:

1. Run `03_verify.py --condition <condition-id>` remotely.
2. Build and record the transfer inventory and archive bytes/SHA-256.
3. Copy the complete declared attempt archive locally and match the remote hash.
4. Extract, reconcile every internal file hash, and run the standalone verifier.
5. Delete that exact Pod only after local verification succeeds.

After all five attempts are local, run cohort verification against Run 013,
record run-local observations, re-list Pods and volumes, confirm zero unintended
billable compute, and refresh posted billing. The pre-existing retained volume
is not deleted or modified by closeout.
