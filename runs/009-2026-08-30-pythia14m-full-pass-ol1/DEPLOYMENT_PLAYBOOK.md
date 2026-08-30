# Run 009 staged RunPod playbook

This is a tracked conversational sequence, not launch authorization. Stop at
every approval gate.

## Iteration 1 - implementation (completed)

1. Lock the four approved OL1 conditions and Run 004 comparators.
2. Implement and test the FP16/OL1 boundary, artifacts, verification, smoke,
   monitoring, and remote setup.
3. Run focused and repository-wide tests.
4. Query RunPod resources and prices read-only; prepare the exact launch packet.
5. Stop for explicit preflight approval.

## Iteration 2 - billable preflight (completed)

1. Re-query live capacity and price across all data centers for either approved
   Secure A100 80 GB type. If no qualifying offer exists, wait five minutes and
   repeat until one appears or the user interrupts. Create exactly one Pod, and
   never exceed the approved hourly price or absolute termination deadline.
2. Verify SSH, pinned image/runtime, source identity, and both cache hashes.
   Keep every reusable file under `/workspace`; use the existing network volume
   only if the winning placement happens to be `EUR-IS-1`, otherwise stage one
   verified cache copy onto the Pod volume.
3. Run `05_remote_preflight.py`: five exact lambda-1 MB32/GAS32 boundaries on
   the target GPU, with initialization, schedule, FP16, OL1, trust, and memory
   checks.
4. Copy the preflight log, environment lock, JSON, and SHA-256 inventory locally.
5. Stop the successful Pod to release GPU billing while retaining `/workspace`.
   Retain it for at most 24 hours while awaiting confirmation, then terminate it
   if approval has not arrived.
6. Report measured ETC and refreshed four-Pod price/cost, identify the matched
   condition that can reuse this Pod without another transfer, and stop for
   scientific launch approval.

## Iteration 3 - four-Pod scientific launch (completed)

1. Reconcile existing Pods/volumes and re-query price/capacity.
2. Restart the retained preflight Pod, polling its data center every five minutes
   until its approved GPU is available or the user interrupts. Its scientific
   assignment is lambda `1.0` after PCIe preflight or lambda `0.05` after SXM
   preflight.
3. Create the other three guarded one-GPU Pods with the approved GPU assignments.
4. Reuse the retained worker's verified `/workspace`; stage the pinned checkout,
   environment, and cache concurrently on only the other three workers. Verify
   every runtime and hash before launch.
5. Start one detached `02_train.py --worker <condition>` process per Pod under
   `/workspace`, recording PID and log path.
6. Verify the first event from all four workers, then begin read-only monitoring.

## Monitoring

Poll every five minutes. Each update reports worker step/input tokens, task and
pressure losses, throughput, memory, loss scale/overflow, OL1 projection/trust,
and refreshed ETC. Warn on an absent process, ten-minute stale event, nonfinite
value, any skipped boundary, trust ratio over `1.0`, memory above the preflight
envelope, less than 5 GB free disk, incomplete validation, or projected overrun
of the termination deadline.

If refreshed ETC is shorter than the normal interval, wait until the projected
completion window. Monitoring never mutates or restarts a worker.

## Retrieval and teardown

For each terminal worker:

1. Verify manifest status, all 712 train events, complete validation/diagnostics,
   checkpoints, and remote transfer inventory.
2. Copy the agreed attempt tree locally and recompute every size/SHA-256.
3. Terminate that exact Pod only after local verification.
4. After all four attempts arrive, run `03_verify.py` against Run 004 evidence.
5. Re-list Pods and volumes. Confirm zero unintended Pods and report the retained
   pre-existing volume and its continuing cost.

Completed on 2026-08-30: all four attempts passed local and global verification,
all six Run 009 Pods were terminated, zero active Pods remained, and only the
pre-existing volume `9luykg5yc3` was retained. Posted Pod spend was
provisionally `$12.2669929022` at `14:15Z`; the final lambda `0.1` billing bucket
had not yet settled.
