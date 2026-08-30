# Infrastructure retry 001

The approved cohort process stopped on 2026-08-29 during optimizer boundary
145 of the `kappa=0.01` condition with a CUDA memory-allocation error. The
`kappa=0` condition had already completed all training, diagnostics, checkpoint
retention, and artifact publication. The failed attempt remains immutable.

This retry preserves the launch-approved config, initialization, schedule,
training code, token budget, gate, and OL1 definition. It skips the already
completed `kappa=0` condition and restarts each condition without a completed
attempt from its declared random initialization. New attempts use the next
append-only sequence because attempt 002 is retained as the failed record.

`retry_remaining.py` changes only attempt allocation and terminal selection:
the locked `training.run_condition` executes each pending scientific condition,
and the locked terminal verifier selects the unique completed attempt whose
manifest contains that condition identity. Failed attempts are never selected
as evidence.

At retry decision time, the GPU reported 9,469 MiB free. The first completed
condition took 768.0 seconds with median throughput 105,900 tokens/s. Four
remaining full conditions therefore had an ETC of about 3,072 seconds
(51m12s); combined active GPU time including the failed partial attempt was
estimated at 67m06s, within the approved 70-minute compute envelope.

The retry controller passed its locked-state check and launched as detached
process 7868. It found exactly one completed condition, four pending
conditions, and exact matches for the launch config and run-code identities.

The first two detached-helper starts (processes 7868 and 12792) terminated
before controller startup because Windows failed to load Python's `_ctypes`
DLL. Both starts left the scientific attempt set unchanged. Direct import and
controller checks continued to pass. A hidden PowerShell `Start-Process` start
initially exposed the known case-duplicate `Path`/`PATH` environment issue; the
retry was then launched after normalizing that process-local environment as
process 20212. It created append-only attempt 003 and entered training normally.

Process 20212 completed replacement `kappa=0.01`, `kappa=0.05`, and
`kappa=0.1` attempts. It then stopped with a second CUDA allocation failure at
boundary 356 of `kappa=0.5`; attempt 006 retains that failed partial record.
The last failed-boundary throughput was 58,742 tokens/s, more than 25% below
calibration, while the GPU returned to 9,347 MiB free after exit. Four unique
completed scientific conditions remain valid, but the cohort is not terminally
complete or verified.

Active process time through both failures was 62m23s. Repeating the final
condition from its required initialization was estimated to require 12m48s,
for about 75m11s combined active compute. That exceeds the approved 70-minute
envelope, so no second retry was launched without an explicit extension.

The user explicitly approved extending the combined active-compute envelope to
80 minutes. This authorizes one unchanged restart of only the incomplete
`kappa=0.5` condition and its terminal verification.

The locked-state check found the four expected unique completed conditions and
only `kappa=0.5` pending, with exact launch config and run-code identities. The
approved final retry launched as hidden process 42452 and created append-only
attempt 007.

Attempt 007 stopped with another CUDA allocation failure at boundary 33 of
`kappa=0.5`. It remained within the OL1 trust budget; its last task loss was
8.0433 and throughput was 89,602 tokens/s. The attempt set is still append-only,
and the four completed conditions are unchanged. Cumulative active attempt time
is 3,780.8 seconds (63m00.8s), leaving about 16m59s inside the approved
80-minute envelope, but the approval authorized one final restart and that
restart has now been consumed. No further process was started.
