# Controller interruption, budget overrun, and bounded recovery

All times are UTC. This is an infrastructure incident, not a scientific retry.
No training inputs or scientific attempts were changed or rerun.

The original authorization was five A100s at USD1.59/hour for at most three
hours each: USD23.85 compute plus USD0.20 estimated storage. The selected
deadline guards ran on the Windows controller. This was insufficient: the
controller stopped executing between the 16 September 21:14 monitoring
snapshot and approximately 17 September 03:13. The local guards did not
enforce their 23:09--23:10 deadlines during that interruption. Their logs show
stops only at approximately 03:13--03:14. The agent's choice of local guards
caused the approved ceiling to be unenforced; this is an actual overrun.

The user was informed and explicitly approved up to USD2 more for at most
15 minutes of artifact recovery. Four Pods restarted at 03:18:34--03:18:39.
The kappa=0.5 Pod could not restart because its host had no GPU capacity;
the explicit CPU-only resume also failed for insufficient host memory.

An initial on-Pod CLI stop guard could not authenticate. It was supplemented
before the deadline by `13_recovery_guard.py`, which validated each exact
Pod ID/name through authenticated REST and then held the API credential only
in memory after unlinking its temporary input. The four independent on-Pod
guards were armed at 03:19:53--03:20:03 for 03:32:28. A live account read after
that deadline confirmed all five Pods EXITED. Thus this recovery window used
about 14 minutes on four A100s, approximately USD1.47 compute, within its new
cap. Posted billing is delayed; this estimate is not a final invoice.

All four accessible jobs had completed training, final evaluation, diagnostics,
the original remote verifier, and archive packaging with exit code zero. Small
metadata archives were downloaded and SHA-256 checked independently. Full
archive transfers stalled: SFTP, native RunPod transfer, and an SCP probe did
not finish within the window. The network-backed /workspace mounts had already
caused latency during setup. The next prepared retry stages only missing files
with bulk reads onto /opt before SFTP; this performance remedy remains untested
remotely and requires a newly approved recovery window.

The interrupted archives are retained locally. `17_salvage_downloads.py`
extracts only complete members matching the original per-condition byte/SHA-256
inventory. Missing identical initial-weight files are copied only on exact
inventory hash equality, with source/destination provenance. Two missing YAML
snapshots are reproduced by the frozen serializer only after their exact
original byte count and inventory hash match. No scientific values are inferred.
`prelaunch/salvage-receipt.json` records these operations and all missing files.

The complete kappa=0.01 artifact/checkpoint inventory now passes the original
standalone local verifier. Kappa=0 has its final checkpoint and all later
recovery states locally, but two early model snapshots remain missing. Kappa
0.05 and 0.1 still lack final checkpoints. Metadata for all four passes the
original event, coverage, and diagnostic checks. Kappa=0.5 remains unretrieved.
The raw setup/training/transfer process logs have not all been downloaded, so
no Pod has yet been deleted, including kappa=0.01. All five persistent 25 GB
volumes remain intentionally retained with compute stopped.

`prelaunch/billing-recovery-snapshot.json` contains the 03:36:44 API snapshot:
USD51.3405578868 posted across these five Pods. Its records still lag both
observed stop times, so the final charge will be higher. Do not describe the
original run as within budget or this snapshot as a settled invoice. No
endpoint was created; the pre-existing 100 GB network volume `9luykg5yc3`
is unchanged.
