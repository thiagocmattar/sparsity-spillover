# Recovery continuation: four conditions fully retrieved

On 17 September the user prioritized complete scientific recovery, then
explicitly stated that more funds had been added and instructed recovery to
continue until success. This superseded the preceding USD2/15-minute recovery
limit for necessary recovery; it did not authorize a new scientific run.

Four existing Pods resumed at approximately 03:42 UTC, at the observed
USD1.59/GPU-hour rate. On-Pod REST shutdown guards were armed for 03:56:08.
`19_retrieve_missing.py` first fetched remaining process logs, then copied
only missing scientific files in 8 MiB reads from /workspace onto /opt and
packed them there. Kappa=0 and 0.01 completed and passed the original local
verifier at 03:43:59 and 03:43:44 respectively. Their Pods were deleted only
after those checks and successful log retrieval.

Staging completed rapidly on kappa=0.05 and 0.1, but the SFTP downloads still
stalled. A read-only 1 MiB SSH command-stream probe succeeded. The stalled
local transfer clients were stopped without altering source data. Following
the expanded user authorization, independent one-hour on-Pod guards replaced
the short guards before `21_stream_staged_archive.py` downloaded those same
staged archives through an SSH command stream. Transfers reached approximately
6--9 MB/s. Archive SHA-256 values and the complete original per-file inventory
were verified, followed by the unchanged original standalone verifier.

Kappa=0.1 passed at 03:50:25 and kappa=0.05 at 03:51:23. Their Pods were then
deleted. All four model and optimizer/RNG checkpoints, complete events,
metrics, diagnostic files, and original process logs are local. Scientific
file and checkpoint integrity is covered by the original transfer inventory;
auxiliary process logs were retrieved over authenticated SSH but do not have
an independent original per-file hash manifest.

Receipts are in `prelaunch/cloud/<pod-id>/retrieval-receipt.json`. The current
`prelaunch/recovered-metadata-audit.json` embeds full standalone verification
results for all four conditions. Earlier incomplete-recovery states remain
recorded in commit `1ddbe13` and infrastructure record 004. No scientific input,
checkpoint content, or training attempt was changed or rerun.

The only remaining Pod is kappa=0.5, `8o1uyxgh6vb6ym`, with its 25 GB volume
retained. GPU resumes are rejected for no host GPU capacity; explicit CPU-only
resumes are rejected for insufficient host memory. Runpod documents an
automatic migration option in its console, but the available in-app browser
is logged out; user sign-in was requested while API capacity retries continue.
The live REST v2 schema and runpodctl v2.14.0 expose no migration action.

At 04:00 UTC, `22_retry_kappa05.py --prepare` configured a one-hour guard in
the stopped Pod's startup command. It uses the documented RUNPOD_POD_ID and
checks its own name before calling the stop API. Its credential is supplied
through a dedicated Pod environment variable, unset before the ordinary
container service starts, and never written to local evidence. Deletion after
verified recovery removes the Pod configuration. This boot-time guard does
not depend on the local controller successfully reaching SSH first.

`23_wait_for_kappa05.ps1` runs as a hidden local process, writes persistent
logs, and uses Start-Sleep for five-minute capacity intervals. It tries CPU-only
access first, then the already-priced GPU resume. Once access is available it
checks the boot guard, stages missing inventory files, streams and verifies
them, and deletes this exact original Pod only after the standalone verifier
passes. A console migration changes the Pod ID and requires reconciling that
new identity before this original-ID worker can continue.

Official documentation: https://docs.runpod.io/pods/troubleshooting/pod-migration
and https://docs.runpod.io/pods/troubleshooting/zero-gpus . No messages or
research data were submitted to Runpod support or its feedback tools.

At 04:16:46 UTC, one additional CPU-only resume used the public GraphQL
`PodResumeInput.syncMachine=true` option. Its semantics are not explained
in that specification; this was a bounded alternate start request, not an
assertion that stale accounting caused the failure. The same insufficient
memory response was returned, followed by the same insufficient GPU response.
See `prelaunch/kappa05-machine-sync-attempt.json`. Normal five-minute retries
continue without this optional flag. The retained control-plane system log
is saved under the remaining Pod's evidence directory; it confirms the
03:13:34 UTC stop but contains no training-result output.
