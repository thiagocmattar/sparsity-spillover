# H100 Secure pilot, infrastructure retry 002

- Pod `ziisaqda7des80`, `run025-calibration-h100-002`.
- Requested 15:32:47 UTC; accepted $3.49/hour Secure H100.
- Stop deadline 16:18:47 UTC (46 minutes); compute maximum $2.6757.
- Independent local stop guard PID 40452 armed at 15:33:17 UTC.
- Verified physical H100 80 GB HBM3, 81,559 MiB, driver 580.126.09.
- Workload unchanged: original U0 supported FFN control plus P0 primitive
  portability. No Pythia full-model calibration on this Pod.

## Workflow

- [x] Reconcile Community failure (zero Pods), price and remaining budget.
- [x] Allocate Secure H100 and verify guard plus real SSH/GPU access.
- [x] Complete small primitive payload transfer; match archive and file hashes.
- [x] Pin environments; pass H100 P0 primitive gate and sanitizer.
- [x] Execute untouched upstream supported control and retain CSV/provenance.
- [x] Retrieve/hash-verify finalized evidence, terminate, reconcile resources.

## Preserved transfer issue

The first extraction check was issued while the SCP session was still active.
The partial archive failed with Unexpected EOF; the bootstrap file had not yet
arrived, and no setup or scientific worker started. This was an orchestration
mistake, not a kernel result. `start-verified.sh` now requires the complete
expected archive SHA256 before extraction or execution. Wait for SCP exit zero.
The transfer is also unusually slow (~1.7 MB after approximately two minutes).
All time remains charged against the original absolute 46-minute lease.

The default SFTP-mode SCP then ended with a connection reset (no completed
upload). A compressed legacy-SCP retry (`scp -O -C`) completed in nine seconds
at approximately 15:38 UTC. Same exact archive bytes, destination and SSH key;
no relay, credential upload or scientific input change. Re-use this verified
transport on this Pod for small evidence retrieval.

## Result and teardown

Bootstrap ran 15:39:32-15:46:05 UTC and exited zero. P0 passed 48 primitive
cases with zero sanitizer errors; the official backbone benchmark completed
at 1.16544x Torch throughput. See the observation for important workload and
correctness-contract limitations.

All 32 evidence files verified locally. Pod deletion returned HTTP 204;
the subsequent list showed zero Pods at 15:51:38 UTC. Guard PID 40452 was
stopped only afterward. Zero serverless endpoints; existing 100 GB network
volume retained unchanged. Temporary Pod data was removed; unique evidence
is local, and the pinned public upstream model remains reproducibly downloadable.
