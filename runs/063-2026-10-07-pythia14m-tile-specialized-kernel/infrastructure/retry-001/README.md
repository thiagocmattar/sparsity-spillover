# Transport retry 001

The user approved launch on 7 October 2026 and retained the two-hour/$3
backstop. Initial EU-RO-1 provisioning failed for unavailable capacity; an
equivalent Secure RTX5090 was allocated in EUR-NO-1 at $.99/hour, host CUDA 12.8.
The pinned application, scientific bundle and benchmark scripts are unchanged.

RunPod's readiness responses intermittently report initializing and omit the
SSH route, even while the assigned server accepts authenticated SSH. The exact
owned Pod/name, previously authenticated route and trusted host key are checked
by [transport.py](transport.py). The GPU UUID was confirmed directly. There has
been no Pod restart. The original workstation and remote deadline guards remained
armed during execution and were disarmed or removed at verified closeout.
No second Pod was created.

The original direct SFTP upload to `/workspace` made only 2,424,832 bytes of
149,872,640 bytes after several minutes; no scientific pipeline had started.
A 1 MiB local-disk SFTP probe took 4.671 seconds. The original local transfer
processes were stopped after verifying their exact command lines. Their partial
remote input was superseded by the same sealed archive via streamed SSH ingress
to `/opt`, followed by a persistent copy and independent SHA256 checks at both
destinations. No scientific input, numerical criterion or timing protocol is
changed. The retry starts the original setup/smoke/scientific scripts detached,
only after transfer verification. Its receipt is `prelaunch/upload-002.json`.

This retry directory contains transport infrastructure only. Scientific attempt
tag001 remains available because no scientific process preceded this retry.
