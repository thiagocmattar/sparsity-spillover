# Infrastructure attempt 001: slow setup

Pod `woj2n6vvtncpxv` was created at `2026-08-31T11:37:17.377Z` on
`US-KS-2`. The exact source and cache transfers completed, but the pinned
`torch==2.11.0` installation remained in the same package-install phase for
more than 27 minutes. The child process was still alive at about 5% CPU, so
this was an infrastructure-speed problem rather than a scientific failure.

Before deletion, the controller verified that no `train.pid` existed and that
no scientific attempt directory had been created. The setup log was retrieved
as `setup.log`. The Pod was then deleted, and replacement Pod
`6yo210ydme7k8u` was created at `2026-08-31T12:12:03.654Z` with the unchanged
image, source, cache, runtime, hardware class, disks, and condition assignment.
No checkpoint or scientific artifact exists for this infrastructure attempt.
