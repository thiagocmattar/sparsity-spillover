# Run040 launch packet — awaiting launch approval

Design and both checkpoints are approved. The user additionally authorized
kernel optimization. The user explicitly approved this USD6 launch envelope. Provisioning follows
the scoped source commit; no additional confirmation is required within it.

## Implemented and checked

- Nine controls x two checkpoints x three fresh processes =54 diagnostic
  processes, each with full338-block qualification and64x7 paired timing.
  Eighteen bounded smokes precede this matrix. Native/full traces are collected
  separately from reported latency. All inputs and frozen sources are hashed.
- Up to12 immutable optimization candidates developed on16 fixed training
  blocks, followed by one frozen candidate's six final processes. No new
  training, pressure/threshold changes, other kappa points or relaxed bounds.
- Eight focused CPU checks and242 bootstrap checks pass. The primitive CUDA
  checks and72 training-input control checks are implemented but require the
  remote GPU. Their success, followed by all18 smokes, gates scientific timing.
- The machine here has an RTX5070Ti Laptop with12GB, not the RTX5090 used by
  the paper benchmarks. The cloud plan preserves GPU class and pinned runtime.
  Previous two-model peak:3.44GiB. The new three-model case is provisioned on
  a32GB card; actual peak is measured in smoke, with at least8GB headroom required.

## Proposed resource and live price

One Secure Cloud RTX5090,32GB, one active benchmark job. RunPod's20 September
catalog returned USD0.99/GPU-hour; capacity was low, with a CUDA13.0-capable
host available. This is the driver's advertised capability: the experiment
retains CUDA12.8 libraries/compiler and PyTorch2.11.0+cu128. Refresh placement
before creation, without changing GPU, runtime or price ceiling.

- Name: `run040-70m-kernel-overhead-001`.
- Image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`,
  the existing verified Run035 image; install the retained exact package lock.
-20GB container disk;80GB Pod volume mounted at `/workspace`; SSH22/tcp.
  All code, inputs, logs, results and environments live under `/workspace/run040`.
- Existing100GB network volume in EUR-IS-1 is preserved and not attached.
  No new independent volume or endpoint is proposed. Account inventory
  currently has no Pods or endpoints.
- Maximum cumulative GPU time:4 hours, including any infrastructure retry.
  Maximum incremental task cost:USD6. At the current rate, four GPU hours are
  USD3.96;100GB running storage adds about USD0.056 over four hours using the
  published USD0.10/GB/month rate. See [RunPod pricing](https://www.runpod.io/pricing).
  The unused balance of the cap is contingency, not permission to exceed4 GPU hours.
  Any stopped Pod disk used briefly for recovery is also included in the cap.

Expected completion:1–2.5 hours from provisioning, approximately USD1–2.6
including storage. This is a preflight estimate, not a fresh GPU calibration:
the prior66 comparable processes took23.5 minutes total, but this run adds a
third reference graph, profiling, more diagnostics and a bounded optimization
search. Setup/transfer and compilation can dominate. Refresh ETC after smoke
and each candidate; stop optimization early if there is no useful progress
or the remaining time is needed for final evaluation/recovery.

## Transfer, retention and monitoring

The input package is about601MB: two original final checkpoints, exact
validation and training-development caches, frozen implementation/vendor
sources, run scripts and provenance. The final receipt identifies the current
bundle; earlier bundles are retained as prelaunch records. No credentials are
in these bundles. Actual upload throughput will refresh the transfer ETC.

Retain raw host/CUDA timings and pair identities, all per-block correctness
and loss records, every candidate/failure and source hash, GPU traces with
operator shapes/correlations, full-validation exact/near-zero and RMS/L2
statistics, layer weight norms, row/tile occupancy and logical/MMA/scalar
counters with native counters marked unmeasured. Preserve thresholds/sites,
checkpoint/cache identities and both final checkpoints. No clipping or
gradient-interaction experiment is added; the latter needs training records.
Confirm any additional later-use diagnostic requirements with launch approval.

Run detached with persistent logs. Monitor read-only every60 seconds, reporting
phase/progress, measured validation loss when available, throughput, memory,
ETC and projected spend. Check sooner for failure, nonfinite values, failed
numerical bounds, unexpected GPU activity, less than8GB free GPU memory,
disk under10GB, or no setup/compile progress for10 minutes / no worker progress
for3 minutes outside compilation. Shorten the final wait to projected completion.

Arm independent on-Pod and hidden local stop guards at the four-hour deadline,
recording exact Pod identity and UTC deadline. Reserve20 minutes for recovery;
the stop guard preserves the Pod volume rather than deleting unverified data.
Copy output archives and verify their archive and per-file hashes locally,
including partial/failed runs, before deleting the Pod. Then confirm no task
Pod/endpoints/new independent volumes remain and record final estimated billing.
