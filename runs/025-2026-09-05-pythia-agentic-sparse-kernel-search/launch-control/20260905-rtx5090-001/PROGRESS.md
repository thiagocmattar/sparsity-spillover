# Approved calibration lease: RTX 5090, attempt 001

Launch approval: user approved the $5 sequential RTX-5090/H100 pilot after
the `afad780` launch review. The $40 total study envelope is unchanged.

- Pod: `ivt0noruc0l142`, `run025-calibration-rtx5090-001`.
- Requested 2026-09-05 15:06:20 UTC; provider-created 15:06:37 UTC.
- Accepted GPU rate: $0.69/hour, COMMUNITY; one RTX 5090, 32,607 MiB.
- Driver 575.57.08; Python 3.12.3. 40 GB container / 80 GB Pod volume.
- Direct SSH/SCP verified; existing network volume untouched.
- Independent stop guard PID 23340 armed at 15:06:42 UTC.
- Absolute stop deadline: 17:06:20 UTC. Worker/transfer windows must end
  earlier; lease clock includes setup, failures, waiting and transfers.

## Workflow

- [x] Refresh account/resources/price and pass 242 bootstrap tests.
- [x] Allocate the approved GPU with public SSH and the pinned image digest.
- [x] Arm and verify independent billing guard; verify physical GPU and volume.
- [x] Transfer primitive bundle and match SHA256
  `a9bed29d27a13652e8901c37946d5f09eaea6ebc5849ea359d12e9f2763f6d9c`.
- [x] Install the pinned Pythia environment and verify all 32 primitive files.
- [x] Pass CUDA compilation / 48-case primitive gate under compute-sanitizer.
- [x] Finish 5.83 GB checkpoint transfer, verify full archive and 88 files.
- [x] Execute calibration gate; stop on numerical failure (1/8 complete).
- [x] Retrieve/hash-verify all finalized evidence, terminate and confirm cleanup.
- [x] Reconcile pilot spend; execute H100 U0/portability if funded and ready.
- [x] Present measured pilot ETC/cost and explicitly unmeasured larger-size scope.

## Infrastructure issue, preserved

The image's CUDA bin directory was not on the SSH shell PATH. The first
bootstrap exported it. Setup completed, but the first primitive worker
(`rtx5090-memcheck-001`) then failed before compilation because Ninja, although
installed in the venv, was not on PATH. Its trace and zero completed primitive
cases are retained. A compute-sanitizer “0 errors” line is NOT a pass when
the Python process exited with an exception.

`memcheck-002.sh` additionally prepends the venv bin directory. This is an
invocation-environment retry, with no kernel, checkpoint, evaluator, numerical
tolerance or scientific configuration changes. Preserve the initial sources.
The same PATH must be passed to the full worker and H100 invocations.

## Primitive result

`rtx5090-memcheck-002` completed at 15:12:55 UTC: 48/48 fixed primitive
gates passed; compute-sanitizer reported zero errors. Compilation took
28.745 seconds and primitive-only execution took 30.461 seconds overall.
Full-model timing and validation are still pending; this is not a speedup result.

## Full-model gate result

Full archive SHA256 matched; all 88 file hashes verified. The detached worker
started at 15:23:35 UTC and stopped at 15:23:50 UTC on its fixed numerical gate.
A0-14M passed all 338 validation blocks (native loss 5.209999416;
P0 loss 5.209966001). A1-H-14M failed the elementwise logit gate on block 1
(zero-indexed), with relative L2 0.001758925 and maximum absolute error 0.375.
The remaining six checkpoints were not executed. No tolerance was relaxed.

An isolated two-block diagnostic reproduced the failure: adapter-dense was
bit-identical, P0 exceeded tolerance at 26 logits on block 1, and the FP32
fused-linear oracle passed both blocks. A split-bias oracle failed more
substantially. This suggests accumulation/rounding sensitivity; it does not
prove a complete cause or justify accepting the failed implementation.
Root scientific sources remain unchanged. RTX calibration ends here; retain
the failure and do not begin the optimization search from an unqualified P0.

57 evidence files verified locally; archive identity is recorded in the
observation. Pod deletion returned HTTP 204 and a subsequent list contained
zero Pods at 15:29:02 UTC. Only then was guard PID 23340 stopped locally.
The copied input cache/checkpoints on this temporary Pod were deleted with it;
their verified originals remain local. No unique evidence was discarded.
