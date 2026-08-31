# Run 015 guarded RunPod playbook

This document is an execution definition, not launch approval.

## Preflight

1. Re-query existing resources, A100 80 GB stock, and live price.
2. Create one time-limited Secure A100 SXM 80 GB Pod with the pinned image only
   after explicit launch approval.
3. Transfer the clean launch commit and hash-locked MiniPile cache; verify the
   commit, runtime, cache, initialization, and schedule identities.
4. Run five exact MB32/GAS32 boundaries at both `kappa=0` and `kappa=0.5`.
5. Require Flash SDPA, finite non-skipped boundaries, OL1 trust ratio at most
   one, at least 10% memory headroom, and exactly the 24 sorted
   `{a,m,h,z}.layer_{0..5}` pressure names at every boundary.
6. Retrieve and hash-verify the preflight packet, then delete the preflight Pod.

Any scientific mismatch stops the launch and requires a new numbered run. An
unchanged infrastructure retry receives a new attempt record inside Run 015.

## Scientific fleet

After a passing preflight, create five independent guarded Secure A100 SXM
80 GB Pods. Assign exactly one condition to each GPU, verify source/runtime/cache
identity on every worker, and launch one detached authoritative process. Confirm
the first finite event—including the 24-tensor capture count/hash—from all five
workers before normal monitoring.

Poll read-only every five minutes unless refreshed ETC is shorter. Report step,
input tokens, task and pressure loss, throughput, ETC, loss scale/overflow,
gradient conflict/projection/trust metrics, pressure capture identity, GPU
memory/utilization, disk, process state, and event age. Warnings are a missing
process, ten-minute stale event, nonfinite value, skipped boundary, wrong
capture identity, trust ratio above one, memory beyond preflight, less than
5 GB disk, incomplete validation, or deadline overrun.

## Retrieval and teardown

For each terminal worker, run standalone verification, inventory every agreed
artifact with bytes and SHA-256, copy it locally, recompute hashes, and only
then delete that exact Pod. After all five attempts are local, run cohort
verification against Run 011, re-list resources, and confirm no unintended
billable compute remains. The pre-existing retained network volume is not
deleted or modified without separate authorization.
