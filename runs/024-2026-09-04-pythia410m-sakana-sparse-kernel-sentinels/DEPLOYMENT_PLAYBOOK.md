# Run 024 deployment playbook

## Fixed launch unit

Use one Hopper Pod and execute all six checkpoints sequentially. Prefer an H100
NVL to match Run 023. A cross-SKU H200/H100 result requires a same-device 70M
reference before a model-scale timing claim. The image is
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with 60 GB container disk,
40 GB Pod volume mounted at `/workspace`, no network volume, SSH enabled, and
a six-hour automatic termination backstop.

## Before provisioning

1. List Pods, endpoints, and volumes; preserve the pre-existing unattached
   volume unless separately authorized.
2. Query live Hopper stock and price, current balance, and storage price.
3. Record the selected SKU/tier/data center, hourly burn, six-hour maximum,
   and payload inventory.
4. Confirm that the local input archive and sidecar SHA-256 agree.

## Remote sequence

1. Transfer the sealed archive and verify its SHA-256 remotely.
2. Extract beneath `/workspace/sparsity-spillover`.
3. Start `05_start_worker.sh` once. It detaches immediately and logs under
   `/workspace/run024-output/control/<attempt>/worker.log`.
4. The worker installs pinned environments, checks the clean and patched
   upstream trees, runs the official positive control and seven-shape derived
   preflight, calibrates batch-32 memory, benchmarks, verifies, and packages.
5. Monitor read-only every ten minutes. Warn on a failed stage, a progress file
   older than twenty minutes while alive, nonfinite loss, memory exhaustion,
   or a projected completion beyond the guard.

## Retrieval and teardown

Copy the result tar and its SHA-256 sidecar locally, recompute the local hash,
extract into a new attempt directory, and run `04_verify.py` independently.
Only after local verification succeeds, terminate the Pod and re-list Pods,
endpoints, and volumes. Record elapsed time and rate-times-duration cost; later
billing records supersede that estimate when available.

## Measured planning values

On the matched H100 NVL, the clean final attempt reached the last condition's
post-timing check in 57m53s. Serialization boundaries give these useful future
ETCs: setup plus A0 6m13s; A1-H 3m06s; A4 `kappa=0` 10m15s; A4 `kappa=0.5`
8m59s; A7 `kappa=0` 15m34s; A7 `kappa=0.5` 13m44s. A prepared-Pod six-
condition pass is therefore about `$2.51` at the observed compute and storage
rates. Allow additional time for the 9.1 GiB transfer, environment setup,
positive control, retrieval, and review.

For each future model or condition:

1. Reproduce the official positive control before interpreting a negative
   Pythia timing.
2. Verify checkpoint/config/logical-product hashes and complete source loss.
3. Run the signed round-trip and exact shape matrix before full validation.
4. Calibrate the optional batch with at least 10% memory headroom.
5. Retain A0 as the adverse no-sparsity control and A1-H as the minimal sparse
   endpoint before four-linear A4/A7 conditions.
6. For A7, report Q/V attention as a separate composition; do not attach it to
   the dense-attention full-model speedup.
7. Stop before custom fused-attention work unless every declared layer clears
   break-even.
