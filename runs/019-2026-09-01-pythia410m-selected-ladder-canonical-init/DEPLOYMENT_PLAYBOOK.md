# Run 019 deployment playbook

This file describes the prepared workflow; it is not launch authorization.

## 1. Calibration launch gate (completed 2026-09-02)

Immediately before proposing billable calibration:

1. list all Pods and network volumes and reconcile them with this project;
2. refresh RTX A6000, A40, L40S, A100 PCIe 80 GB, and A100 SXM 80 GB stock and
   prices;
3. choose at least two simultaneously available candidates, normally the
   low-cost RTX A6000 and the high-throughput A100 SXM;
4. measure or conservatively bound provisioning, the 7.070 GiB input upload,
   package setup, and output retrieval;
5. propose one Pod per candidate, exact per-Pod deletion deadlines, maximum
   compute/storage cost, and cleanup behavior for explicit approval.

The retained network volume was not attached because it constrained data-center
choice. Calibration Pods used 35 GB container disk and a 60 GB Pod volume
mounted at `/workspace`; both disappeared with the Pod after verified
retrieval. The final audit found no Pods or serverless workers. See
`prelaunch/calibration/RESULTS.md` and
`launch-control/calibration-20260901/README.md`.

## 2. One source and input identity

After the implementation commit, build one complete Git bundle from that exact
commit and record its byte count and SHA-256 in the launch record. A plain
`git archive` is insufficient because `01_setup_remote.sh` requires a real
`.git` checkout. Transfer the same bundle to every worker and explicitly check
out its verified ref. Transfer the seven files in
`prelaunch/input-manifest.json` and verify their sizes and SHA-256 before setup.
Do not regenerate initialization on a Pod.

Extract the source at `/workspace/sparsity-spillover`, place data and
initialization files at their declared repository-relative paths, and run:

```bash
bash runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/01_setup_remote.sh
```

Setup installs the pinned runtime and executes the two-load canonical
initialization verifier on CPU. Any hash, parameter, RNG, runtime, or cache
failure stops the candidate before a CUDA calibration boundary.

## 3. Detached exact calibration

For each candidate, use the current live catalog values and a unique output:

```bash
setsid /workspace/run019-venv/bin/python \
  runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/05_calibrate.py \
  --gpu-type-id 'NVIDIA A40' \
  --cloud-type SECURE \
  --hourly-price-usd '<live-price>' \
  --output /workspace/run019-control/calibration-a40-secure.json \
  > /workspace/run019-control/calibration-a40-secure.log 2>&1 </dev/null &
```

The controller records the PID, command, Pod/catalog identity, source bundle
hash, input-manifest hash, setup start/finish, calibration start/finish, and
external deletion deadline. An exit code of zero plus a JSON `status` of
`passed` is required; a running Pod is not success.

## 4. Monitoring

Wait ten minutes between read-only checks unless a declared warning occurs or
the projected completion is sooner. Each update tails the log and reports the
active condition/boundary, latest finite loss, tokens/second, peak reserved
VRAM, elapsed time, and refreshed completion estimate. Warning conditions are:

- missing progress for 20 minutes;
- non-finite loss, overflow, or skipped optimizer update;
- CUDA OOM or reserved VRAM above the 90% headroom limit;
- disk below 10 GB free;
- process exit without a terminal calibration JSON.

## 5. Retrieval and teardown

For each candidate, retrieve the calibration JSON, log, setup log, initialization
verification, package freeze, execution context, and controller timestamps.
Build a remote file inventory, verify every local byte count and SHA-256, then
delete the Pod. Re-list Pods and volumes and confirm that no unintended billable
resource remains. Record both the teardown estimate and later posted Pod-ID
billing refresh.

## 6. Human GPU selection and scientific launch

`09_compare_calibrations.py` has verified the A40 and A100 SXM records. Its
cost/makespan Pareto set is in `prelaunch/calibration/comparison.json`; human
options are in `prelaunch/calibration/RESULTS.md`. The human selects one GPU SKU
and concurrency for all twelve final conditions. Then update the null GPU
fields and both deletion guards in `config.yaml`, rerun focused and full tests,
commit the calibrated launch definition, refresh prices/balance, and request a
new explicit scientific-launch approval.

The intended scientific execution is one condition per GPU with up to twelve
identical-SKU Pods in parallel. If stock requires waves, preserve the same SKU
and exact run identity. Do not silently change microbatch, accumulation,
checkpointing, precision, topology, threshold, pressure, data, or diagnostics.
