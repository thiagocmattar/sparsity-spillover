# Run 022 deployment playbook

This is the proposed launch procedure. It is not launch authorization. Refresh balance, GPU availability, and prices immediately before provisioning.

## Provisioning envelope

- One on-demand Hopper Pod, preferably community H100 SXM 80 GB. H100 NVL and then H200 are engineering fallbacks if no H100 SXM is available; do not pool timings from different SKUs.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- 60 GB container disk and 20 GB Pod volume mounted at `/workspace`; no new network volume.
- Maximum wall-clock guard: 3 hours from Pod creation. The setup/build, upstream control, and Pythia stages are all inside that guard.
- Monitor every 10 minutes. Check sooner only for a declared warning: no new log/progress for 20 minutes while GPU utilization is under 5%, disk exhaustion, nonzero worker exit, CUDA OOM, source/hash mismatch, or a projected finish inside 10 minutes.

The pre-launch estimate is 60–100 minutes, dominated by two environment installs, CUDA extension compilation, and the unmodified SparseLM0.5B positive control. This is an engineering estimate, not a prior calibration of this exact workload. Re-estimate after setup, after the upstream control, and during the Pythia validation. The 3-hour guard is a spending/safety ceiling, not a claim that completion is guaranteed by that time; if the refreshed ETC crosses it, preserve diagnostics and stop rather than silently expanding the bill.

The live catalog audit on 2026-09-04 quoted community on-demand rates of $2.69/hour for H100 SXM, $2.59/hour for H100 NVL, and $3.59/hour for H200, all at low availability. For the 60–100 minute estimate, compute alone is therefore $2.69–$4.48 on H100 SXM; its three-hour compute ceiling is $8.07. The 60 GB container disk plus 20 GB Pod volume add about $0.011/hour while running at the documented $0.10/GB/month rate (about $0.033 at the ceiling). Refresh every quote immediately before provisioning.

## Before provisioning

1. Run `python runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/01_static_preflight.py`.
2. Run the focused Run-022 tests and the full bootstrap suite.
3. Run `python runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/00_prepare_inputs.py`.
4. Inspect `prelaunch/input-inventory.json`. Confirm that it contains only run/source code, the public MiniPile validation cache, and the A0 final checkpoint; it must exclude `training_state.pt`, optimizer state, train data, credentials, and caches.
5. Record the input archive SHA-256, live RunPod balance, exact quoted hourly price, selected GPU ID/SKU/cloud, maximum cost, and deadline.

## On the Pod

1. Transfer `tmp/run022-inputs.tar` to `/workspace` and verify its recorded SHA-256 before extraction.
2. Extract into `/workspace/sparsity-spillover` and re-run the static preflight.
3. Start one disconnect-safe worker:

   ```bash
   bash /workspace/sparsity-spillover/runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/05_start_worker.sh
   ```

4. Record the attempt ID and PID printed by the worker launcher.
5. After each bounded 10-minute wait, take exactly one read-only snapshot:

   ```bash
   python /workspace/sparsity-spillover/runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/06_monitor.py
   ```

Expected stages are `setup`, `upstream_positive_control`, `raw_ell_preflight`, `pythia_a0_benchmark` (with finer JSON progress), `verification`, `packaging`, and `complete`.

## Retrieval and teardown

1. Require `exit-code.txt == 0`, `status.txt == complete`, and `verification.json: passed == true`.
2. Download the result tar and its `.sha256` sidecar to `artifacts/attempts/<attempt-id>/incoming/`.
3. Verify the archive SHA-256 locally before extraction. Re-run `04_verify.py` locally against the extracted attempt.
4. Reconcile CSV/JSON, environment freezes, logs, checkpoint/cache hashes, GPU identity, timing-block counts, and coverage.
5. Only after those checks, terminate the Pod. Confirm that there are no unintended running Pods, endpoint workers, or newly retained volumes. Report the final billed duration/cost.
6. Add result observations only after review. Run 022 produces no publication figure by itself unless the user approves a result presentation.

## Failure playbooks

- **No H100 inventory:** try H100 NVL and H200 only if their device reports compute capability 9.0 and the live billable envelope is approved. Treat the selected exact SKU as fixed for this attempt.
- **Dependency/setup failure:** keep the run inputs unchanged; repair only the attempt setup, record the error, and use a new attempt directory.
- **Upstream positive-control failure:** stop. Do not interpret Pythia raw-ELL timings.
- **ELL compile/correctness/overflow failure:** stop. Preserve compiler output and preflight JSON; do not promote A1-H.
- **A0 ELL slowdown:** expected scientific negative control, not a launch failure.
- **Artifact transfer failure:** keep the Pod running, retry retrieval, and do not terminate until local hashes verify or the hard safety deadline requires an explicit decision.
