# Launch proposal: Run047

Status: design confirmed; implementation and local checks complete; launch
approval pending. No resource has been created.

## Workload and implementation

- Three unchanged final checkpoints: Base c00, T2/Ph kappa=.1 c25, and the later
  Run046 T2/Ph kappa=.5 c26. Native/original-port/opt073 are paired within every
  process, with the Run045 protocol and byte-identical kernel/measurement code.
- Three preflight processes: one per checkpoint, eight validation blocks and
  four timing inputs x2 passes. Then nine final processes: three per checkpoint,
  full338-block numerical qualification and64inputs x7passes per backend.
- Publication latency: geometric mean of1344 host timings per implementation
  and checkpoint. Record this new GPU session separately from Run045; repeated
  Base and kappa=.1 establish descriptive session comparisons, not silent
  rescaling or pooling of old/new timings.
- Scripts01/04/09 prepare inputs/runtime/cache;02/03/05 benchmark and control;
  06 audits/reduces;07 CPU-smokes;08 packages. `prelaunch/` supplies native SCP,
  read-only monitoring, verified retrieval and isolated deadline-stop helpers.

## Checks and resource fit

- Seven focused tests passed in0.10s; all242 bootstrap tests passed in8.21s.
  They check the exact matrix, content/cache identity, frozen arithmetic,
  paired timing coverage, geometric means, every validation block and the
  pooled prediction-token denominator, and incomplete-cohort rejection.
- Three real CPU checkpoint smokes passed, including the missing kappa=.5:
  finite full-vocabulary outputs, correct h/z placement and thresholds, and
  both frozen implementations install. These are not CUDA execution tests.
- 1467 retained source/input copies and three checkpoint identities verify.
  All24 Python files parse; shell and PowerShell guard syntax pass. Pinned
  runpodctl2.14.0 authentication works. Exact command receipts are in
  `local-checks.json`; GPU qualification remains a preflight/final requirement.
- Prior representative three-backend peak Torch allocation is4.23GB. Require
  at least8GiB free GPU memory after capture. Local RTX5070Ti Laptop has12227MiB
  total and10500MiB free, but cannot provide the requested RTX5090 comparison.
  Local storage has749.2GiB free, sufficient for all retained inputs/results.

## Live execution definition and budget

- Pod name: `run047-70m-t2-kappa05-001`; one Secure NVIDIA GeForce RTX5090,
  32GB, CUDA>=12.8. The live quote in `live-quote.json` is USD0.99/GPU-hour,
  stock LOW in EU-CZ-1, EU-RO-1, EUR-IS-1 and EUR-NO-1. Prefer EU-RO-1;
  recheck the quote/availability immediately before creating the Pod.
- Image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
  Pinned Python3.12/Torch2.11.0/CUDA12.8/Transformers5.12.1/NumPy2.5.0 environment.
- Storage:30GB container plus40GB Pod volume at /workspace; no network volume.
  Runtime/extensions use local container disk; inputs, immutable source originals,
  logs, progress and result artifacts persist in /workspace. Existing shared
  volume `9luykg5yc3` remains untouched. Live audit found no active Pods.
- Maximum90 cumulative billable GPU minutes and USD2 total including temporary
  storage and infrastructure retries. At the current quote,90min GPU costs
  USD1.485;70GB running storage adds approximately USD0.0146, using a30-day month.
  Both container and running volume storage cost USD0.10/GB/month; transfer has
  no provider ingress/egress fee. A stopped40GB Pod volume is USD0.20/GB/month
  until recovery/deletion. Source: https://docs.runpod.io/pods/pricing,
  checked20September2026. No new continuing storage is planned.
- Deadline: creation time+90min, reduced by any earlier attempt's billable time.
  A workstation provider-stop guard and on-host process guard enforce the
  deadline. Keep20min for collection/recovery. Stop preserves outputs if they
  have not yet been verified locally; delete only after verified recovery.
- Expected ETC:45--75min including provisioning, upload, pinned setup, compilation,
  preflight, measurements, download and verification. Run045's78 final processes
  had median68.66s; nine at that median take10.3min. Setup/compile/transfer dominate
  uncertainty. Refine the estimate using the exact on-Pod preflight.

## Transfer, monitoring and teardown

Upload only the three existing final model weights/configs, full validation
cache, frozen source/runtime requirements and benchmark scripts: roughly0.9GB
uncompressed; `input-bundle.json` records final archive bytes and SHA256. The
three models and their original full optimizer/recovery histories stay local.
No checkpoint is modified, so no new weight transfer back is required.

Return all raw timings, full numerical gates/losses, exact/near-zero activation
counts, RMS/L2 and weight statistics, logical/occupancy/MMA/scalar counters,
profiles, failures, runtime/source identities, logs, reduction and inventories.
The user already confirmed this diagnostic inventory is sufficient; no clipping
or new training-gradient measurements are added. Verify archive and every member's
byte count/SHA256 and rerun the reducer locally before deleting the run-owned Pod.
Confirm no run-owned billable resource remains; preserve unrelated storage.

Monitor every five minutes using a real sleep followed by one read-only snapshot.
Report completed/9, current validation loss, evaluation input tokens/s, GPU/disk
headroom and refreshed ETC. Shorten the interval near predicted completion.
Warn on nonfinite or failed qualification, source/runtime drift, less than8GiB
free VRAM after capture, less than5GB free storage, ten-minute stale progress,
or a projected deadline/cost overrun. No concurrent GPU jobs or automatic retuning.
