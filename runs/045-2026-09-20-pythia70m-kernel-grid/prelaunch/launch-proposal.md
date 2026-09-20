# Launch proposal: Run045

Status: design approved, launch approval pending. No resource created.

- Pod: `run045-70m-kernel-grid-001`, one Secure NVIDIA GeForce RTX5090 (32GB),
  CUDA>=12.8. Live18:41:51UTC quote20September: USD0.99/GPU-hour; stock LOW
  in EU-RO-1, EUR-IS-1 and EUR-NO-1. Recheck immediately before allocation.
- Image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
  Separate pinned Python3.12/Torch2.11.0/CUDA12.8/Transformers5.12.1 environment.
- Storage:20GB container plus80GB Pod volume at `/workspace`; no new network
  volume. Existing100GB shared volume stays untouched. Both temporary storage
  types currently cost USD0.10/GB/month while running. Three hours of100GB is
  about USD0.042 (30-day conversion). Ingress/egress has no provider fee.
  Source: https://docs.runpod.io/pods/pricing, checked20September2026.
- Proposed incremental cap: USD5 total including compute, storage and any
  infrastructure retries; maximum3 cumulative billable GPU-hours. At the quoted
  rate a full3h lease is about USD3.02 including running storage. Stop deadline
  is creation+3h, with an independent guard and20min reserved for recovery.
  Retrieve/verify before deletion; use stop rather than destructive deletion as
  deadline backstop if outputs have not yet been verified locally.
- Expected ETC:60-100min, provisional until exact on-Pod calibration. Evidence:
  Run042 comparable three-graph processes took18.36-31.29s, with9-10s full
  diagnostics on replicate1. Budget30-45min for78 final processes and two
  diagnostic implementations;25-45min for provisioning,7.36GB uncompressed
  input transfer, runtime installation and compilation;5-10min for smoke,
  reduction, return and verification. No training or search time is included.
- Resource fit: historical peak4.06GB Torch allocation. Exact three-backend
  preflight must confirm at least8GiB free GPU memory after capture. The local
  RTX5070Ti Laptop has12227MiB total/10313MiB free at inspection, but cannot
  supply the approved RTX5090 latency comparison. Local disk has824GB free.
- Preflight: three fresh GPU processes (Base, T7/Ph kappa.5, T2/Ph kappa.1),
  8 validation blocks and4inputs x2 timing passes each. These are smoke results,
  not final evidence. Then78 full qualification processes:26checkpoints x3,
  each with native/legacy/opt073 graphs,338blocks and64inputs x7passes.
- Retention: all26 final models are already local. Transfer the verified input
  archive and inventory outward; return every raw timing, validation gate,
  loss, activation/weight/work diagnostic, failure example, profile, runtime
  log, source identity, reduction and file inventory. No weights need to return
  because this inference-only run cannot update them. Verify all result sizes
  and SHA256 before deleting the Pod and confirming absence.
- Monitor every60s: completed/78, current finite validation loss, evaluation
  tokens/s and refreshed ETC. Shorten near completion. Warn on nonfinite output,
  qualification rejection, runtime/source drift, insufficient VRAM/disk, stale
  progress for10min or projected deadline/cost overrun. No concurrent GPU jobs.

The user confirmed the proposed diagnostic inventory is sufficient. Gradient
interaction/OL1 history remains in the original training artifacts; no new
gradient diagnostics or post-hoc clipping can be inferred from this evaluation.

## Local verification completed

- Six focused tests pass: full78-process grid;26endpoint identities and full
  validation provenance; byte-identical frozen arithmetic with HZ-only adapter
  extension; geometric-mean estimand; paired timing completeness/full logits;
  per-block numerical gates and pooled691886-token loss denominator.
- Full bootstrap:242tests passed in10.43s.
- Four real CPU smoke checks: c00(Base), c21(T7/Ph kappa.5), c22(T2/Ph kappa0),
  c25(T2/Ph kappa.1). Native finite full-vocabulary outputs on four tokens,
  both frozen implementations install, topology and h/z thresholds preserved.
  These checks do not execute CUDA kernels or establish performance.
-1,627 retained input/source copies and all26 checkpoint identities verified;
  85 runtime source/config files frozen. All18 root Python files parse.
- The quality-audit parser also accepts retained full338-block Run042 evidence
  in a local parser-only regression check. That check is not a new measurement.

Commands: `python -m pytest -q tests`; `python -m pytest -q
runs/045-2026-09-20-pythia70m-kernel-grid/test_protocol.py`; `07_cpu_smoke.py
--condition <c00|c21|c22|c25>`; `01_prepare.py freeze`; `01_prepare.py bundle`.
