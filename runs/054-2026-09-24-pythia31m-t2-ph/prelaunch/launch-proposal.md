# Run 054 launch proposal (not yet approved)

## Completed preparation

- 11 focused tests passed: exact gate/gradient placement at all five thresholds,
  six identical initial parameter hashes, matched prior recipes/flat data order,
  validation coverage, real Base/OL1 CPU optimizer boundaries, checkpoint reload,
  analytic ceiling and non-hardcoded batch/LR validation.
- Full bootstrap suite: 242 tests passed.
- Local production-shaped GPU calibration used RTX 5070 Ti Laptop (12,227 MiB),
  WSL `SparsityGPU`, Torch 2.11.0+cu128 and Transformers 5.12.1. Six updates each
  for Base and kappa=0.1; the first update was excluded from timing. Complete
  validation, checkpoint/recovery save, and kappa=0.1 full activation/logical
  diagnostics passed, with no skipped updates.
- Base median 16.7322 s/update; T2/Ph kappa=0.1 median 21.0762 s/update. Peaks
  reserved were 8.1915 and 8.2355 GB respectively of 12.8205 GB device memory.
  Approximate six-condition training ETC is 24.15 hours, planning 24-26 hours
  with diagnostics/checkpoints. Other thresholds may differ; only two conditions
  were calibrated locally. This uses the laptop GPU continuously.
- Ten CUDA component cases passed: zero, short, mixed, dense, and last-z-tile,
  each with skip disabled/enabled. Arithmetic and instruction/scalar counters
  passed. Full-model Base and kappa=0.1 six-update calibration checkpoints each
  qualified on all 338 blocks, with identical pooled eager/native-graph/kernel
  losses. Timing smoke used 64 inputs x seven passes, one process, laptop GPU;
  it is not the final RTX 5090 result or a trained endpoint.
  A third full-model check on the two-update kappa=0.5 lifecycle checkpoint also
  passed all 338 blocks and untimed work-counter checks after adding explicit
  runtime/source identity enforcement.
- `14_lifecycle_smoke.py` passed the actual two-update Base/kappa=0.5 training,
  checkpoint publication, final reload, four full validation/diagnostic passes,
  and `03_verify.py` under a separate non-evidence directory. The final receipt
  is `lifecycle-local-002/result.json`. The first attempt exited before any
  optimizer update because an isolated source copy lacked Git metadata; the
  deployment identity now comes from its hash-verified package, while this smoke
  explicitly identifies itself as non-evidence staged source.

## Proposed execution and spend

One new Secure RTX 5090 32 GB Pod, `run054-training-latency-001`, broad placement
with CUDA >=12.8. The live quote is **$0.99/GPU-hour**; latest inventory reports
**no Secure capacity available**. Recheck capacity and price after approval;
do not substitute a GPU for the final latency measurement. Quote/audit details
are in `live-resource-quote.json`.

Pinned image and runtime: see `config.yaml` and `00_setup_remote.sh`. Allocate
40 GB container disk and 80 GB persistent Pod volume at `/workspace`; expose
SSH only; no network volume or new endpoint. Keep source, logs, checkpoints,
environment and transfer archives on `/workspace`. Stage the immutable train
cache to container storage for training only if its read throughput requires it,
without changing bytes or the persistent master.

Maximum **16 aggregate GPU-hours, $20 new-resource spend**, including setup,
preflights, permitted infrastructure retries and retrieval. At the quote,
16 GPU-hours cost $15.84; 120 GB running storage for 16 hours is about $0.27.
Reserve an additional 24 hours of stopped-volume storage if retrieval encounters
a problem (80 GB is about $0.022/hour). Existing resources are outside this new
spend envelope. RunPod's published storage rates are $0.10/GB-month while
running and $0.20/GB-month for stopped Pod volumes; no transfer bandwidth fee:
https://docs.runpod.io/pods/pricing . Retained-storage billing continues until
termination, so a retrieval problem must be surfaced promptly.

The desktop RTX 5090 ETC is **not yet measured**. First run six six-update
production-shaped preflights; do not begin scientific training unless measured
training ETC x1.3 +45 minutes latency +1 hour retrieval fits the remaining
absolute deadline. A preflight fit failure ends scientific launch within this
envelope. Do not silently reduce tokens, validation, diagnostics or checkpoints.
All six scientific conditions execute sequentially from the same initialization.

Record the absolute UTC deadline from the first creation, including any retries.
Arm `13_local_stop_guard.ps1` on the workstation with only that newly created
Pod's ID/name and the deadline, using pinned runpodctl v2.14.0. Its CLI help and
read-only authentication were checked; no account credential goes to the Pod.
Launch `08_pipeline.py --deadline <UTC> --tag <unique-tag>` with `nohup`/`setsid`,
redirect output to persistent storage, and retain its PID. The pipeline enforces
the one-hour retrieval reserve and kills timed-out subprocess groups. The
workstation guard independently stops the Pod at the deadline if still present.
Keep the workstation awake/networked while the guard is needed.

## Transfer and teardown

Inputs: approximately 152 MB source/initialization plus 5.970 GB byte-identical
train/validation caches and metadata. Input archive and per-file SHA-256/size
receipts are verified remotely before work. No credentials are packaged.

Outputs: approximately 13.2 GB scientific checkpoint/recovery payload plus six
preflight recovery snapshots (about 2.2 GB), all training events/manifests,
config/source/cache identities, coverage, canonical activation/weight/logical
statistics, raw paired latency samples, full numerical qualification, runtime
moments/occupancy/work counts, environment freeze, setup/compile/pipeline logs,
resource receipts, and final transfer inventories. Download/checkpoint copies
can begin after each condition completes; never copy an actively written file.

Seal terminal evidence with `10_seal.py`; download the archive and receipt over
SSH using a resumable transfer where available. Verify archive SHA-256/size and
all contained files, then run `11_retrieve.py` and the scientific verifier
locally. Copy `/workspace/run054-control` environment/setup/guard logs separately
with hashes. Only after verification, terminate the new Pod and audit account
resources. If deadline arrives first, **stop** compute and retain its volume
until retrieval is verified; do not auto-delete uncopied artifacts.

Existing resources observed: stopped Run052 Pod `qa00sm05lmrs7w` and 100 GB
shared volume `9luykg5yc3` in EUR-IS-1. Neither is owned by this task and neither
will be altered.

## Monitoring

Check every five minutes during training, every minute during setup/latency,
or at the projected completion window if sooner. Each update reports condition,
step, latest task loss, measured tokens/s, and refreshed remaining ETC including
unstarted conditions, diagnostics and retrieval. Use `04_monitor.py` plus the
pipeline's measured forecast; avoid repeated unchanged checks between intervals.

Warnings: nonfinite loss/gradients/diagnostics, any skipped optimizer update,
less than 10% GPU headroom, less than 10 GB free storage, no event progress for
ten minutes, numerical qualification failure, or forecast exceeding the deadline.
Pause further conditions on a scientific failure and preserve evidence. No
automatic scientific recipe changes or expansion of the billable envelope.

The user explicitly retained the approved post-hoc package; no extra measurements
were requested. The subsequent instruction to use available RunPod GPUs for the
fastest execution authorized the parallel launch and superseded this sequential
proposal. See `parallel-fleet-001.json` and the run README for the executed
placement, $200 cap, 19:01 UTC deadline, infrastructure retry and parallel
retrieval entry points.
