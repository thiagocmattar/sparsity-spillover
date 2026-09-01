# Run 018 deployment playbook

This checklist is not launch authorization. It becomes actionable only after
the implementation handoff states the refreshed live billable envelope and the
user explicitly approves launch.

## Immutable transfer inventory

Transfer the committed source bundle and both hash-verified MiniPile cache
files. In addition, transfer these Run 018 files outside the Git bundle to the
same relative paths on every Pod:

- `prelaunch/initialization/pythia70m-seed1234.safetensors`, 281,715,344 bytes,
  SHA-256 `024e01975e1a52ead00340afd7a5c3f0b7c2fa0542d9dd5998e648ec14f73501`;
- `prelaunch/initialization/pythia70m-seed1234-rng.pt`, 14,823 bytes, SHA-256
  `ff839f490cbbbec528181113451802f52c734fb45ae693fc800991bc2be36762`.

Verify byte counts and SHA-256 on the controller before transfer and again on
every Pod. The tracked `metadata.json` must agree. A missing or mismatched file
is a hard stop; remote regeneration is forbidden.

## Provisioning and non-evidence gate

Refresh RunPod balance, all resources, live GPU price, and data-center stock.
Prefer four independent Secure H200 Pods, one condition per GPU, with the
pinned image digest, 30 GB container disk, 25 GB Pod volume at `/workspace`,
SSH key injection, and an independent deletion guard. If H200 is unavailable,
select the fastest available GPU that passes the memory and exact preflight;
report its price and revise the maximum cost before creation. Do not attach or
delete the pre-existing retained network volume.

Secure all four sentinel Pods before setup. Transfer repository and
initialization artifacts to all Pods, start `01_setup_remote.sh` concurrently,
then transfer the 5.97 GB cache in projected slowest-condition order (A7, A4,
A1, A0). Verify clean scoped source, commit/bundle identity, runtime/package
pins, caches, schedule, artifact files, strict parameter hash, and CUDA seed.

On the A7 Pod, run `06_remote_preflight.py` detached with PID, log, JSON, package
freeze, and transfer records under `/workspace/run018-control`. It creates no
attempt and must pass before any scientific worker starts. The exact gate
constructs on CPU, strictly loads the canonical artifact, reproduces parameter
SHA-256 `e8b8…`, moves to CUDA, completes five A0 and A7 boundaries, full
validation, diagnostics, checkpoint serialization, and at least 10% VRAM
headroom. If it fails, retrieve and verify the records, delete all newly created
Pods, and leave Run 018 unchanged.

## Scientific sentinel

After the preflight passes, start one detached worker on each Pod:

```bash
cd /workspace/sparsity-spillover
condition=a7-ol1-kappa-0
control=/workspace/run018-control/$condition
mkdir -p "$control"
setsid env PYTHONPATH=/workspace/sparsity-spillover/src \
  /workspace/run018-venv/bin/python -u \
  runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init/03_train.py \
  --worker "$condition" > "$control/train.log" 2>&1 < /dev/null &
echo $! > "$control/train.pid"
```

Use `a0-gelu`, `a1h-relu`, `a4-ol1-kappa-0`, and `a7-ol1-kappa-0`, one per Pod.
Poll no more often than every five minutes. Every update includes completed
step, current loss, throughput, elapsed time, and refreshed ETC. Declared
warnings are an event older than ten minutes, non-finite metrics,
overflow/skipped boundary, capture mismatch, hash/schedule/runtime drift, VRAM
above 90%, disk risk, or ETC beyond the deletion guard.

After a condition finishes, run `04_verify.py --condition <id>`. For A0 and
A1-H also run `07_teal_posthoc.py --condition <id>` on the checkpoint-bearing
Pod. Retrieve the complete attempt, final recovery checkpoint, diagnostics,
TEAL output, logs, package freeze, identities, and transfer inventory. Verify
all local hashes before deleting that Pod. Confirm zero unintended Pods after
teardown; retain only the pre-existing network volume.

The eight nonzero-kappa A4/A7 workers are a separate wave. Do not create them
without a new explicit decision after the sentinel evidence, ETC, and actual
spend are reported.
