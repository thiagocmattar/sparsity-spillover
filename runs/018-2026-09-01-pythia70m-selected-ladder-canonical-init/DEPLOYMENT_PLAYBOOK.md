# Run 018 deployment playbook

The timing-preflight section is authorized. Scientific sections are not launch
authorization and become actionable only after the measured ETC, refreshed live
billable envelope, and explicit scientific launch approval are recorded.

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

## Approved timing-only gate

Refresh RunPod balance, all resources, live H200 price, and candidate-data-center
stock. Create one independent Secure H200 Pod for the A7 preflight, using the
pinned image digest, 30 GB container disk, 25 GB Pod volume at `/workspace`, SSH
key injection, and an independent external 1.5-hour deletion guard. Do not attach
or modify the retained network volume.

Transfer the committed repository and all immutable inputs, run
`01_setup_remote.sh`, and then run `06_remote_preflight.py` detached with its PID,
log, JSON, package freeze, and transfer records under
`/workspace/run018-control`. It creates no attempt. Retrieve and hash-verify all
records, then delete the Pod whether the preflight passes or fails.

A pass must strictly load parameter SHA-256 `e8b8...`, complete five exact A0
and five exact A7 boundaries, full validation, activation/logical diagnostics,
checkpoint serialization, and retain at least 10% VRAM headroom. Use its
per-condition projection to calculate the scientific guard as 1.5 times the
projected workload plus measured provision/setup/transfer time. Do not start a
scientific worker in the timing Pod.

## Future scientific provisioning

After separate approval, refresh RunPod balance, all resources, live GPU price,
and data-center stock. Prefer four independent Secure H200 Pods, one condition per GPU, with the
pinned image digest, 30 GB container disk, 25 GB Pod volume at `/workspace`,
SSH key injection, and the measured deletion guard. If H200 is unavailable,
select the fastest available GPU that passes the memory and exact preflight;
report its price and revise the maximum cost before creation. Do not attach or
delete the pre-existing retained network volume.

Secure all four sentinel Pods before setup. Transfer repository and
initialization artifacts to all Pods, start `01_setup_remote.sh` concurrently,
then transfer the 5.97 GB cache in projected slowest-condition order (A7, A4,
A1, A0). Verify clean scoped source, commit/bundle identity, runtime/package
pins, caches, schedule, artifact files, strict parameter hash, and CUDA seed.

The retrieved timing preflight must still match the committed code/config/cache
identities before scientific workers start.

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
