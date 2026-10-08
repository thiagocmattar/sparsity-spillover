# Run068 launch packet

Status: separate launch approved; the owned two-H100 Pod started at
2026-10-08 22:33:38 UTC for $7.98/hour. Setup and production preflights are running.
The user confirmed the two-model design, retained inventory and $60 envelope.
Scientific fields are in [config.yaml](config.yaml); scope and limitations are
in [README.md](README.md). The explicit user approval is recorded in
[launch-approval-001.json](prelaunch/launch-approval-001.json); the actual provider
identity and deadline are in [lease-training-001.json](prelaunch/lease-training-001.json).

## Implemented and checked

Independent h/z8x16 kappa0 and head16x16 kappa.1/.5 configurations, exact native
gate/backward/pressure ports, original initialization/data-order replay, full
validation and diagnostic coverage,24 model checkpoints plus6 recovery states,
terminal count/hash verification, detached training, scoped deadline guards,
and archive retrieval are implemented. The adapted head consumer and paired
dense/no-skip/sparse endpoint benchmark are under `inference/`.

242 bootstrap+44 focused tests pass. Both real14M CPU smokes check exact random
parameters, one OL1 boundary, causal-prefix logits and checkpoint restoration.
Final WSL tag003 qualifies both development models across all338 blocks:2028 exact
full-logit comparisons and768 exact standalone head comparisons. Four reduced
FP16 training boundaries and63 CUDA component checks pass. See
[checks](prelaunch/checks.json). No production training or RTX5090 result exists.

## Resource fit and estimated completion

The workstation GPU has12GB. The requested MB32/GAS32 workload does not fit:
the matched Run066 production preflight reserved about61.47GB on an80GB H100
(72.3% of physical capacity). Reduced local MB1/GAS2 checks are development
evidence and cannot establish the production fit or ETC.

Run066's Ph+head/kappa0 preflight median was3.498312 seconds per update; a single
712-update wave is41.5 minutes before setup/diagnostics/transfer. The new head
tile size and thresholds are unmeasured production changes. Estimate75-120
minutes for the training Pod including setup/cache rebuild, both preflights,
one concurrent training wave, validation/checkpoints and verified transfer;
15-30 minutes for a separate5090 setup/qualification/measurement/transfer stage.
Overall estimate90-150 minutes, with uncertainty. Actual production preflights
replace the extrapolation. Preserve flexible backstops rather than stopping at
the nominal estimate.

## Proposed cloud definition and current envelope

[Current catalog/capacity and account inventory](prelaunch/catalog-and-inventory.json)
were queried on8 October2026. Secure2-H100 SXM80GB and2-H200 placements have LOW
stock with CUDA12.8 available. The latest CUDA12.8-specific5090 probe reports
OUT; an earlier unfiltered probe reported LOW. Its catalog rate below is a
planning quote, not a currently allocatable offer. Recheck after training and
retrieve/delete the training Pod before waiting for benchmark capacity. Prices:

| Stage | Primary placement | Current hourly compute | Flexible maximum |
|---|---|---:|---:|
|Two concurrent independent trainings|One Pod,2 H100 SXM80GB|$7.98 total ($3.99/GPU)|4 hours|
|Training fallback if H100 unavailable|One Pod,2 H200 SXM|$10.58 total ($5.29/GPU)|4 hours|
|Endpoint comparison after training retrieval|One Pod,1 RTX5090|$1.19|2 hours|

H100 is the primary because the same full recipe has measured fast throughput
and memory headroom there. H200 is an availability fallback with more memory;
its workload speed is not assumed. Both conditions use the same training GPU
type/runtime/global batch. No scientific batch, pressure or gate change is
authorized by a provisioning failure. No simultaneous training/benchmark Pods.

Proposed names: `run068-aligned-vocabulary-training-001` and
`run068-aligned-vocabulary-head-001`. Secure on-demand only. Pin image digest
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`;
Python3.12, Torch2.11.0+cu128, Transformers5.12.1, NumPy2.5.0. Training:40GB
container disk+50GB `/workspace` Pod volume. Benchmark:40GB container disk+25GB
Pod volume. No network volume is created/attached. Use registered SSH keys and
the verified runpodctl2.14.0 binary/current MCP control plane.

**Maximum aggregate incremental spend: $60**, including setup, both stages,
storage and infrastructure retries. At current rates the full primary compute
backstops cost$31.92+$2.38=$34.30; H200 fallback costs$42.32+$2.38=$44.70.
Running disks add about$0.07 at those full limits (730-hour month). Refresh the
actual offer before creation and remain within the aggregate ceiling. A failed
placement can be retried only with fresh attempt records and available budget.
Do not overwrite executed evidence or silently change scientific inputs.

[RunPod's pricing documentation](https://docs.runpod.io/pods/pricing) specifies
per-second compute/storage billing and no ingress/egress fees. Running disks
cost$0.10/GB/month; stopped volume disks cost$0.20/GB/month. If a guard stops
before verified retrieval, retained50GB training storage costs about$0.0137/hour
and25GB benchmark storage about$0.00685/hour until deletion. Retrieve promptly
within the total envelope. Foreign retained resources are outside this budget
and will remain untouched.

## Launch, production barrier and monitoring

After approval, commit identity/source bundle, original random replay and cache
identities are verified remotely. Build the pinned MiniPile/tokenizer cache and
check both token-file hashes and the exact order hash before training. Detached
processes and logs live on `/workspace`; a dropped terminal does not stop work.

Run six complete MB32/GAS32 preflight boundaries per condition concurrently,
then full338-block validation, complete activation/logical diagnostics and an
exact B1/T2048 production-FP16 checkpoint roundtrip. Discard preflight updates.
Only after both pass, start fresh from the approved original initialization.
The all-condition barrier rejects source/cache/order/initialization drift,
skipped/overflow boundaries, nonfinite values, wrong pressure capture, failed
coverage/corner checks, changed checkpoint logits/loss, reserved memory>90% or
median full update>6.8 seconds. Investigate failures without loosening criteria.

Training monitoring interval:5 minutes; use shorter projected-completion waits
when ETC is below that interval. Report each model's step, loss, throughput,
refreshed ETC, GPU memory/utilization and disk. Declared warnings: process exit,
nonfinite/overflow/skipped boundary, events stale>10 minutes, inadequate memory
or disk, source/coverage/certificate mismatch, and a cost/deadline overrun.
Benchmark monitoring:60 seconds; report completed process/block, reference
loss, validation throughput, qualification status and refreshed ETC. Laptop
development timings are never substituted for accepted5090 timing.

Record provider start times, actual rates, Pod IDs, absolute deadlines and guard
PIDs. A credential-free remote guard stops the training process group at four
hours; a hidden workstation provider guard stops the owned Pod while preserving
unverified storage. Use the same scoped provider guard with a two-hour deadline
for the5090. Automatic guards are backstops; finish and tear down promptly.

## Transfer inventory and teardown

Upload a Git bundle of the committed source and56,279,344 bytes of original
random parameter replay. Datasets are rebuilt and hash-verified; no credential
enters source, artifacts or logs. Source bundle receipt is generated after the
implementation commit. All source/cache/init/schedule hashes are retained.

The prepared source bundle deploys implementation commit
`f69ac644166de774c6a1ecc9abf1646105aa69a9`:212,032,868 bytes, SHA256
`57d5d362b269c2c5b8170a02ee44ba66b25234eb7c0f91fa0154f533f3ea08d7`.
Its [receipt](prelaunch/source-001.receipt.json) records the actual source identity.
The bundle and random replay remain ignored local transfer payloads. The later
packet-sealing commit adds this receipt and does not change runtime arithmetic.

Retrieve both configs/manifests/statuses/events, all validation losses,
head/block pre/post zero/near-zero/moment/energy/occupancy/corner statistics at
steps0/32/128/712, weight/final-LayerNorm records, logical counts, training
gradient/OL1 data, runtime/preflight/deadline/pipeline logs,24 scheduled model
checkpoints and6 optimizer/scaler/RNG states. Expect roughly2.1-2.3GB including
preflight checkpoints, extrapolated from Run066; archive actual bytes/hashes.

Verify every archive member locally before deleting the training Pod. Then
seal and upload the two verified terminal endpoints plus pinned full-validation
bytes to the5090; verify the bundle before any benchmark. Retrieve six complete
processes, all raw paired timing rows, head/full numerical checks, counters,
input indices, setups/memory/environment/source identities, failures and seals.
Verify every returned hash before deleting the5090 Pod. Re-list account Pods
and volumes, confirm no owned unintended billable resources remain, and record
teardown-time cost estimates separately from posted billing.

The foreign EXITED Run052 Pod `qa00sm05lmrs7w` and100GB network volume
`9luykg5yc3` were discovered read-only. Neither will be started, reused or deleted.
