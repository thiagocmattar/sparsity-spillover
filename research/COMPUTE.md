# Compute, ETC, RunPod, and Monitoring

## Decision rule: local or cloud

Measure a representative workload before deciding.

Choose local when the exact model/method/sequence fits with reasonable memory
headroom, the projected ETC is acceptable to the user, and occupying the local
machine is acceptable. Choose RunPod when local memory is insufficient or the
local ETC/impact is unacceptable and the user approves the current cloud price
and maximum cost.

Do not encode a universal hour or VRAM threshold. Present the evidence and let
the user decide.

## Representative calibration

Use the exact architecture, precision, sequence length, microbatch,
accumulation, optimizer, pressure path, and diagnostics proposed for the run.
After warmup, measure several complete optimizer boundaries. Separately measure:

- setup/model construction;
- optimizer-step time and tokens/second;
- one complete full-validation pass;
- checkpoint serialization;
- required post-hoc diagnostics;
- expected transfer size and measured transfer throughput for cloud.

Report sample count and variability, not one convenient step.

### Timer boundary

The representative step timer must include every recurring part of an update:
batch/index construction, host-to-device staging, forward/backward work,
gradient processing, and the optimizer update. A narrower synchronized kernel
or optimizer-boundary timer may also be retained, but it must be labeled and
must not drive ETC by itself. Run 004 demonstrated this failure mode: its
boundary timer omitted batch construction and staging and therefore
underestimated pressure-worker wall time even though the measured CUDA work was
correct.

## ETC model

```text
training_seconds = planned_steps * representative_step_seconds
validation_seconds = planned_validation_passes * full_validation_seconds

local_ETC = setup + training + validation + diagnostics + checkpoint

cloud_ETC = provision + setup/upload + training + validation + diagnostics
            + checkpoint + download_and_verify
```

Count update 1, cadence evaluations, and final evaluation without duplicating a
step that satisfies two rules. Include uncertainty from measured step variation
and provisioning/transfer overhead.

For cloud:

```text
maximum_compute_cost = live_hourly_gpu_price * gpu_count * maximum_hours
maximum_total_cost = maximum_compute_cost + estimated_storage_cost
```

Prices and capacity are time-sensitive. Query them immediately before proposing
the launch; never copy a historical A40 price.

`tools/estimate_etc.py` performs the transparent arithmetic from a calibration
JSON file. It is an estimator, not a launch gate.

## Two approvals

1. **Design approval:** scientific question and exact comparison.
2. **Launch approval:** implemented code, tests, ETC, local/cloud choice, live
   price/cost ceiling, persistent storage, transfer inventory, monitoring, and
   teardown.

Creating a billable resource requires the second approval.

## RunPod route

Training is a long-lived batch job: use a RunPod Pod, not Serverless. At the
start of a RunPod task, load the installed `runpod` router skill. Prefer
structured RunPod MCP operations for current catalog/resource reads and simple
lifecycle calls when connected; use current `runpodctl` for SSH information,
file transfer, and reproducible shell workflows. Tool schemas and `--help` are
authoritative.

The operational connection, launch, transfer, and teardown sequence is in
`RUNPOD.md`.

### Before creation

- Verify authentication without printing the API key.
- List existing Pods and volumes and reconcile resources for this project.
- Query current GPU availability, VRAM, data center, cloud tier, and hourly price.
- Choose the smallest GPU that fits the representative workload with headroom.
- State Pod name, image/digest, GPU/count, data center constraints, storage,
  maximum duration, termination deadline, maximum cost, and cleanup intent.
- Ensure an SSH public key is registered/injected before Pod creation.

Resolve these choices in dependency order: exact workload memory fit, live GPU
availability and acceptable data centers, then persistent-storage placement. A
network volume is data-center-bound; creating it first can unnecessarily reduce
the GPU pool. If placement flexibility dominates persistence, seed one verified
Pod and distribute an immutable hash-checked cache, or use an approved
region-independent object-transfer path.

Never send credentials to chat or commit them.

### Persistence

Keep the checkout, caches, logs, run artifacts, and checkpoints under
`/workspace`. A Pod volume survives stop/restart but is deleted with the Pod. A
network volume survives Pod termination but continues billing and narrows GPU
placement to its data center. Use a network volume only when artifacts must
outlive or be shared beyond one Pod; otherwise transfer and verify before
terminating the Pod.

Pin the image or digest and package versions. Re-query SSH host/port after every
restart. Long jobs must be detached from SSH and log to `/workspace`.

### Launch and monitoring

- Verify the remote Git commit and environment before starting.
- Use one authoritative process and writable checkout.
- Write `manifest.json` and `events.jsonl` continuously.
- Start a detached process whose PID and log path are recorded.
- Poll with bounded sleeps in separate calls. Monitoring is read-only.
- Report active step, tokens, latest finite losses/diagnostics, throughput,
  refreshed ETC, GPU memory/utilization, disk, and stale-event warnings.

Do not infer that a Pod marked `Running` means the training process is healthy.

### Completion and teardown

1. Verify terminal status and required remote files.
2. Build a transfer inventory with byte counts and SHA-256 values.
3. Copy the agreed artifacts locally.
4. Recompute local hashes against the inventory.
5. Only then terminate the Pod.
6. Re-list Pods and volumes; report anything retained and its continuing cost.

Provider billing buckets can lag resource deletion. Record the teardown-time
estimate as provisional, then refresh posted billing during run closeout and
state which value supersedes it. Keep one-time Pod charges separate from any
retained volume's continuing monthly cost.

Use an automatic termination deadline as a backstop, not as the primary cleanup
mechanism.

## Post-hoc decision before launch

Ask whether to collect now or retain a checkpoint for:

- exact-zero and near-zero activation counters at named thresholds;
- activation RMS/L2 and distribution statistics by site/layer;
- weight norms/histograms by role/layer;
- `R_block`/`R_model` actual-operand counters;
- clipping frontier sites and cutoff grid;
- gradient conflict and OL1 trust-budget metrics;
- any examples/predictions needed for qualitative analysis.

Gradient interaction is training-time-only. The other diagnostics require a
checkpoint plus exact validation/cache and implementation identity if deferred.
