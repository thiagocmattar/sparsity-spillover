# RunPod Pod Procedure

This procedure covers GPU pretraining and long diagnostics. It supplements
`COMPUTE.md`; it does not authorize cloud spend.

RunPod tools and flags evolve. At the start of every RunPod task, load the
installed `runpod` router skill and confirm current MCP schemas or
`runpodctl --help`. The command shapes below were checked during this handoff,
but live tool output is authoritative.

## Run 004 deployment evidence

Run 004 is the repository's first complete multi-Pod deployment record. Its
full chronology and measured commands are in
`../runs/004-2026-08-29-pythia14m-full-pass-l1n/RUNPOD_STATUS_2026-08-29.md`
and `DEPLOYMENT_PLAYBOOK.md`. Reusable lessons are:

- `RUNNING`, visible `nvidia-smi`, CUDA initialization, flash-attention
  availability, and production memory fit are separate checks. The exact
  MB32/GAS32 workload failed on a healthy Secure RTX 4090 and passed on A100
  80 GB with about 56.95 GiB peak reserved memory.
- Choose exact GPU fit and live placement before creating a network volume.
  RunPod volumes constrain attached Pods to one data center and Secure Cloud;
  they do not synchronize across regions.
- A pinned Hugging Face rebuild can be faster and safer than a poor workstation
  upload when it reproduces the declared cache hash. Build once, then distribute
  the immutable cache and verify it on every worker; do not retokenize
  independently on each Pod.
- Keep the training cache on fast local container storage during random block
  access even when a durable master lives on a network volume.
- Measure complete update elapsed time. A timer beginning after batch assembly
  and host-to-device staging understates ETC.
- Prefer resumable transfer for checkpoint trees. Accept artifacts only after
  byte-count and SHA-256 inventory reconciliation.
- Delete each worker only after its artifacts pass locally, then perform an
  account-level Pod/volume audit and a later posted-billing refresh.

These are deployment facts, not authorization to reuse Run 004's A100 fleet,
region, storage, historical price, or scientific configuration.

## Run 009 multi-Pod deployment evidence

Run 009 is the first complete reusable-preflight and distributed-condition
record. Its run-local chronology is in
`../runs/009-2026-08-30-pythia14m-full-pass-ol1/README.md`,
`../runs/009-2026-08-30-pythia14m-full-pass-ol1/DEPLOYMENT_PLAYBOOK.md`, and
`../runs/009-2026-08-30-pythia14m-full-pass-ol1/prelaunch/launch-plan.json`.
Future agents should carry forward these lessons:

- A successful preflight Pod can become one scientific worker without another
  source, environment, or cache transfer when every reusable byte lives under
  its Pod volume's `/workspace`. Stop it only for the bounded approval wait,
  re-query SSH details after restart, recheck identities, and give the restarted
  scientific worker a fresh teardown deadline. Pod-volume persistence ends when
  the Pod is deleted.
- Keep placement broad when the experiment does not require a data center. Run
  009 left the pre-existing EUR-IS-1 network volume untouched, staged verified
  caches on per-Pod storage, and placed workers where the approved GPUs were
  actually available. This avoided making one volume's region a fleet-wide
  capacity constraint.
- Parallel workers may each own one condition while the final verifier owns the
  cohort. Run 009 therefore used an existing per-attempt scientific verifier as
  each teardown gate, then ran the four-condition Run 004 comparison only after
  all attempts were co-located locally. Design both scopes before launch; do not
  discover at teardown that the only verifier requires absent workers.
- A setup failure before cache/scientific-attempt creation is infrastructure,
  not a new scientific condition. Run 009 retained and hashed two stalled pip
  logs and one empty slow-filesystem log, terminated those Pods, and retried the
  unchanged source/config/GPU/price contract. Every replacement received its
  own absolute 2.5-hour deadline.
- Repeated HTTPS `CLOSE-WAIT` with no installer child, or a `venv`/`ensurepip`
  operation that remains far slower than peer Pods, can be a bad host or storage
  placement rather than a package-resolution problem. After a bounded repeated
  no-progress diagnosis, move the unchanged workload to another eligible Pod or
  data center instead of retrying indefinitely on the same placement.
- The exact OL1 workload used about 56.97 GiB reserved and sustained roughly
  360k--370k input tokens/s on the selected A100 80 GB workers. These numbers
  validate that run only; use a new production-shaped preflight for a changed
  boundary, model, batch decomposition, precision, or diagnostic set.
- For every worker, record the remote transfer-inventory byte count, create an
  archive, record its remote byte count and SHA-256, copy it, match the local
  archive, extract it, and run the per-attempt verifier before deletion. Run 009
  deleted each Pod independently as soon as its local evidence passed, rather
  than retaining an idle fleet for the global verifier.
- Billing snapshots changed after all Pods were already gone. Record the query
  timestamp and latest returned bucket end, keep the value provisional while a
  worker's final interval is absent or still changing, and distinguish that
  posted Pod spend from the continuing monthly cost of any retained volume.

These are operational precedents, not standing approval for a future parallel
fleet, retry policy, GPU type, price, region, or cost ceiling.

## 1. Authenticate and inventory without exposing secrets

For an interactive human setup, `runpodctl doctor` can configure the API key and
SSH key. Agents/scripts should consume `RUNPOD_API_KEY` from the environment;
never print, paste, log, or commit it.

Useful read-only checks are:

```text
runpodctl user
runpodctl pod list
runpodctl network-volume list
runpodctl gpu list
runpodctl datacenter list
runpodctl ssh list-keys
```

Prefer the connected RunPod MCP equivalents for structured inventory, catalog,
and lifecycle operations. Use the CLI for SSH details and terminal workflows.
Reconcile existing project resources before creating anything.

If no SSH public key is registered, the human may run `runpodctl doctor`, or a
scripted workflow may generate an Ed25519 key locally and register only its
public half with `runpodctl ssh add-key --key-file <public-key>`.

## 2. Define the exact approved resource

The launch packet names:

- Pod name and project/run number;
- pinned image or official template and package lock;
- GPU type/count, VRAM evidence, cloud tier, and data center constraints;
- container and Pod-volume sizes;
- optional network-volume ID and continuing storage cost;
- ports (normally SSH only for batch training);
- absolute termination deadline, maximum duration, and maximum cost.

For a multi-Pod run, also map each condition to one worker/GPU type, state which
preflight workspace may be reused, and define a separate absolute deadline for
every original and replacement Pod. The total envelope includes setup failures
and retries, not only workers that reach scientific training.

Use the smallest live-available GPU that fits the production-shaped calibration
with headroom. Do not use a historical catalog price.

## 3. Create, then verify reachability

Inspect the current create flags first:

```text
runpodctl pod create --help
```

A current CLI shape is:

```text
runpodctl pod create --name <name> --image <pinned-image> \
  --gpu-id <gpu-type-id> --ports "22/tcp" \
  --terminate-after <ISO-8601-time> --wait
```

Add only approved volume/environment flags. Do not put secrets in tracked
commands or config files. `--terminate-after` deletes the Pod; `--stop-after`
only stops compute and can leave storage billing.

Record the returned Pod ID even if creation or `--wait` times out. Check it with
`runpodctl pod get <pod-id>` and obtain fresh connection details with
`runpodctl ssh info <pod-id>`. After a restart, re-query the host/port until an
actual SSH probe succeeds; cached details may be stale.

## 4. Establish the remote experiment

Keep mutable work and artifacts below `/workspace`; root-filesystem changes are
ephemeral. Over SSH:

1. clone/fetch the approved clean Git commit into `/workspace`;
2. verify the commit and disclose any dirty state;
3. create the environment with pinned versions and record Python, Torch,
   Transformers, CUDA, GPU, and precision identities;
4. prepare or transfer caches and verify their hashes;
5. run the exact smoke/calibration again if the hardware differs;
6. create the running manifest before the scientific process starts.

Time-box setup progress using comparable milestones: source identity,
environment creation, package lock, cache bytes/documents, and cache hash. If a
Pod is materially stalled, inspect its process tree, sockets, disk latency, and
log timestamp before deciding whether it is an infrastructure retry. Preserve
and hash the failure log before removal. A retry rechecks the approved source,
commit, image/runtime, GPU, and cache identities from the beginning.

Environment values supplied to a container's PID 1 are not automatically
available in an SSH shell. Set needed non-secret values explicitly in the
detached launch command; provide secrets through a protected remote environment,
never a tracked file.

## 5. Launch detached and monitor read-only

Use `setsid`, `tmux`, or an equivalent mechanism so training is independent of
SSH. Record PID, command, attempt path, and log path under `/workspace`.

Poll in bounded intervals rather than holding a busy connection. On each report,
check the authoritative process plus manifest/events, step and input tokens,
finite loss/diagnostics, throughput and refreshed ETC, GPU memory/utilization,
disk space, and timestamp of the latest event. Monitoring must not mutate the
run or start a duplicate process.

When ETC is shorter than the regular interval, wait until that completion window
and perform one check. Keep connection waits and sleeps separate from Pod status
queries so a quiet worker is not repeatedly polled. Treat process presence and a
terminal manifest as separate facts; final validation/diagnostics may complete
after the last training boundary.

## 6. Transfer by an agreed inventory

Before teardown, finish all approved post-hoc diagnostics. Build
`transfer_inventory.json` with relative paths, byte sizes, and SHA-256 values.
Then transfer by `scp`/`rsync`, object storage, a network volume, or the current
encrypted `runpodctl send <path>` / `runpodctl receive <code>` flow. Large
checkpoint sets usually favor resumable transfer or object storage.

Recompute hashes locally and compare every agreed file. A successful copy
command without inventory verification is insufficient.

For distributed conditions, the run must expose a worker-local verification
path that does not require other workers' directories. Record one retrieval
receipt per worker with Pod ID, attempt ID, inventory/archive byte counts,
archive SHA-256, local verification result, and principal scalar result. Run the
cohort-level verifier only after every independently accepted attempt is local.

## 7. Terminate and reconcile billing

After local verification, terminate the exact Pod (currently
`runpodctl pod remove <pod-id>`; confirm with `runpodctl help pod remove`). Then
re-list Pods and network volumes. Delete only explicitly approved disposable
volumes; network volumes survive Pod termination and keep billing.

The completion report includes Pod ID, runtime, observed/maximum cost, transfer
inventory result, termination result, and every resource intentionally retained.

Query billing with an explicit scope and time window. Record the query time,
provider bucket end, unique Pod count, GPU amount, Pod-disk amount, and total.
If the final worker interval is absent or the same closed bucket is still
changing, label the total provisional and refresh later. Zero active Pods proves
compute teardown; it does not prove that billing ingestion has settled.
