# RunPod Pod Procedure

This procedure covers GPU pretraining and long diagnostics. It supplements
`COMPUTE.md`; it does not authorize cloud spend.

RunPod tools and flags evolve. At the start of every RunPod task, load the
installed `runpod` router skill and confirm current MCP schemas or
`runpodctl --help`. The command shapes below were checked during this handoff,
but live tool output is authoritative.

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

## 6. Transfer by an agreed inventory

Before teardown, finish all approved post-hoc diagnostics. Build
`transfer_inventory.json` with relative paths, byte sizes, and SHA-256 values.
Then transfer by `scp`/`rsync`, object storage, a network volume, or the current
encrypted `runpodctl send <path>` / `runpodctl receive <code>` flow. Large
checkpoint sets usually favor resumable transfer or object storage.

Recompute hashes locally and compare every agreed file. A successful copy
command without inventory verification is insufficient.

## 7. Terminate and reconcile billing

After local verification, terminate the exact Pod (currently
`runpodctl pod remove <pod-id>`; confirm with `runpodctl help pod remove`). Then
re-list Pods and network volumes. Delete only explicitly approved disposable
volumes; network volumes survive Pod termination and keep billing.

The completion report includes Pod ID, runtime, observed/maximum cost, transfer
inventory result, termination result, and every resource intentionally retained.
