# Run 016 deployment playbook

This is an execution checklist, not launch authorization. Refresh every live
price, stock, resource, and connection detail before use.

## Phase 1: one non-evidence preflight Pod

After explicit preflight approval, create exactly one Pod:

- name: `sparsity-run016-preflight-a40`;
- GPU: one `NVIDIA A40`, Secure Cloud;
- preferred data center: `EU-SE-1`, then `CA-MTL-1` if explicitly accepted;
- pinned image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`;
- 30 GB container disk, 25 GB volume disk at `/workspace`;
- SSH public key injected at creation; no credentials in code or logs;
- absolute termination backstop: 1.5 hours;
- current guarded maximum: `$0.672`, subject to the immediate price refresh.

Before creation, list Pods and volumes and re-query the A40's per-data-center
availability and price. Do not attach retained volume `9luykg5yc3`: it is bound
to `EUR-IS-1`, outside the current A40 locations.

Transfer the committed repository state and both token-cache files. On the Pod,
verify the Git commit, clean scoped checkout, metadata byte counts, and SHA-256
before setup. Run `00_setup_remote.sh`, then record the package freeze.

Start the preflight detached from SSH, with its PID and log under persistent
`/workspace` storage. The operative command is:

```bash
cd /workspace/sparsity-spillover
setsid env PYTHONPATH=/workspace/sparsity-spillover/src \
  /workspace/run016-venv/bin/python -u \
  runs/016-2026-08-31-pythia70m-selected-ladder/05_remote_preflight.py \
  > /workspace/run016-control/preflight.log 2>&1 \
  < /dev/null &
echo $! > /workspace/run016-control/preflight.pid
```

Poll no more often than every five minutes. Report current probe condition and
boundary, loss, throughput, reserved memory, elapsed time, and refreshed ETC.
Check sooner only for the declared warning conditions in the README.

Retrieve and hash-verify:

- `prelaunch/remote-preflight.json`;
- `/workspace/run016-control/preflight.log`;
- `/workspace/run016-control/runpod-pip-freeze.txt`;
- transfer timing/byte records.

Then terminate the Pod and re-list Pods/volumes. Do not start scientific
training. Commit the measured preflight/ETC record and request science launch
approval.

## Phase 2: scientific sentinel, only after a new approval

Create four independent one-GPU Pods for:

1. `a0-gelu`;
2. `a1h-relu`;
3. `a4-ol1-kappa-0`;
4. `a7-ol1-kappa-0`.

Use a 9-hour deletion guard per Pod. At the post-preflight `$0.44/GPU-hour`
price, the four-Pod sentinel envelope is `$15.84` compute plus about `$0.28`
prorated Pod storage, maximum `$16.12`. Refresh price, CUDA-12.8 stock, account
balance, and existing resources immediately before creation. Training now
checks the pinned A40 initialization hash before the first optimizer boundary;
any mismatch stops that Pod rather than creating hours of invalid work.

Each Pod uses its condition ID as the `--worker` argument. It is condition-level
parallelism, not DDP. Start one detached authoritative process per Pod and keep
the log/PID in `/workspace/run016-control/<condition>/`.

```bash
cd /workspace/sparsity-spillover
condition=a0-gelu
control=/workspace/run016-control/$condition
mkdir -p "$control"
setsid env PYTHONPATH=/workspace/sparsity-spillover/src \
  /workspace/run016-venv/bin/python -u \
  runs/016-2026-08-31-pythia70m-selected-ladder/02_train.py \
  --worker "$condition" > "$control/train.log" 2>&1 < /dev/null &
echo $! > "$control/train.pid"
```

For A0 and A1, after `03_verify.py --condition <id>` passes, execute
`06_teal_posthoc.py --condition <id>` on the same Pod before teardown. This
avoids retransferring the final checkpoint and cache.

For every condition, retrieve the complete attempt except excluded transient
files, including the final checkpoint and `training_state.pt`. Recompute the
attempt transfer inventory locally and run per-condition verification. Only
then terminate its Pod.

## Phase 3: remaining eight, only after sentinel review

Launch the eight nonzero-kappa A4/A7 conditions in one wave only if live A40
stock supports eight. Otherwise run two waves of four. Never substitute a GPU
type or enable activation checkpointing without an updated execution proposal.

After all 12 attempts are local, run cohort verification and consolidate the
two TEAL artifacts:

```bash
PYTHONPATH=src python \
  runs/016-2026-08-31-pythia70m-selected-ladder/03_verify.py
PYTHONPATH=src python \
  runs/016-2026-08-31-pythia70m-selected-ladder/06_teal_posthoc.py --consolidate
```

Verify all artifact hashes, terminate every Pod, re-list resources, and refresh
posted billing. The existing `EUR-IS-1` network volume is retained only because
it predates Run 016; report its continuing charge separately.
