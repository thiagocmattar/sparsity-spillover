# Run 017 deployment playbook

This checklist is not launch authorization. Refresh live price, stock, balance,
resources, and SSH details immediately before use.

## Provision and guard

Create four independent Secure `NVIDIA H200` Pods for `a7-ol1-kappa-0`,
`a4-ol1-kappa-0`, `a1h-relu`, and `a0-gelu`. Use the pinned image digest from
`config.yaml`, 30 GB container disk, 25 GB Pod volume at `/workspace`, SSH key
injection, and an independent 6.5-hour deletion guard per Pod. Do not attach the
pre-existing `EUR-IS-1` network volume.

Transfer the committed repository bundle to all Pods, verify its SHA-256 and
Git commit, and start `00_setup_remote.sh` detached on each. Transfer both
hash-verified cache files in slowest-condition order while setups run. On every
Pod verify the cache byte counts and SHA-256, clean scoped checkout, pinned
runtime, package freeze, schedule hash, and portable CPU initialization hash.

## Non-evidence H200 gate

On the A7 Pod, run `05_remote_preflight.py` detached with its log and PID under
`/workspace/run017-control`. It must pass before any scientific worker starts.
The preflight creates no attempt. Retrieve its JSON, log, package freeze, and
transfer records even if it fails. A failure deletes all four Pods after those
records are copied; do not edit Run 017 inputs in place.

## Scientific sentinel

After the preflight passes, launch one detached worker per Pod:

```bash
cd /workspace/sparsity-spillover
condition=a7-ol1-kappa-0
control=/workspace/run017-control/$condition
mkdir -p "$control"
setsid env PYTHONPATH=/workspace/sparsity-spillover/src \
  /workspace/run017-venv/bin/python -u \
  runs/017-2026-09-01-pythia70m-selected-ladder-portable-init/02_train.py \
  --worker "$condition" > "$control/train.log" 2>&1 < /dev/null &
echo $! > "$control/train.pid"
```

Poll no more often than every five minutes. Each update includes progress,
current loss, tokens/second, elapsed time, and refreshed ETC. Check sooner only
for a declared warning condition.

After each condition finishes, run `03_verify.py --condition <id>`. Run
`06_teal_posthoc.py --condition <id>` for A0 and A1-H on the same Pods. Retrieve
the complete attempt except excluded transient caches, including final
checkpoint and `training_state.pt`; rebuild and verify the transfer inventory
locally. Delete a Pod only after its local hashes and per-condition verification
pass.

The remaining eight kappa conditions require a separate decision after the
sentinel. After the full cohort is eventually present, run cohort verification
and `06_teal_posthoc.py --consolidate`.
