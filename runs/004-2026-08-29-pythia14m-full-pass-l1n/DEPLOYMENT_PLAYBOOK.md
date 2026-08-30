# Run 004 RunPod deployment playbook

This is the measured deployment path for Run 004. It records infrastructure
only; `config.yaml` remains the scientific contract. Do not change the batch,
sequence length, precision, optimizer, activation/pressure definitions, cache
identity, or validation coverage to make a deployment fit.

## Closeout verdict

Run 004 ultimately completed on five Secure A100 80 GB Pods, not the originally
planned RTX 4090 fleet. The unchanged MB32/GAS32 workload failed closed on a
healthy Secure 24 GB RTX 4090 and passed on A100 PCIe/SXM with approximately
56.95 GiB peak reserved memory. This is implementation-specific evidence for
Run 004, not a universal GPU requirement; every future workload still needs its
own production-shaped preflight.

The reusable deployment order is:

1. lock the scientific config and retained artifacts;
2. preflight the most demanding and control paths on the exact candidate GPU;
3. resolve live GPU capacity and acceptable data-center placement;
4. only then choose location-bound persistent storage;
5. build or stage the immutable cache once and distribute it with hash checks;
6. launch detached workers, monitor end-to-end update elapsed time, retrieve
   resumably, verify every inventory, and delete each finished Pod;
7. refresh lagging billing after teardown and separately report continuing
   storage.

Run 004 showed why this order matters. Creating the network volume first pinned
attached Pods to Secure `EUR-IS-1`, while suitable A100 capacity was secured in
other regions. The retained volume was valuable for durable cache/evidence, but
it did not solve unrestricted multi-region distribution. A first verified A100
therefore became the cache seed; concurrent restricted rsync copied the exact
5.97 GB cache to four workers in 78--84 seconds. Rebuilding MiniPile on every
worker would have repeated roughly 17 minutes of seed work per Pod.

The first ETC model used the synchronized optimizer-boundary timer and omitted
batch construction plus host-to-device staging. Later projections used total
condition elapsed time per completed update. Future calibration artifacts must
name their timer boundary and include all recurring update work in the ETC.

One large recursive SCP retrieval reset; resumable SFTP completed the existing
partial transfer and all declared hashes passed. A copy command is not delivery
evidence without the inventory reconciliation. At final closeout, the live
inventory contains zero Pods and only the intentionally retained 100 GB volume.
Posted Pod charges total `$21.5213452158`, superseding the teardown-time
estimate; the volume continues at `$7/month`. Exact final state is in
`artifacts/closeout.json`.

The sections below retain the measured Run 004 commands and historical prices.
They are a reproducibility record, not a current launch packet: re-query live
CLI schemas, availability, price, and placement before another deployment.

## Durable resources and constraints

- Network volume: `sparsity-spillover-shared`, ID `9luykg5yc3`.
- Location and tier: `EUR-IS-1`, Standard, 100 GB.
- Price observed 2026-08-29: `$0.07/GB/month`, or `$7/month` while retained.
- Pod mount: `/workspace`; Serverless mount: `/runpod-volume`.
- The volume survives Pod termination and can be mounted by multiple workloads,
  but Pod attachment is Secure Cloud only and all attached Pods are constrained
  to `EUR-IS-1`. It must be selected when a Pod is created.
- Multiple readers are safe. Concurrent writers must use distinct paths. Run
  004 workers use distinct worker and attempt directories.
- Keep the immutable master token cache under `/workspace/shared-cache/`.
  Before training, copy it to each Pod's container disk and expose that local
  copy at the config's fixed repo-relative path. This avoids random 8 KiB reads
  from network storage during the 729,088-sequence schedule.
- The shared volume is convenience storage, not the sole backup for completed
  evidence. Retrieve and hash-verify the agreed artifacts locally before Pod
  teardown.

The volume can grow but cannot shrink. Its location can only be changed by
creating another volume and copying data. RunPod does not automatically sync
volumes in different data centers.

## Tools and immutable identities

Use current `runpodctl --help` or RunPod MCP schemas for resource operations.
Two verified CLI binaries were used during this launch:

- `runpodctl 2.8.0-22dc71f` for Pod creation because its CLI exposes the
  absolute `--terminate-after` guard;
- `runpodctl 2.12.0-51ca7f0`, Windows SHA-256
  `f434915e19632097c0ec89d48fac3e25af187e14ee3d172dc37e4d5b2154a7f3`,
  matched against the official release checksum and used for transfer tests.

The production identities are:

- base Git commit: `39e5be7b27a9f0b6df3746b71991e3439526871f`;
- run-code SHA-256:
  `501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d`;
- canonical resolved-config SHA-256:
  `e5eba31eefe5a721c6908abff7cc5a1db6a8dccda50bdc7e6b7ff9d4a8ecf511`;
- image digest:
  `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.

Never put API keys, SSH private keys, or Hugging Face tokens in the run folder,
logs, source packet, or this playbook.

## One-time volume creation

First list existing volumes and data centers. Do not create a duplicate if the
ID above is still present. The equivalent creation command is:

```text
runpodctl network-volume create \
  --name sparsity-spillover-shared \
  --size 100 \
  --data-center-id EUR-IS-1
```

The live 2026-08-29 catalog offered Secure RTX 4090 and A100-SXM-80GB in this
data center. RTX 4090 was `$0.74/GPU-hour`; availability was low. A hardware
substitution requires a new target-GPU preflight and a recorded cost/ETC
revision, but no scientific-config change.

## Create a guarded attached Pod

Generate an absolute RFC3339 deadline immediately before creation. For the
seven-hour Run 004 guard, the command shape is:

```text
runpodctl pod create \
  --name NAME \
  --image runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --data-center-ids EUR-IS-1 \
  --container-disk-in-gb 30 \
  --network-volume-id 9luykg5yc3 \
  --volume-mount-path /workspace \
  --ports 22/tcp \
  --min-cuda-version 12.8 \
  --terminate-after ABSOLUTE_RFC3339_DEADLINE
```

Immediately verify the returned Pod ID, GPU, `$0.74/hour` price, Secure Cloud,
data center, network-volume ID/mount, SSH reachability, and termination guard.
Record the exact Pod ID rather than relying on its name during teardown.

## Stage source and the pinned environment

Create a small archive containing only Run 004, `src/sparsity_research`, tests,
and `pyproject.toml`; record its byte size and SHA-256. On the volume, clone the
base commit and overlay the archive:

```text
cd /workspace
git clone --no-checkout https://github.com/thiagocmattar/sparsity-spillover.git sparsity-spillover
cd sparsity-spillover
git checkout --detach 39e5be7b27a9f0b6df3746b71991e3439526871f
tar -xzf ../RUN004_SOURCE.tar.gz -C .
```

Verify the Git commit, source-archive hash, run-code hash, and resolved-config
hash. Then run the pinned setup detached and log to the persistent volume:

```text
cd /workspace/sparsity-spillover/runs/004-2026-08-29-pythia14m-full-pass-l1n
setsid bash 00_setup_remote.sh > prelaunch/setup-shared-pod.log 2>&1 < /dev/null &
```

`/workspace/run004-venv` intentionally lives on the shared volume. This makes
the exact environment reusable by Pods using the same image and Python ABI.
Do not run concurrent `pip` mutations against it. Confirm `pip check`, the
freeze file, Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.12.1, and CUDA
12.8 before use.

## Build the MiniPile cache directly from Hugging Face

Local-to-Pod peer transfer was rejected as the primary path after two matched
CLI attempts stalled at 74% and 91%. The direct Hugging Face path fetched the
pinned dataset, generated all three Arrow splits, and reproduced the validation
cache exactly in roughly two minutes. Tokenizing and publishing the complete
training cache then took 894.3 seconds (14m54s). Use
`06_build_cache_from_hf.py`; it publishes no split until counts, bytes, and
SHA-256 match.

Keep Hugging Face's raw/download/Arrow cache on disposable container storage.
Only the exact final token cache belongs on the network volume:

```text
mkdir -p /hf-cache
cd /workspace/sparsity-spillover/runs/004-2026-08-29-pythia14m-full-pass-l1n
HF_HOME=/hf-cache \
HF_DATASETS_CACHE=/hf-cache/datasets \
HF_HUB_CACHE=/hf-cache/hub \
TOKENIZERS_PARALLELISM=true \
/workspace/run004-venv/bin/python -u 06_build_cache_from_hf.py \
  --splits validation train --batch-size 256
```

Pinned inputs:

- dataset `JeanKaddour/minipile` at
  `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`;
- tokenizer `EleutherAI/pythia-14m-deduped` at
  `7386d9a4ae45aef494a6e704910394def3037fc5`.

Required outputs:

- validation: 500 documents, 693,668 tokens, 2,774,672 bytes,
  SHA-256 `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`;
- train: 1,000,000 documents, 1,491,711,416 tokens, 5,966,845,664 bytes,
  SHA-256 `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.

The script builds validation first as a cheap tokenizer/order equivalence gate.
Any mismatch is a hard stop; do not bless a new cache identity inside Run 004.
After exact verification, place the durable master at:

```text
/workspace/shared-cache/minipile-pythia-14m-full/
```

If an exact cache already exists on another Pod, Pod-to-Pod `rsync` over a
temporary SSH key is a supported alternative. Cross-region transfer from the
first Pod measured about 8.3 MB/s. Use `--partial --append` for resumability and
require the complete destination SHA-256 afterward. `--append-verify` was tested
but rejected because it re-read the multi-gigabyte prefix across regions before
resuming. Delete the temporary private key and remove its public key from
`authorized_keys` after the verified copy.

## Stage the master cache locally on every worker

Before preflight or training on each Pod:

```text
mkdir -p /run004-local-data
rsync -a --delete \
  /workspace/shared-cache/minipile-pythia-14m-full/ \
  /run004-local-data/minipile-pythia-14m-full/
mkdir -p /workspace/sparsity-spillover/data/tokenized
ln -sfn /run004-local-data/minipile-pythia-14m-full \
  /workspace/sparsity-spillover/data/tokenized/minipile-pythia-14m-full
```

Run the cache verifier after copying. The shared symlink target is deliberately
the same absolute container path on every identically configured Pod.

Observed on Secure Pod `qq6u7wif86vdyk`: the 5.97 GB shared-to-local copy took
14 seconds. Both local file hashes, the normal loader, and the exact schedule
identity passed. The retained log is
`prelaunch/attempt-secure-preflight-r1/secure-cache-localize.log`.

## Preflight and conditional scale-out

Run the exact target-GPU preflight detached:

```text
cd /workspace/sparsity-spillover/runs/004-2026-08-29-pythia14m-full-pass-l1n
setsid env PYTHONDONTWRITEBYTECODE=1 \
  /workspace/run004-venv/bin/python -u 05_remote_preflight.py \
  > prelaunch/remote-preflight.log 2>&1 < /dev/null &
```

It must verify the runtime and both cache hashes, then pass two complete
MB32/GAS32 optimizer boundaries for both ReLU control and lambda 1. Both paths
must be finite, have no FP16 overflow/skipped update, expose the flash SDPA
kernel, and keep peak reserved CUDA memory at or below 90% of the CUDA device's
reported physical memory. The exact device name and byte count are written into
the smoke result; do not carry a prior GPU type's memory ceiling into a
substituted-GPU preflight.

Only after `prelaunch/remote-preflight.json` says `passed`, create the four
remaining guarded Pods with the same image, GPU, data center, and network-volume
ID. On every Pod, verify the shared source/environment, make the disposable
local cache copy, and verify its hash before starting a worker.

Observed on the Secure RTX 4090 at MB32/GAS32: this gate failed closed. ReLU
control had 20.00 GiB allocated and only 3.03 GiB free when another 12.28 GiB
allocation was requested; lambda 1 also failed on a 12.28 GiB request. No
complete boundary or ETC timing was produced, and no scale-out occurred. Do not
reuse the 4090 launch commands for this exact decomposition. A larger-memory
GPU needs its own exact preflight and live price/ETC record. A reduced
microbatch requires explicit scientific/operational approval before changing
the fixed config.

Observed on Secure A100 PCIe 80 GB Pod `zqet1vl0hzsxa0`: the unchanged gate
passed. ReLU control boundaries took 6.384 and 4.806 seconds and reserved
56.941 GiB; ReLU lambda 1 took 6.879 and 5.728 seconds and reserved 56.953 GiB.
Both conditions completed two finite, non-overflowing, non-skipped boundaries;
the device-specific 90% limit was 71.325 GiB. Evidence is retained under
`prelaunch/attempt-a100pcie80-r1/`.

When unrestricted placement is required, do not rebuild MiniPile on each Pod.
Build once on the first verified Pod, retain its Pod volume while it remains a
worker/cache seed, transfer the immutable `train/` and `validation/` cache tree
to every later Pod, and verify both declared token SHA-256 identities before
preflight or training. The retained `EUR-IS-1` network volume remains useful to
Pods created in that data center, but cannot be attached across arbitrary
regions.

## Launch the five workers

Worker assignment is fixed:

- `controls`: GeLU control followed by ReLU control;
- `relu-l1n-0p05`;
- `relu-l1n-0p1`;
- `relu-l1n-0p5`;
- `relu-l1n-1`.

Use unique persistent logs:

```text
cd /workspace/sparsity-spillover/runs/004-2026-08-29-pythia14m-full-pass-l1n
setsid env PYTHONDONTWRITEBYTECODE=1 \
  /workspace/run004-venv/bin/python -u 02_train.py --worker WORKER \
  > /workspace/run004-WORKER.log 2>&1 < /dev/null &
```

Record every Pod ID, worker, PID, launch time, deadline, GPU, price, source hash,
config hash, runtime, local-cache hash, and log path.

## Monitoring, retrieval, and teardown

Run `04_monitor.py` read-only and inspect the authoritative process and worker
log. Monitoring began at five-minute intervals and changed to ten-minute
intervals at the user's explicit direction. Warn on an absent process with a
running manifest, ten minutes without an event, non-finite loss, FP16
overflow/skipped update, memory above 90%, low local or shared disk, a projected
deadline miss, or validation / diagnostic coverage mismatch.

As each single-condition worker becomes terminal:

1. Inspect its remote transfer inventory.
2. Copy manifests, metrics, diagnostics, logs, all declared model snapshots,
   optimizer/scaler/RNG states, and the final checkpoint locally. Use resumable
   SFTP when a large recursive SCP transfer is interrupted.
3. Recompute every declared local SHA-256. Only after every declared file
   passes may that exact Pod be terminated.
4. Re-list Pods to confirm that exact Pod is gone and the unfinished workers
   remain.

The controls Pod is the exception because it runs two conditions sequentially:
retain it until both GeLU and ReLU controls are terminal, then retrieve and
hash-verify both attempts before terminating it. After all six attempts are
local, run `03_verify.py` across the six attempts and confirm the exact data,
schedule, initialization, validation, diagnostic, and checkpoint coverage.
Finally, re-list Pods and volumes. Zero unintended Pods should remain; retain
network volume `9luykg5yc3` intentionally at `$7/month`. Remove ephemeral
Pod-to-Pod keys and disposable local/Hugging Face caches.

Do not terminate the final scientific Pod until artifacts have been retrieved
and verified locally. Do not delete the shared network volume as part of normal
Run 004 teardown.
