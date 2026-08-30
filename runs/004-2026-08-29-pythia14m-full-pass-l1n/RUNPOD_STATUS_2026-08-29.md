# Run 004 RunPod launch status and incident record

## Executive summary

Run 004 is closed with evidence label `valid`. All six conditions completed 712
optimizer updates, full final validation, terminal diagnostics, and the retained
checkpoint inventory. Every remote transfer inventory passed locally, the
six-condition verifier passed, the two post-hoc spillover figures and
observations are complete, and no scientific artifact was rewritten. These are
descriptive observations, not an approved consolidated finding or manuscript
claim.

The authoritative read-only closeout at `2026-08-30T10:17:58Z`, using
checksum-verified `runpodctl 2.12.0-51ca7f0`, returned zero Pods and zero active
GPU cost. Exactly one resource is intentionally retained: 100 GB Standard
network volume `sparsity-spillover-shared` (`9luykg5yc3`) in `EUR-IS-1`, which
continues at `$7/month`. The complete posted Pod history has 24 billing rows
totaling `$21.5213452158`, including the two RTX 4090 preflight Pods, five A100
production Pods, and Pod disk. Posted volume billing through that audit is
`$0.1361111160` and continues accruing.

The successful production path used one Secure A100 PCIe 80 GB Pod and four
Secure A100 SXM 80 GB Pods after the unchanged MB32/GAS32 workload failed the
RTX 4090 memory gate. The sections below preserve the chronological stop-points,
including earlier statements that training had not yet started; they are
historical snapshots and do not override this final summary.

## Step 1 completion update

Approved Step 1 completed at `2026-08-29T20:32:59Z`. The historical audit above
remains unchanged as the record of the earlier stop-point. Since that audit:

- the exact train metadata and the Community cache-build, setup, transfer, and
  failed-preflight logs were copied to persistent storage with a SHA-256
  inventory;
- that evidence was copied locally into `prelaunch/attempt-community-r2/`, where
  the same inventory passes;
- the verified train bytes were atomically renamed to `tokens.int32.bin`, and
  the metadata was published last as `metadata.json`;
- the normal Run 004 loader verified the final train and validation files in
  5.052 seconds;
- the approved full-pass schedule reproduced shape `712 x 32 x 32`, 729,088
  scheduled blocks, 714 wrapped blocks, and SHA-256
  `f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`;
- the retained verification output is
  `prelaunch/step1-cache-verification.log`.

No scientific training or target-GPU optimizer boundary was started. Both Pods
remain live pending separate approval for Step 2, which removes the temporary
transfer credentials and terminates only the failed Community Pod.

## Step 2 completion update

Approved Step 2 completed at `2026-08-29T20:38:42Z`:

- the temporary pod-to-pod public authorization was removed from the Secure
  Pod;
- the transfer key was confirmed rejected afterward;
- the corresponding temporary private/public files were removed from the
  Community Pod;
- the normal account SSH key still reached the Secure Pod, and the persistent
  cache and evidence remained present;
- Community Pod `ottlrd7awawf8h` was terminated with deletion status 204, and a
  follow-up resource read returned 404;
- the live inventory then contained exactly one Pod: Secure RTX 4090
  `qq6u7wif86vdyk`, running with network volume `9luykg5yc3` at `$0.74/hour`;
- the 100 GB Standard volume remained present in `EUR-IS-1`.

No scientific training or preflight optimizer boundary was started. Step 3 is
the separately approved Secure-Pod cache-localization and exact GPU preflight.

## Step 3 completion update

Approved Step 3 completed at `2026-08-29T20:45:59Z`:

- the 5.97 GB immutable master cache copied from the shared volume to disposable
  container storage in 14 seconds;
- both local cache SHA-256 identities passed, as did the normal loader and exact
  `712 x 32 x 32` schedule identity;
- the pinned runtime passed and flash SDPA was available;
- ReLU control at MB32/sequence 2048 failed CUDA OOM with 20.00 GiB already
  allocated, 3.03 GiB free, and another 12.28 GiB requested;
- ReLU lambda 1 also failed on a 12.28 GiB request;
- zero complete optimizer boundaries ran, so the required timing and <=90%
  memory evidence do not exist;
- evidence was copied and SHA-verified locally under
  `prelaunch/attempt-secure-preflight-r1/` and remains on persistent storage.

The Pod returned to an idle state with no training/preflight process. No
additional Pod was created and no scientific attempt was started.

## Step 4 completion update

Approved Step 4 completed at `2026-08-29T20:54:24Z`:

- the exact deletion target was verified as idle Secure RTX 4090 Pod
  `qq6u7wif86vdyk`, attached to volume `9luykg5yc3`;
- pod deletion returned 204 and a follow-up resource read returned 404;
- the full Pod inventory returned zero items;
- volume `9luykg5yc3` remained present as the sole network volume: 100 GB,
  Standard, `EUR-IS-1`;
- active GPU billing fell from `$0.74/hour` to zero;
- no scientific process or artifact was lost because the immutable cache,
  source/environment, and preflight evidence were already persistent.

## Current live resources

Audit source: live RunPod MCP resource reads plus read-only SSH inspection.
RunPod's billing endpoint returned no records yet, so the accrued cost below is
derived from creation timestamps and live hourly prices rather than the lagging
billing total.

| Resource | ID | State at audit | Price | Role |
| --- | --- | --- | --- | --- |
| 100 GB Standard network volume | `9luykg5yc3` | present in `EUR-IS-1` | `$7/month` | Durable source, environment, exact master cache, and future artifact storage |

Creation-time guards were set to terminate the Community Pod at
`2026-08-30T02:12:35Z` and the Secure Pod at `2026-08-30T02:43:55Z`. The
current MCP Pod representation does not echo those schedules, so the accepted
creation records remain the schedule provenance.

At the latest `2026-08-29T20:54:24Z` audit, approximate GPU charges derived
from creation/termination timestamps were:

- terminated Community GPU: about `$0.486`;
- terminated Secure GPU: at most about `$0.861`;
- combined GPU compute: at most approximately **`$1.347`**, plus small storage
  accrual. This estimate no longer increases because no Pod remains.

The original `$12.17` maximum applied to five Community Pods without a network
volume. The user-requested reusable volume changes the feasible Pod path:
RunPod network volumes are data-center-bound and Pod attachment requires Secure
Cloud. Five seven-hour Secure RTX 4090 guards plus the original Community
preflight produce a revised Run 004 maximum of about `$28.55`; the retained
volume is the separate ongoing `$7/month` resource. No four-Pod scale-out has
occurred, so the maximum is not the current spend.

## Scientific contract: unchanged

None of the deployment retries changed the approved experiment:

- six conditions: GeLU control, ReLU control, and ReLU `l1_naive` at `h` with
  lambda in `{0.05, 0.1, 0.5, 1.0}`;
- random Pythia-14M initialization from the pinned architecture revision;
- 2,048-token sequences;
- global batch 1,024 sequences, implemented as microbatch 32 and gradient
  accumulation 32;
- 712 optimizer updates and the approved full-pass wrap arithmetic;
- peak LR `1e-3`, mapped Pythia schedule/optimizer choices, FP16 dynamic loss
  scaling, and global L2 gradient clipping at 1.0;
- complete 500-document validation coverage: 338 complete blocks and the
  explicit 1,444-token excluded tail;
- the complete approved diagnostics and checkpoint-retention inventory.

The changes described below are infrastructure and data-staging decisions only.
No fallback batch reduction, activation checkpointing, precision change,
optimizer change, cache redefinition, released-weight initialization, or
validation reduction was made.

## Timeline and what happened

### 1. Initial implementation and approval

The design and staged launch were explicitly approved. The implementation had
already passed 18 focused checks and the 90-test bootstrap suite. The local CUDA
smoke correctly refused to estimate flash-attention behavior because the local
Windows PyTorch build lacked that kernel. This made a real target-GPU preflight
mandatory before cloud scale-out.

### 2. First Pod request: no capacity and no bill

The first Community RTX 4090 creation request at approximately 16:09 -03:00 was
rejected because matching capacity disappeared. A resource inventory confirmed
that the failed request created no Pod or network volume and incurred no bill.
At that point the original approved policy did not permit an automatic cloud or
GPU substitution, so execution stopped and reported the capacity failure.

The user then explicitly authorized continued retries and GPU-type changes,
emphasizing that the launch should continue.

### 3. Community retry succeeded

Community RTX 4090 capacity returned. Pod `ottlrd7awawf8h` was created at
`2026-08-29T19:12:56Z` for `$0.34/hour`, with a 30 GB container disk, a 25 GB
Pod volume at `/workspace`, the pinned image digest, and a seven-hour guard.

The source archive was verified before extraction. The base Git commit and
dirty overlay model were preserved. `00_setup_remote.sh` installed and verified
Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.12.1, and the remaining pinned
dependencies. Because the environment occupied about 7.2 GB, it was moved from
the 25 GB Pod volume to the 30 GB container disk and exposed through a symlink.
That was an infrastructure-only space optimization.

### 4. Local-to-Pod cache transfer failed twice

The original plan transferred the already-built 5.97 GB token-cache archive
from the local machine. This was the wrong data path for this network route:

- `runpodctl 2.8` progressed quickly at first, then stalled near 74% at very low
  throughput and was cancelled;
- a checksum-verified `runpodctl 2.12.0-51ca7f0` retry reached about 91%, then
  degraded to tens of kilobytes per second with multi-hour ETAs and was also
  cancelled;
- each incomplete remote copy was removed; no partial archive was treated as
  valid.

The Community Pod offered only proxy SSH, not direct SCP/SFTP, which made the
peer-to-peer relay path more fragile. Matching the CLI version did not solve the
underlying transfer-route behavior.

### 5. Direct Hugging Face cache construction worked

The user correctly suggested downloading MiniPile directly from Hugging Face
instead of routing a multi-gigabyte cache through the local machine. A focused
run-specific builder, `06_build_cache_from_hf.py`, was added and transferred as
a 10.1 KB file with a matching source SHA-256. It pins:

- dataset `JeanKaddour/minipile` revision
  `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`;
- tokenizer `EleutherAI/pythia-14m-deduped` revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`.

The builder publishes a split only after its document count, token count, byte
size, tail, complete-block count, and SHA-256 all match the approved local cache.
It built validation first as a cheap tokenizer/order equivalence gate.

Observed results:

- Hugging Face download, Arrow preparation, and exact validation gate: roughly
  two minutes end to end;
- validation tokenization/publish portion: 0.683 seconds;
- validation identity: 500 documents, 693,668 tokens, 2,774,672 bytes, 338
  complete blocks, 1,444-token tail, exact approved SHA-256;
- train tokenization and publish: 894.326 seconds (14m54s);
- train identity: 1,000,000 documents, 1,491,711,416 tokens,
  5,966,845,664 bytes, 728,374 complete blocks, 1,464-token tail, exact approved
  SHA-256.

This path was both faster and more reliable than local relay transfer. Raw Hugging
Face and Arrow caches were kept on disposable container storage; only the final
int32 cache was retained as durable data.

### 6. Persistent shared storage was added

After observing the repeated setup/transfer cost, the user explicitly requested
persistent storage accessible to later RunPod workloads. Live RunPod constraints
made the trade-off important:

- a network volume survives Pod deletion and can be attached to multiple
  workloads;
- it is pinned to one data center;
- Pod attachment is Secure Cloud only and must be chosen at Pod creation;
- it can grow but cannot shrink;
- Standard storage costs `$0.07/GB/month` for the first terabyte.

A 100 GB Standard volume, `sparsity-spillover-shared` (`9luykg5yc3`), was
created in `EUR-IS-1`. That location had live RTX 4090 availability and an A100
fallback, and it supports the S3-compatible access path. It is intentionally
retained after Run 004 at `$7/month`.

Secure Pod `qq6u7wif86vdyk` was then created in the same data center at
`$0.74/hour`, attached to that volume at `/workspace`, and guarded for seven
hours. This is why the webapp now shows two Pods.

### 7. Shared source and environment worked, but setup was slower

The network volume was initially empty. The source packet alone did not contain
`.git`, while the setup contract requires definitive Git provenance. Rather
than faking Git state, the staging procedure was corrected:

1. clone the repository;
2. detach at base commit
   `39e5be7b27a9f0b6df3746b71991e3439526871f`;
3. overlay only Run 004, `src/sparsity_research`, tests, and `pyproject.toml`;
4. verify the archive and cache-builder hashes;
5. run the pinned setup.

The shared `/workspace/run004-venv` completed with `pip check` clean. Installing
many small environment files directly on network storage took about eleven
minutes, noticeably slower than local Pod storage. The benefit is that later
identical-image Pods can reuse the exact environment without reinstalling it.
Imports are also slower from network storage, but the training worker imports
once, so this is expected to be a startup rather than per-update cost.

Two CUDA checks passed on this Secure Pod:

- the image's PyTorch 2.8.0+cu128 saw one RTX 4090;
- the shared Run 004 PyTorch 2.11.0+cu128 environment saw one RTX 4090, reported
  CUDA available, and reported both flash-attention availability and flash-SDPA
  enabled.

### 8. Cross-region cache promotion worked, with one transport adjustment

The exact Community cache was copied Pod-to-Pod into the `EUR-IS-1` network
volume using a temporary SSH key and resumable `rsync`.

An initial growing-file snapshot copied about 4.07 GB in 8m38s. A first final
resume used `--append-verify`; it spent roughly 2m39s re-reading the existing
multi-gigabyte prefix across regions without extending the destination. That
mode was cancelled because the complete destination SHA-256 was already the
hard integrity gate. Plain `--append` then sent the remainder in 4m39s at about
6.44 MB/s average. This transport adjustment cannot silently accept bad data:
the destination file is unusable unless its full SHA-256 matches afterward.

At the audit, the shared file had the exact expected byte size and independently
matched the approved training SHA-256. Validation also matched exactly. The
temporary train filename and absent train metadata are the only remaining cache
publication steps.

### 9. The first target-GPU preflight failed for infrastructure reasons

On the Community Pod, `05_remote_preflight.py` verified the pinned runtime and
both exact caches, then attempted the GPU smoke. At that point:

- `nvidia-smi` reported the RTX 4090 and 24,564 MiB;
- `/dev/nvidia*` device nodes were present;
- PyTorch reported one device at the low-level count stage;
- `torch.cuda.is_available()` emitted `CUDA unknown error` and returned false;
- the smoke raised `RuntimeError: CUDA is required for the prelaunch smoke.`

No `remote-preflight.json` was published and no optimizer boundary was run.
The live MCP representation later showed the Community Pod as `RUNNING` but
with no runtime object. Together with the fully healthy Secure RTX 4090 using
the same image and pinned environment, this strongly indicates a faulty or lost
GPU/container binding on that Community allocation, not a scientific-code,
cache, or pinned-runtime mismatch. This is an inference from the combined
evidence; it is not a provider-issued root-cause report.

## What worked

- Explicit design and launch gates were followed before scientific execution.
- A Community RTX 4090 was eventually allocated without changing the science.
- Exact source/archive/config/code identities were checked.
- The pinned environment installed and passed dependency checks on both storage
  layouts.
- Direct pinned Hugging Face acquisition reproduced both approved token caches
  exactly and was substantially better than local relay transfer.
- The 100 GB network volume was created and attached correctly.
- Git provenance on the shared volume was corrected through clone-plus-overlay.
- The Secure Pod's system and pinned environments both see CUDA and flash SDPA.
- Resumable Pod-to-Pod transfer completed, and both destination file hashes
  match exactly.
- A reusable deployment playbook now records the measured path.
- After infrastructure-document updates, all 12 Run 004 focused tests pass.

## What did not work

- The first Pod creation request had no capacity.
- Large local-to-Pod `runpodctl` transfers stalled twice, even after matching
  CLI versions.
- Proxy-only SSH on the Community Pod prevented a straightforward direct SCP
  path.
- The first shared setup attempt used an overlay without `.git` and correctly
  failed the provenance guard. The folder was preserved, then staging was
  redone through a real clone at the base commit.
- Installing the reusable virtual environment on network storage was slower
  than local disk because it writes many small files.
- Cross-region `rsync --append-verify` was counterproductive; final whole-file
  hashing is the better integrity gate for this one-time promotion.
- The Community Pod lost or never achieved a usable PyTorch CUDA binding even
  though `nvidia-smi` could see the GPU.
- The healthy Secure RTX 4090 cannot fit the approved MB32/sequence-2048 path;
  both sampled conditions failed before one complete optimizer boundary.

## What we learned

1. **Download data near compute.** For public, revision-pinned Hugging Face data,
   direct download plus exact identity verification is both faster and more
   reliable than relaying a prepared multi-gigabyte archive through a local
   workstation.
2. **A durable master cache removes repeated preparation, but does not imply
   training directly from network storage.** Run 004's seeded schedule performs
   many small random 8 KiB sequence reads. The plan is therefore to keep the
   immutable master on the volume, copy its 5.56 GiB to each Pod's disposable
   container disk, and train from the local copy. This should preserve startup
   reuse without making network latency part of every update.
3. **Persistent storage changes placement and price.** The network volume solves
   recurrence, but constrains Pods to Secure Cloud in `EUR-IS-1`, raises RTX 4090
   cost from `$0.34` to `$0.74/hour`, and reduces cross-data-center failover.
4. **Environment reuse is viable but should be treated as image/ABI-specific.**
   The shared venv is appropriate only for the same pinned image/Python ABI and
   must not be mutated concurrently.
5. **A running Pod is not a verified GPU.** `nvidia-smi`, the provider status,
   PyTorch CUDA initialization, flash availability, and the exact workload
   boundary are separate gates.
6. **Do not call transferred bytes a published cache.** Size and hash can pass
   while the file still has a staging name and lacks metadata. Atomic promotion
   remains a distinct auditable step.
7. **Preserve evidence from infrastructure failures.** The failed CUDA log is
   useful because it distinguishes a provider/allocation problem from a model
   memory-fit or scientific-code failure.
8. **The exact workload boundary is the capacity gate.** A healthy 24 GB GPU,
   flash attention, and a small parameter count do not guarantee MB32 fit; the
   sequence-by-vocabulary training path can dominate memory. The observed OOM,
   rather than model parameter count, governs the next hardware/decomposition
   decision.

## Repository records created or changed

- `06_build_cache_from_hf.py`: pinned, exact-gated direct Hugging Face builder;
- `DEPLOYMENT_PLAYBOOK.md`: reusable source/environment/cache/preflight/worker/
  retrieval/teardown procedure;
- `prelaunch/launch-plan.json`: append-only capacity, Pod, volume, transfer,
  cache-build, and revised cost records;
- `README.md`: current infrastructure decisions and implications;
- `tests/test_run_004_full_pass.py`: revised explicit infrastructure-envelope
  assertions;
- this status/incident record.

The cache builder compiled and its CLI was checked locally. The updated Run 004
focused suite is 12/12 passing. The earlier complete bootstrap suite was 90/90
before these infrastructure-only additions; it has not yet been rerun after the
new documentation/cache-builder additions.

## Exact state: done, pending, and not started

### Done

- scientific implementation and prior full bootstrap verification;
- first capacity-failure audit;
- two guarded RTX 4090 Pods provisioned during the incident, with the failed
  Community Pod subsequently terminated and one Secure Pod retained;
- persistent 100 GB network volume;
- exact direct-Hugging-Face validation and train cache construction;
- pinned shared source and environment;
- exact shared validation hash;
- exact shared train byte size and SHA-256;
- atomic train-cache and metadata publication under their final names;
- normal loader and full-pass schedule verification;
- persistent and local SHA-verified Community infrastructure evidence;
- removal of the temporary transfer credentials;
- verified termination of the failed Community Pod;
- measured 14-second shared-to-local cache localization and exact local cache/
  schedule verification;
- exact Secure RTX 4090 preflight execution and SHA-verified OOM evidence;
- verified termination of the failed Secure Pod, zero-Pod inventory, and
  continued presence of the separate persistent volume;
- measured deployment playbook and focused-test reconciliation.

### Pending before scientific launch

- choose either a larger-memory GPU or explicitly approve a different
  microbatch/accumulation decomposition and its numerical/throughput trade-off;
- run and pass the same two-boundary exact-target preflight on that approved
  execution definition;
- update ETC and cost from measured boundary timings;
- only then create the remaining workers and launch the six fixed conditions.

### Not started

- no Run 004 scientific attempt;
- no optimizer update;
- no condition validation;
- no checkpoint or diagnostic artifact;
- no four-Pod scale-out.

## Recommended next sequence

1. Price the viable larger-memory GPU candidates in `EUR-IS-1`, or define the smallest
   microbatch/GAS probe that preserves global batch 1,024 if that trade-off is
   approved.
2. Run a fresh exact two-boundary preflight for the chosen definition and use
   its measurements to replace ETC/cost estimates.
3. Provision the remaining workers only after that gate passes.
4. Launch all workers only after every Pod has verified the source,
   environment, and local cache copy.
5. Monitor every five minutes; retrieve and locally hash-verify all agreed
   artifacts before terminating scientific Pods.
6. Retain volume `9luykg5yc3` intentionally at `$7/month`, but leave no
   unintended Pods running.

## External operational references

- [RunPod network volumes](https://docs.runpod.io/storage/network-volumes)
- [RunPod Pod storage types](https://docs.runpod.io/pods/storage/types)
- [RunPod S3-compatible network-volume access](https://docs.runpod.io/storage/s3-api)

## Continuation: unrestricted placement and successful A100 preflight

At the user's direction, the `EUR-IS-1` restriction was removed and GPU types
could be substituted while preserving the complete scientific configuration.
Secure A100 PCIe 80 GB Pod `zqet1vl0hzsxa0`
(`run004-preflight-a100pcie80-r1`) was created in `CA-MTL-3` at
`2026-08-29T21:02:05Z` for `$1.39/hour`. It uses the pinned image digest, 30 GB
container disk, 25 GB persistent Pod volume at `/workspace`, direct SSH, and a
three-hour automatic termination guard. The separately retained 100 GB network
volume remains in `EUR-IS-1`; it cannot be attached to this cross-region Pod.

The source archive SHA-256 was
`5bfca065217c9b474a2d756d6e394efd97628ab7203ccec4eefb1a5e795694be`.
The public repository was checked out at base commit
`39e5be7b27a9f0b6df3746b71991e3439526871f`, then the inventoried Run 004
overlay was applied. The pinned environment realized Python 3.12, PyTorch
2.11.0+cu128, CUDA runtime 12.8, and Transformers 5.12.1 on an NVIDIA A100 80GB
PCIe. Setup took roughly eight minutes on the Pod's persistent MFS-backed
volume.

The Pod downloaded MiniPile from Hugging Face at the pinned dataset revision
and tokenized it once. Validation reproduced 500 documents, 693,668 tokens,
1,444 excluded tail tokens, and SHA-256
`51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`.
Train reproduced 1,000,000 documents, 1,491,711,416 tokens, 1,464 excluded tail
tokens, 5,966,845,664 bytes, and SHA-256
`da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.
Train tokenization and atomic publication took 1,034.9 seconds. This cache is
the unrestricted-placement seed: later Pods receive the immutable tree and
verify both hashes; they do not download and tokenize MiniPile again.

The original preflight reporter still held the initial 4090's 24 GiB target.
That would have falsely judged a substituted A100 against 4090 capacity. The
reporting/gating interface was minimally changed to accept the actual CUDA
device name and reported physical memory from `05_remote_preflight.py`. The
focused Run 004 suite remained 12/12 passing, and the full bootstrap suite is
99/99 passing after the evidence/documentation update. No model, initialization, data,
optimizer, pressure, precision, sequence length, MB32/GAS32 decomposition, or
validation/diagnostic setting changed.

The first detached-launch helper invocation started nothing because its
existence guard expected flat cache filenames, while the declared layout is
`train/tokens.int32.bin`, `train/metadata.json`, and the corresponding
`validation/` files. The four checks were corrected. This caused no optimizer
boundary and no partial result. The retry launched PID 2183 and the exact
preflight passed at `2026-08-29T21:36:20Z`:

- ReLU control: 6.3836 and 4.8058 seconds per complete update; 56.9414 GiB peak
  reserved;
- ReLU lambda 1: 6.8791 and 5.7280 seconds per complete update; 56.9531 GiB peak
  reserved;
- actual 90% device limit: 71.3248 GiB;
- both paths: two complete finite boundaries, no FP16 overflow, no skipped
  optimizer update, flash attention available.

The remote result, exact smoke, cache/setup/preflight logs, pip freeze, cache
metadata, and staged preflight sources were retrieved to
`prelaunch/attempt-a100pcie80-r1/`. Principal local hashes match the remote
inventory: `remote-preflight.json` is
`8080d61236eb1239f1c3e372b1a78f2dd6095aac0268d42485fe67024d536d1b` and the
preflight log is
`bee3f3f78068302a9a2d6c52de9ea1625ec5bd7d2785beb9e8b98b5dd706b047`.

At `18:40:43 -03:00`, the live catalog listed A100 SXM 80 GB Secure at
`$1.59/hour` in several low-stock data centers. A practical five-worker fleet
is the retained A100 PCIe at `$1.39/hour` plus four A100 SXM workers at
`$1.59/hour`. The existing Pod should receive a single pressure condition so
it completes before its shorter guard; a new guarded SXM worker receives the
two sequential controls; three other SXM workers receive the remaining pressure
conditions. Measured steady timings imply 1.901 training-only hours for the
two-control worker and 1.133 training-only hours for a pressure worker. With
parallel setup/cache staging, checkpoints, full validation, and post-hoc
diagnostics, the expected wall ETC from scale-out approval is 2.6-3.1 hours,
or approximately 21:16-21:46 local if approved at the estimate timestamp.

Expected remaining compute is about `$14.50` if each completed worker is
retrieved and terminated promptly. A conservative three-hour envelope with all
five online is `$23.25`; four-hour guards on the full fleet cap that new
envelope at `$31.00`. The two terminated 4090 attempts accrued at most about
`$1.347`; the successful A100 had accrued about `$0.895` through the estimate
timestamp. The intentionally retained network volume remains a separate
`$7/month`. No Run 004 scientific training, validation, checkpoint, diagnostic,
or four-Pod scale-out has started as of this continuation record.

### Step 8: five-GPU fleet secured

At `2026-08-29T21:44:48Z` through `21:45:03Z`, four unrestricted Secure A100
SXM 80 GB workers were requested with the pinned image, one GPU, 30 GB
container disk, 25 GB persistent `/workspace` volume, port 22, CUDA >=12.8,
and termination at `2026-08-30T01:45:00Z`. Three initial requests succeeded.
The first `relu-l1n-0p05` allocation lost a low-stock machine race, returned a
no-resources error, created no Pod, and incurred no bill. Its immediate retry
succeeded.

The live inventory after readiness verification is:

- `zqet1vl0hzsxa0`: A100 PCIe 80 GB, CA, `$1.39/hour`, planned `relu-l1n-1`;
- `7b41lfe965txbh`: A100 SXM 80 GB, US, `$1.59/hour`, planned controls;
- `58ukutbtquykle`: A100 SXM 80 GB, US, `$1.59/hour`, planned
  `relu-l1n-0p05`;
- `6pi00lbzvp6kpx`: A100 SXM 80 GB, US, `$1.59/hour`, planned
  `relu-l1n-0p1`;
- `15vi5yw3cw1u1j`: A100 SXM 80 GB, US, `$1.59/hour`, planned
  `relu-l1n-0p5`.

All five report `RUNNING`; all four new Pods expose direct public SSH. Active
GPU rate is `$7.75/hour`. Step 8 performed provisioning only: the new workers
have not received source, runtime, or cache, and no scientific process has
started.

### Step 9: source, runtime, cache, and A100 SXM gate passed

Step 9 staged the fixed source archive and pinned environment on all four A100
SXM workers. The archive initially retained its local descriptive basename,
while the safety gate expected `/workspace/RUN004_SOURCE.tar.gz`; all four
launchers stopped before extraction. Renaming the already verified archive to
its declared remote name resolved this without changing its SHA-256. Three
workers installed in roughly 5-6 minutes. The `US-MO-1` MFS-backed worker took
about 15 minutes, mainly during package-file installation and the final Torch
import, but its child processes remained active and it completed normally.

An attempted unrestricted root transfer-key authorization was rejected before
being applied. The replacement used `restrict` plus forced command
`/usr/bin/rrsync -wo` confined to the exact Run 004 cache directory. Its first
literal construction lost the forced-command quotes during shell parsing and
was invalid; the tagged malformed line was replaced from an uploaded literal
public-key file. The key never offered an unrestricted shell. Four compressed
rsync transfers then ran concurrently from the A100 PCIe seed:

- controls: 84 seconds;
- `relu-l1n-0p05`: 82 seconds;
- `relu-l1n-0p1`: 80 seconds;
- `relu-l1n-0p5`: 78 seconds.

Each destination independently reproduced train SHA-256
`da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`
and validation SHA-256
`51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`.
The tagged authorization was removed from all four workers, the key was shown
to receive `Permission denied`, and its private/public files were deleted from
the seed.

The first readiness-script invocation from `/tmp` lacked the Run 004 import
path and exited before writing an artifact. Rerunning with explicit
`PYTHONPATH` produced four passing readiness records. Every A100 SXM worker has
the identical pip-freeze SHA-256
`2fb0bddde1bd36623b940bb81983cff39e2011ac0a60afb7821eb449d65f08b5`,
pinned Python/PyTorch/Transformers/CUDA runtime, flash attention, exact cache
identities, schedule SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`,
and run-code SHA-256
`501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d`.

The controls A100 SXM exact preflight passed at
`2026-08-29T21:59:36Z`. ReLU control boundaries took 5.4626 and 4.7166 seconds;
ReLU lambda 1 took 6.8320 and 5.6259 seconds. Both completed twice without
overflow or skipped update, reserving 56.9414 and 56.9531 GiB against a
71.3248 GiB limit. Local evidence lives in
`prelaunch/step9-a100-fleet-readiness/`; the preflight result SHA-256 is
`2ffa1854b0229b81d10be1e8106f27799a1e9192a6bd1902ab8fdf7399c3b3a2`.

Updated measured ETC from scientific-launch approval is 2.2-2.6 hours, or
approximately 21:19-21:43 local if approved at 19:07. Expected remaining
compute is `$11.25-$12.94` with prompt per-worker artifact retrieval and
teardown; keeping all five online for the entire conservative 2.6 hours would
cost `$20.15`. All GPUs were idle at the Step 9 closing audit. No scientific
attempt, validation, checkpoint, or diagnostic has started.

### Step 10: scientific run launched

After explicit Step 10 approval, the five detached scientific workers launched
at approximately `2026-08-29T22:10Z`:

- A100 PCIe `zqet1vl0hzsxa0`: `relu-l1n-1`, PID 2775, attempt
  `006-20260829-221054-e7b8a083`;
- A100 SXM `7b41lfe965txbh`: `controls`, PID 1082, first attempt
  `001-20260829-221007-bb5288c8` for GeLU control;
- A100 SXM `58ukutbtquykle`: `relu-l1n-0p05`, PID 893, attempt
  `003-20260829-221010-062e6eae`;
- A100 SXM `6pi00lbzvp6kpx`: `relu-l1n-0p1`, PID 949, attempt
  `004-20260829-221037-231415c7`;
- A100 SXM `15vi5yw3cw1u1j`: `relu-l1n-0p5`, PID 893, attempt
  `005-20260829-221010-866aa09f`.

The first authoritative snapshot found GeLU control at step 16 and the four
pressure conditions at steps 9, 6, 10, and 4. Every observed loss was finite,
FP16 loss scale remained 4096, no gradient overflow or skipped optimizer update
occurred, and reserved CUDA memory matched the passed preflight envelope.
Five-minute read-only monitoring began immediately.

### Step 11: live monitoring

The startup, `19:16`, `19:22`, `19:28`, and `19:31` BRT snapshots all found
the five detached worker processes alive and advancing. At the `19:31` snapshot,
GeLU control was at update 242/712; the lambda 0.05, 0.1, 0.5, and 1.0 pressure
conditions were at updates 159, 152, 167, and 137. Observed task losses were
finite, the FP16 loss scale remained 4096, current events recorded no gradient
overflow or skipped optimizer update, and reserved CUDA memory remained
61,169,729,536 bytes for the pressure workers and 61,513,662,464 bytes for the
control worker. These are below the 90 percent warning threshold from the exact
A100 preflight.

The attempt directories were already about 450--507 MB at `19:28`, consistent
with the approved early checkpoint cadence. All workers reported ample storage
headroom. Sustained update times remained approximately 4.7--4.9 seconds for
GeLU and 5.6--5.8 seconds for pressure conditions. The resulting live ETC is
approximately `20:25--20:35` BRT for the four pressure workers, including
post-hoc work, and `21:20--21:30` BRT for both sequential controls. Step 11
remains in progress; no artifact has yet been retrieved and no live Pod has
been stopped.

At the user's explicit direction after the `19:36` snapshot, controller polling
was changed from every five minutes to every ten minutes. This is an operational
monitoring change only: the detached scientific processes, source packet,
resolved scientific configuration, and recorded run-code hash are unchanged.
The implication is that a new stall can take up to approximately ten minutes
to detect instead of five. The ten-minute stale-event warning condition remains
in force.

The `19:56` snapshot found GeLU at update 522/712 and the lambda 0.05,
0.1, 0.5, and 1.0 pressure conditions at updates 342, 333, 359, and 301.
Cumulative scans of all five event logs found zero gradient-overflow records,
zero skipped optimizer updates, and zero NaN or Infinity records.

This snapshot also exposed an ETC-estimation omission. The recorded
`step_wall_seconds` measures the synchronized optimizer boundary after batches
have been constructed; it excludes the per-update `microbatches_for_step`
construction and host-to-device staging performed immediately before that
timer. The original preflight ETC used boundary timing and therefore
underestimated actual pressure-worker wall time. No training input or running
process changed. Estimating from authoritative condition elapsed time per
completed update gives approximately `20:45--21:05` BRT for the four pressure
workers and retains approximately `21:20--21:30` BRT for both controls. At
`19:56`, remaining compute is estimated at roughly `$7.50--$9.00` if completed
workers are retrieved, hash-verified, and terminated promptly.

At `20:06`, the corrected lambda-1 projection exposed a deadline risk on A100
PCIe Pod `zqet1vl0hzsxa0`: its three-hour creation-time termination guard ends
at approximately `21:02` BRT, while training plus final diagnostics and
retrieval can cross that time. The pinned CLI, REST v2 OpenAPI schema, and
documented GraphQL `PodEditJobInput` all show that `terminateAfter` is accepted
at creation but is not a mutable active-Pod field. No signed-in browser session
was available to inspect a possible console-only cancel/extend control. The
user was therefore asked to cancel the guard or extend it to at least `22:00`
BRT without restarting, resetting, stopping, or editing the container.

As a non-mutating data-safety action, the immutable lambda-1 step-256 recovery
checkpoint was copied locally under `prelaunch/emergency-guard-backup/` while
training continued. Its five files total approximately 163 MB and all remote
and local SHA-256 values match. This is a recovery safety point, not final
artifact retrieval.

At the `20:16` snapshot, GeLU attempt
`001-20260829-221007-bb5288c8` was `completed` after all 712 updates, final
checkpoint reload, full validation, and diagnostics. The controls worker had
already started fresh ReLU-control attempt `002-20260829-231305-195b22c6` and
reached update 37. The four pressure attempts were at updates 493, 479, 514,
and 435 for lambda 0.05, 0.1, 0.5, and 1.0 respectively.

The user confirmed that the lambda-1 Pod's termination guard had been cancelled
or extended to at least `22:00` BRT. The user also approved per-worker teardown
after complete local retrieval and hash verification. Attempts
`003-20260829-221010-062e6eae`, `004-20260829-221037-231415c7`, and
`005-20260829-221010-866aa09f` completed the lambda 0.05, 0.1, and 0.5
conditions. Each attempt declared 60 transfer files; the verified local byte
counts were 1,014,222,381, 1,014,220,023, and 1,014,220,259 respectively, with
zero missing, size-mismatched, or SHA-256-mismatched files. Exact Pod identity
was checked immediately before deletion. Pods `58ukutbtquykle`,
`6pi00lbzvp6kpx`, and `15vi5yw3cw1u1j` were then deleted successfully and each
returned `404 not_found` on the closing read.

The `21:07` BRT snapshot found lambda-1 attempt
`006-20260829-221054-e7b8a083` completed after all 712 updates, final checkpoint
reload, full 338-block validation, and diagnostics. Its final reloaded-checkpoint
validation loss was 5.10227634215496. A first recursive SCP copy was interrupted
by an SSH connection reset after only 10 files and approximately 24 MB had
arrived. The Pod and remote attempt remained intact. A recursive SFTP `reget`
resumed the existing files and completed the transfer without restarting from
zero. All 60 declared files and 1,014,217,543 declared bytes then passed local
size and SHA-256 verification. Exact identity was rechecked and Pod
`zqet1vl0hzsxa0` was deleted successfully.

The locally verified pressure-condition final validation losses are 5.2061491323
for lambda 0.05, 5.1654738505 for lambda 0.1, 5.1126882931 for lambda 0.5,
and 5.1022763422 for lambda 1.0. These are artifact facts, not yet a scientific
comparison: cross-condition verification and the approved diagnostic analysis
remain pending.

At `21:14` BRT, exactly one Pod remained: controls Pod `7b41lfe965txbh` at
`$1.59/hour`. ReLU control was healthy at update 664/712 with finite loss,
loss scale 4096, no overflow, and no skipped optimizer update. Its projected
completion was approximately `21:19--21:21` BRT. The intentional 100 GB network
volume `9luykg5yc3` remained present and unchanged. The active GPU rate had
fallen from `$7.75/hour` at launch to `$1.59/hour`.

### Steps 12--13: final controls, retrieval, verification, and teardown

At `21:19:45` BRT, the controls worker was terminal. GeLU control attempt
`001-20260829-221007-bb5288c8` and ReLU control attempt
`002-20260829-231305-195b22c6` had each completed 712 optimizer updates, final
checkpoint reload, all 338 complete validation blocks, diagnostics, and transfer
inventory generation. Their final validation losses were 5.208582500028893 and
5.269645995642307. Recursive resumable SFTP copied both attempts locally. Each
declared 60 files; 1,013,910,055 and 1,013,914,278 declared bytes respectively
passed local size and SHA-256 verification with zero failures.

With all six attempts local, `03_verify.py` returned `verified 6` and wrote
`artifacts/verification.json`. It confirms 4,272 optimizer updates,
8,959,033,344 aggregate training input tokens, one initial parameter SHA-256
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`,
one full-pass schedule SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`,
and one run-code SHA-256
`501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d`.
The local artifact tree contains 370 files and 6,084,775,993 bytes.

A subsequent strict JSONL scan exposed one parser-level anomaly not checked by
`03_verify.py`. ReLU control `events.jsonl` begins with 4,055 NUL bytes followed
by a complete valid step-1 JSON object. The raw 597,831-byte file has SHA-256
`33484adf815c554e549dd5fde1f1475c2a748dd4d52e9303155b4960ae3f0862`,
exactly matching the remote transfer inventory; it was therefore preserved
unchanged. Removing exactly that leading prefix for parsing recovers step 1:
task loss 11.036670565605164, pre/post-clip norms 2.3357067108154297 and
0.9999995337145842, clipped true, loss scale 4096, no overflow, and no skipped
update. The normalized log has all steps 1--712. Across all six normalized
event histories there are zero missing updates, zero overflow records, zero
skipped optimizer updates, and zero non-finite recorded losses. The machine-
readable provenance and normalization rule are in
`artifacts/event-history-audit.json`; no scientific artifact was rewritten.

The focused Run 004 suite passed 12 tests after retrieval. The repository-wide
suite produced 107 passes and one unrelated failure in
`tests/test_run_007_a4z_threshold_ol1_local.py`: that test expects Run 007's
launch-plan status to be `awaiting_launch_approval`, while the current Run 007
plan says `approved_for_launch`. No Run 004 assertion failed, and this audit did
not change another run's approval record.

Only after the six-condition verifier passed was controls Pod
`7b41lfe965txbh` deleted. Its closing get returned `404 not_found`. The final
RunPod audit returned an empty Pod list. Network volume `9luykg5yc3` remained
present in `EUR-IS-1` at 100 GB as explicitly requested. Active GPU billing is
zero; the retained volume continues at `$7/month`.

### Final billing and elapsed time

At the original teardown, the provider billing API had not yet posted the final
partial-hour buckets, so this record estimated the final Pod total at
`$17.67--$17.80`. A closeout refresh at `2026-08-30T10:17:58Z` returned all 24
hourly Pod rows and a final posted total of `$21.5213452158`. This includes the
two RTX 4090 preflight Pods, five A100 production Pods, and Pod disk, and
supersedes the earlier estimate. The difference demonstrates that a teardown-
time reconstruction is provisional until lagging provider buckets post.

The same refresh returned `$0.1361111160` in posted network-volume charges for
the query window. Volume `9luykg5yc3` remains intentionally present and
continues accruing at `$7/month`; it is not part of the closed Pod total.

Scientific workers launched at approximately `19:10` BRT; the last condition
finished at `21:19:45` BRT, about 2 hours 10 minutes later. Retrieval, all local
hashes, cross-condition verification, final deletion, and the zero-Pod audit
finished at approximately `21:28` BRT. This landed at the early edge of the
revised `21:19--21:43` training ETC and about eight minutes later for the full
verified teardown. The original ETC error was confined to the timing model:
`step_wall_seconds` omitted batch construction and host-to-device staging. The
scientific configuration never changed.

Run 004 is now execution-complete and locally verified. The final validation
losses, diagnostics, and two requested terminal spillover figures remain
descriptive evidence until the user approves a consolidated scientific
statement; they are not yet a finding or manuscript claim. The figures were
generated from stored integer diagnostics, while the optional checkpoint
validation-loss trajectory remains deferred in the run README.
`artifacts/closeout.json` is the machine-readable final state.
