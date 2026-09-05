# GPU choice, bounded costs, and execution playbook

## Live planning snapshot and recommendation

The RunPod MCP catalog was queried on 2026-09-05 at 13:11:22 UTC. Its response
is preserved in [gpu-market-snapshot.json](planning/gpu-market-snapshot.json).
Catalog availability is not a reservation; placement/cloud tier must be
re-queried immediately before rental.

| GPU | VRAM | Community quote / h | Secure quote / h | Global stock | Role |
| --- | ---: | ---: | ---: | --- | --- |
| RTX PRO 6000 Blackwell Server Edition | 96 GB | $1.69 | $2.09 | High | Recommended development device |
| H100 SXM | 80 GB | $2.69 | $3.49 | High | Hopper compatibility and transfer |
| H100 NVL | 94 GB | $2.59 | $3.19 | Low | H100 alternative; distinct SKU, re-baseline |
| RTX 5090 | 32 GB | $0.69 | $0.99 | Medium | Cheaper conditional option, memory uncalibrated |

RTX PRO provides workspace headroom for batch-32 410M and fast compile/test
turnaround without H100's hourly premium. This is an engineering preference,
not a measured efficiency result. RTX 5090 is 2.45 times cheaper per hour:
if both fit, RTX PRO must complete more than 2.45 times as many valid trials
per hour to be cheaper in GPU dollars. Reduced OOM/rebuild risk and a usable
full-model memory envelope may still justify RTX PRO, but must be measured.

Current placement hints: RTX PRO has medium availability in US-NC-2 and
EUR-IS-2, but low in the existing volume's EUR-IS-1. Do not force that volume's
placement or create a new volume before finding a suitable GPU. H100's high
global stock coexists with low per-data-center indications; allow placement
flexibility and do not promise immediate acquisition.

Use one GPU per Pod and no spot interruptions or multi-tenant benchmark jobs.
Normally rent only the development Pod; rent the H100 only for its bounded
stages. There is no need for 12 concurrent training GPUs: this is inference
engineering and one candidate is evaluated at a time.

## Hours, ETC, and money

At the quoted community prices:

| Recommended allocation | RTX PRO hours | H100 hours | Compute USD |
| --- | ---: | ---: | ---: |
| Baseline/calibration gate | 4 | 2 | 12.14 |
| Three closed-loop trajectories, up to 6 h each | 18 | 0 | 30.42 |
| Three matched no-performance-feedback trajectories | 18 | 0 | 30.42 |
| Frozen final matrix, contribution modes, verification/transfer | 6 | 0 | 10.14 |
| H100 portable transfer/configuration retune | 0 | 4 | 10.76 |
| H100 final confirmation, verification/transfer | 0 | 2 | 5.38 |
| Unallocated RTX PRO contingency | 2 | 0 | 3.38 |
| Maximum allocated total | 48 | 8 | 102.64 |

The calibration gate has a $15 all-in RunPod cap and is **included** in the
recommended full-study $115 cap. Expected compute use is 36-48 RTX PRO hours
and 6-8 H100 hours ($76.98-$102.64); reserve storage/operational margin to give
a planning range of $85-$115. Final confirmation takes priority over spending
the last trial hour. Stop candidate generation early if the measured final
matrix needs that time. Never omit checkpoints to fit an optimistic forecast.

At secure-cloud quotes, 48 RTX PRO hours plus 8 H100 hours would cost $128.24
in compute alone. Therefore secure placement does not fit the same $115 scope;
revise hours or the explicitly stated ceiling before using it. H100 NVL changes
both price and hardware identity; record and re-baseline that choice.

These are search caps, not calibrated completion estimates. A result can be
"no correct kernel beat dense within budget." No finite time allowance can
guarantee a successful kernel discovery. Approximately 2-4 calendar days is a
working estimate after implementation, dependent on provisioning, model-call
latency, compile times, and transfers. Allow 6-12 working hours initially for
local implementation; an attention compatibility problem can change this.

The smaller alternative (18-24 RTX PRO hours + 4 H100 hours; roughly $45-$60
including margin) is a single exploratory case study without the replicated
feedback comparator. It cannot support the stronger agent-workflow argument.

Agent-model charges are **excluded** from these GPU estimates. Before launch,
record the exact route/model, call and input/output-token caps, and its priced
maximum if separately billed. GPU idle time awaiting model responses is
already billable and included in trajectory wall time. Do not describe $115
as the total project cost if an additional paid model endpoint is introduced.

## Calibration decision and refreshed ETC

Measure the exact three shapes, two batches, BF16 path, gates, and diagnostics.
Collect at least three repeated candidate-equivalent timing suites after
warmup and complete validation timings for each size; report their range.
Include failed-build rate and occupancy/packing cost. Separate setup from
recurring work and include both in the financial forecast.

The forecast is transparent arithmetic, not the CUDA kernel time alone:

```text
remaining_seconds = build/setup still required
                  + remaining candidate equivalents * measured trial wall time
                  + remaining quality evaluations * measured full-validation time
                  + remaining timing suites * measured suite time
                  + measured/provisioned transfer and verification allowance
remaining_cost = sum(per-Pod hourly rate * remaining billable hours)
               + storage through verified teardown + explicit reserve
```

For a time-capped search, also report the fixed upper bound and the projected
number of valid candidates within it. The ETC to reach the budget limit is
not an ETC to discover a winning kernel. Reconcile the final matrix separately
from the search budget before phase C starts.

Use live account balance before launch and each monitoring report. Required
additional balance is `max(0, remaining forecast + reserve - available balance)`,
including other retained-resource charges. Account balance was not checked in
this planning turn. Do not infer it from the historical funding messages.

If compilation/validation is too slow, input rotation fails, numerical tests
fail, VRAM headroom is inadequate, or attention cannot compete with fused SDPA,
stop at the calibration gate and present the measured limitation. Candidate
selection within the confirmed search space is autonomous; expanding that
space, changing precision/workload, or exceeding the envelope is not.

## Provisioning and transfer inventory

After implementation/tests and the launch review:

1. Re-list Pods/endpoints/volumes; check balance and current GPU price/capacity.
   Prefer a reusable owned compatible Pod only if it is not doing other work.
2. Pin a CUDA devel/PyTorch image digest and toolchain compatible with the
   selected GPU. Resolve the actual Blackwell/Hopper build targets and upstream
   dependencies locally where possible; catalog CUDA support is not proof that
   the old extension builds. Verify the SSH public key before creation.
3. Propose a single-GPU `run025-kernel-dev` Pod, with 80 GB container disk and
   150 GB Pod volume at `/workspace`. Use a corresponding short-lived transfer
   Pod for H100. Record actual data center, cloud tier, SKU, UUID, rate, start
   time, maximum duration, and guard deadline in each attempt directory.
4. Upload a hash-checked allowlist: source commit/patch/tests, model-only final
   checkpoints, architecture/gate metadata, canonical count records, pinned
   tokenizer/cache identities, selected development blocks, and complete
   validation cache. Include the separately identified official positive-control
   assets if needed. Do not upload optimizer states, secrets, or the entire
   working tree. Estimate 25-35 GiB including weights and supporting assets;
   compute actual bytes and transfer ETC before launch. Load one model at a
   time and avoid duplicating weights across candidate directories.
5. Test detached logging, worker timeout, process isolation, and GPU-budget
   guard with a short smoke. Store all logs/candidates under `/workspace` and
   synchronize small result artifacts locally after each completed candidate.
6. Keep model source checkpoints locally. No new trained model is produced;
   retrieve candidate source/patches, agent logs with credentials removed,
   package/toolchain manifests, tests, raw paired samples, full validation and
   count records, profiler summaries/traces, dispatch tables, memory records,
   event/cost logs, and the final SHA-256 inventory.

The estimated Pod-disk charge for 80+150 GB over 56 aggregate Pod-hours is
about $1.77 at $0.10/GB-month using 730 hours/month; actual billing may differ.
Stopped Pod volume storage is charged at a higher rate and must be included
if a Pod waits for retrieval. RunPod documents no ingress/egress transfer fee,
but rental time during transfers is billable. See
[RunPod Pod pricing](https://docs.runpod.io/pods/pricing).

At the planning check, there were zero Pods/endpoints and one pre-existing
100 GB standard network volume, `sparsity-spillover-shared`, in EUR-IS-1.
Leave it unchanged. Its roughly $7/month baseline storage is not a new Run-025
GPU expense and continues unless the user separately chooses to remove it.

## Monitoring, warning conditions, and teardown

Use persistent logs/events and detached workers. During an active search,
monitor at 30-minute intervals using an asynchronous PowerShell `Start-Sleep`
on the local controller; do not busy-poll unchanged state. Check earlier at
the projected phase completion or a declared warning. The controller must
remain able to report while waiting; do not block the agent tool for 30 minutes.

Each update reports per Pod and in aggregate: phase/trajectory/candidate,
completed/valid trials, incumbent full-model score, latest quality delta,
candidate throughput, GPU memory/utilization, elapsed billed hours/cost,
refreshed ETC, remaining cost, balance shortfall, and next todo. There is no
training loss curve in an inference study; report validation loss/delta rather
than inventing training progress.

Warnings: projected balance below remaining cost plus reserve; no new worker
event beyond its declared timeout; disk/VRAM headroom below the measured safe
margin; numerical failure; sustained clock/thermal deviation; impending
deadline; and GPU idle for >20 minutes without a scheduled validation/build.
For host-side development waits, retrieve state and stop GPU billing instead
of keeping an unused accelerator running. Restart/reconnect details belong
in a new infrastructure attempt directory, with unchanged scientific inputs.

Reserve time for retrieval before each deadline. The backstop must stop GPU
billing without deleting the sole copy of unverified artifacts: use a tested
stop-and-preserve action if retrieval failed, then retrieve from retained
storage and delete the Pod. Report any storage that continues billing. Normal
completion remains: terminal status -> inventory -> copy -> local SHA-256
verification -> terminate Pod -> re-list resources. Do not delete a Pod while
required artifacts exist only on its volume.

At closeout, report final per-Pod and total cost, no unintended active GPU or
endpoint, and the existing volume's continuing charge. Provider billing may
lag: distinguish the teardown estimate from subsequently posted charges.
