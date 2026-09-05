# GPU choice, $40 total ceiling, and execution playbook

Revised 2026-09-05. The user's $40 total cash budget supersedes the earlier
larger proposal. No GPU or external model API has been launched by this plan.

## Live GPU recommendation

Quotes were refreshed through RunPod MCP at 13:32:58 UTC on 2026-09-05; see
[the budget-revision snapshot](planning/gpu-market-snapshot-budget40.json).
The [earlier snapshot](planning/gpu-market-snapshot.json) is historical.
Capacity and actual Pod prices must be checked again before rental.

| GPU | VRAM | Community / h | Secure / h | Global stock | Role |
| --- | ---: | ---: | ---: | --- | --- |
| RTX 5090 | 32 GB | $0.69 | $0.99 | Medium | Recommended budget development GPU |
| RTX PRO 6000 Blackwell Server Edition | 96 GB | $1.69 | $2.09 | High | Shorter-search fallback |
| H100 SXM | 80 GB | $2.69 | $3.49 | High | Upstream positive control and frozen transfer |

The $40 revision makes batch-1 prefill the required workload. Loading one
410M model at a time in BF16 should leave substantial room on 32 GB for
logits and sparse workspaces, but measure the actual peak before searching.
Batch 32 is optional, not a reason to rent 96 GB by default. No precision or
model-size reduction is permitted to make a candidate fit.

RTX PRO costs 2.45 times as much per hour. It must deliver more than 2.45
times as many valid candidate evaluations per hour to beat RTX 5090 in GPU
dollars when both fit. Faster iteration or memory headroom could justify it,
but this has not been calibrated. We choose RTX 5090 provisionally to buy
more search time while retaining the two-hardware comparison.

RTX 5090 currently has low per-data-center availability in EUR-IS-1 and
EU-RO-1 despite medium global stock. RTX PRO has medium indications in
EUR-IS-2 and US-NC-2. H100 has high global but low per-location indications.
These are not reservations or guarantees of a particular cloud-tier price.

Use one GPU per Pod, no spot interruptions, and no concurrent benchmarks.
Do not create a new network volume or constrain placement to the old volume
before discovering a suitable GPU. An unavailable cheaper GPU is not reason
to incur unbounded retries.

## Budget allocation

| Phase, including its build/transfer overhead | RTX 5090 hours | H100 hours | Compute USD |
| --- | ---: | ---: | ---: |
| Calibration and upstream positive control | 2 | 1 | 4.07 |
| One feedback-driven search, at most 40 attempts | 20 | 0 | 13.80 |
| Frozen 36-checkpoint matrix, repeats, component ablations | 8 | 0 | 5.52 |
| Frozen H100 transfer on all 18 endpoint sentinels | 0 | 3 | 8.07 |
| Retrieval/rebuild/infrastructure allowance | 2 | 0 | 1.38 |
| Maximum planned compute | 32 | 4 | **32.84** |
| Storage, billing uncertainty, and cash reserve | - | - | **7.16** |
| Hard total ceiling | - | - | **40.00** |

The pilot is capped at $5 total and is included in $40, not additional.
Unused time is not an obligation to spend. All billed startup, compilation,
agent-response waits, failed attempts, transfers, restarts, and cleanup count
toward the per-GPU hours and global cash cap. Reserve final evaluation and
teardown before authorizing another candidate.

Expected use is 24-32 RTX-5090 hours plus 3-4 H100 hours, or $24.63-$32.84
compute; allow approximately $28-$40 total. A rough calendar ETC is 1-3 days
after local preparation, availability, and agent-session continuity. Initial
local preparation is estimated at 6-12 working hours, with no rented idle GPU.
These are uncalibrated engineering estimates. The time cap bounds search,
not the time needed to discover a successful kernel.

No independently paid agent API is planned. Use the existing authorized
agent session; confirm its incremental accounting before launch and retain
model/configuration and token-use records where available. If there are
additional model charges, allocate them INSIDE $40 by reducing search time
before execution. Do not launch a paid API with unspecified charges or treat
it as an excluded cost. Existing subscription fees are not a new purchase;
metered usage is not assumed free.

### RTX PRO fallback, not another experiment

If RTX 5090 is unavailable or the exact required workload fails the memory
gate, a fresh-start alternative is at most 16 RTX-PRO hours and 3 H100 hours:
$35.11 compute at the quoted community rates. Suggested RTX-PRO split:
2 hours calibration, 8 search, 5 final evidence, 1 transfer/cleanup.

This is NOT an extra $40 envelope. Subtract every earlier attempt's charges
and storage commitments before recomputing affordable fallback hours. If the
final evidence cannot fit, stop and return the partial result. Do not drop
models or quality coverage silently. At secure-cloud quotes the primary
32+4-hour allocation costs $45.64 in compute alone and is not permitted.

## Calibration and go/no-go gate

Use pinned official Sakana source, prepare the minimal Pythia integration
locally, and carry the known shape-correctness tests. Reproduce the upstream
control on H100 within its one-hour pilot allocation. On RTX 5090, start
with A0 14M correctness, then A1-H 14M, followed by the declared 70M/410M
shapes and representative A7 sparsity. Measure actual memory, transfer,
build times, complete-validation time, and at least three candidate-equivalent
timing suites after warmup. Confirm actual input rotation.

Before search, project the exact mandatory matrix: dense/P0/final on all
36 primary batch-1 checkpoints, 18-sentinel contribution ablations, repeated
timings, and dense/P0/final H100 transfer on 18 sentinels. Protect this cost.
If it needs more than its allocation, shorten the search before commencing.
If $40 cannot cover the mandatory evidence even without search, stop at the
pilot and explain the missing evidence; do not declare the study complete.

Forecast:
- remaining time = unfinished setup + candidate count times measured trial
  wall time + complete quality/timing suites + transfer/hash verification;
- remaining cost = each Pod's actual rate times remaining billable hours
  + storage through teardown + model charges if any + explicit reserve;
- affordable search time = remaining cash after protecting the final matrix,
  storage, and teardown, divided by the actual development-Pod hourly rate.

Track both the fixed search deadline and the projected valid-candidate count.
An ETC to the budget limit is not an ETC to a winning kernel. Query actual
account balance before launch and during monitoring. Spendable funds are the
lesser of the unused $40 allowance and available balance after other resource
commitments. Prior funding messages do not establish the current balance.

If Blackwell needs a materially different algorithm rather than a compatible
build of the Sakana starting point, document the issue at the pilot; do not
consume the search budget on an undisclosed from-scratch replacement.

## Provisioning and artifact inventory

After implementation, focused tests, the full bootstrap suite, and launch review:

1. Re-list resources, verify balance, current quote, cloud tier, and placement.
   Register/inject the SSH public key before creating the Pod.
2. Pin a CUDA-devel/PyTorch image digest compatible with the exact GPU and
   Sakana source. Prepare dependencies/build files locally; catalog CUDA
   availability is not evidence that the extension compiles.
3. Use a single-GPU run025 development Pod, with 80 GB container disk and
   150 GB Pod volume at /workspace. Record actual SKU, UUID, data center,
   software, rate, deadline, worker PID, and log paths per attempt.
4. Transfer a SHA-256 allowlist: U0/P0 source/patches/tests, model-only weights,
   configs/gate metadata, logical counts, tokenizer/cache identities, selected
   training development blocks, and complete validation tokens. Include
   official positive-control assets separately. Estimate 25-35 GiB for the
   complete supporting input inventory; calculate actual bytes and transfer
   ETC before launch. No credentials, optimizer states, or whole-worktree upload.
5. Keep model weights single-copy and load one checkpoint at a time. Store
   candidates/logs persistently and sync small completed-trial artifacts locally.
6. Retrieve all candidate sources, provenance, build/test failures, raw paired
   timings, quality/count records, profiles, dispatch tables, memory records,
   and cost/event logs. Verify hashes locally before Pod deletion.

Budget about $1.13 for 230 GB of running Pod disks across 36 aggregate hours
at $0.10/GB-month (730 hours/month). Stopped Pod volume retention costs more;
include it if the H100 waits between pilot and final transfer. The existing
100 GB network volume costs roughly $7/month; reserve its prorated charge
during execution as well, since it drains the same balance. Two days is
approximately $0.46. No new volume purchase is planned.

RunPod documents no ingress/egress fee, but GPU rental during transfer is
billable. Storage rates and rounding must be reconciled against actual billing:
[RunPod Pod pricing](https://docs.runpod.io/pods/pricing).

## Monitoring, stop conditions, and teardown

Use detached workers and durable events. Monitor at 30-minute intervals with
asynchronous PowerShell Start-Sleep; check sooner for a projected phase finish
or declared warning. Do not busy-poll or block the agent's tool for 30 minutes.

Report per Pod and in aggregate: phase/candidate, valid/failed attempts,
incumbent full-model score, latest validation loss/delta, trials per hour,
GPU memory/utilization, billed hours, accrued charges, remaining cost/ETC,
balance, unused $40 allowance, protected final-test reserve, and next todo.
This is inference; do not report fictitious training-loss progress.

Warnings: forecast exceeds remaining cash; stale worker events or timeouts;
numerical failures; insufficient disk/VRAM headroom; thermal/clock drift;
approaching budget deadline; or >20 minutes of unexplained GPU idle time.
For host-side development pauses, retrieve state and stop GPU billing.

Stop proposing candidates when the next trial would consume protected final
evidence or cleanup funds. A provider/controller backstop must act before
the cash ceiling, allowing billing lag and storage. Test stop-and-preserve
behavior for a retrieval failure; do not delete the sole copy of required
artifacts. Every restart/retry retains the same scientific inputs and counts
toward the same global budget.

Normal closeout: verify terminal state -> inventory -> copy -> local hashes
-> terminate Pod -> re-list Pods/endpoints/volumes. Confirm no unintended GPU
spending, report retained storage, and distinguish estimated from posted cost.
At this planning refresh, there are zero Pods and zero endpoints. The existing
network volume is unchanged; planning incurred no GPU rental charge.
