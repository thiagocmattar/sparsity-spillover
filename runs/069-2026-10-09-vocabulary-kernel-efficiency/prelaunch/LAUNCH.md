# Run069 launch packet

Status: ready for explicit launch approval. No Pod has been created.
Design approval: user's **Go on** after the fixed-checkpoint diagnostic proposal.
Retention inventory: user confirmed **Use this inventory** on 2026-10-09.

## Implemented and verified

Eight bounded variants: original N128/N256/N512; direct stores N256; prefetch
N128/N256; independent two-row-tile weight reuse N128/N256. Each has no-skip and
sparse modes. Native PyTorch/cuBLAS and original N256 remain controls. No training,
threshold, pressure, weight, output contract or quality change is introduced.

242 bootstrap tests and 7 focused mathematical/artifact tests pass. Final local
attempt 002 passes 160 CUDA checks covering column tails, full vocabulary,
random/live/dead/opposite-neighbor masks, graph replay and counted MMA paths.
Both trained checkpoints pass two-block full-model smokes, with exact native
logits/loss and no rejected modes. Full 338-block qualification remains pending.

The RTX 5070 Ti laptop has 12 GiB VRAM and adequate headroom: screen peak is
4.171875 GiB; model smoke peak is 3.041016 GiB. WSL Torch 2.11.0/cu128,
Transformers 5.12.1 and NumPy 2.5.0 match the proposed runtime. Local screening
uses two inputs/two passes; its ranking does not determine the RTX 5090 shortlist.

Local Nsight Compute 2025.1.1 failed with `ERR_NVGPUCTRPERM`. Cloud access will be
checked separately. If counters are denied or missing, retain the failure and
continue matched timings/ablations, reporting the hardware bottleneck as unresolved.
No global laptop permission or driver setting is changed.

## Exact execution

One Secure **NVIDIA GeForce RTX 5090**, 32 GB VRAM, preserving the hardware family
of the existing endpoint evidence. All final controls/candidates share one physical
GPU; capture, screen, endpoint and available profiler UUIDs are reconciled.

Live catalog: **USD 1.19/hour**, availability **LOW**, host CUDA 13.0 available.
Userland CUDA 12.8 is pinned. Recheck price/stock immediately before creation.
Do not exceed the proposed envelope or substitute hardware without reporting it.

- Pod name: `run069-vocabulary-efficiency-001`.
- Image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
- Container disk: 30 GB. Persistent Pod volume: 40 GB at `/workspace`.
- SSH `22/tcp`; no network volume or endpoint. Existing registered keys are used.
- Foreign Run052 Pod `qa00sm05lmrs7w` and volume `9luykg5yc3` remain untouched.

Deploy the sealed input archive under `/workspace/run069`, record the reviewed
Git commit as `RUN069_SOURCE_COMMIT`, and verify every runtime/cache/model file
against `provenance/upload-inventory.json`. Install pinned packages with
`00_setup.sh`, recording environment/profiler identity. Launch `07_execute.sh`
detached, with PID, logs and control receipts on persistent storage.

Stages: component checks; capture 64 matched real inputs per checkpoint; paired
projection screen of all variants; separate Nsight profiles on four inputs;
three fresh final processes per checkpoint; verification and process reduction.
Each final process qualifies all 338 blocks and measures three overlapping
boundaries: projection, gate/cast/projection and full forward. Final timing uses
64 inputs x 7 randomized paired passes, matching Run068. The body is native dense.
This does not establish a new global quality-latency frontier or restore trained loss.

## ETC, cost and monitoring

Planning ETC: **25-60 minutes**, including setup, transfer, compilation, profiling
and qualification, with provisioning/profiler uncertainty. This is an estimate,
not a cutoff. Local seven-mode qualification runs at 2.7-2.9 blocks/s, implying
roughly 12-13 minutes for six complete passes on the laptop before other stages.
Refresh ETC from cloud progress. Use a **flexible four-hour maximum from Pod
creation**, reserving the final 30 minutes for retrieval and verification.

Compute ceiling at the quoted rate: 4 h x USD 1.19 = **USD 4.76**. Running storage
is 70 GB x USD 0.10/GB/month, approximately **USD 0.04** over four hours. RunPod
documents no ingress/egress charge. Planned maximum compute plus running storage
is about **USD 4.80**; the proposed service-cost budget is **USD 6**, allowing
setup/failure margin within the same four-hour envelope. Existing resources are
excluded. Sources: [RunPod pricing](https://docs.runpod.io/pods/pricing) and
[live inventory](live-gpu.json). Re-query actual creation price.

Immediately after creation, arm `12_local_stop_guard.ps1` in a hidden workstation
process, with the exact Pod ID, creation timestamp and deadline no later than
creation + 4 hours. Save the PID and `ARMED` receipt; do not begin remote execution
if the guard fails. It verifies the exact run name before any provider stop, uses
the authenticated CLI, and stops compute rather than deleting unverified artifacts.
Cancel only after normal retrieval/hash verification and confirmed Pod deletion.

This guard depends on workstation uptime and API reachability; it is not a
provider-native scheduled cutoff. If the backstop fires, recover artifacts promptly
and delete the Pod after verification. Stopped 40 GB storage costs USD 0.20/GB/month
(about USD 0.011/hour) until deletion, so the compute stop alone is not an absolute
total-cost guarantee. Report any failed stop or recovery delay explicitly.

Monitor every 60 seconds: stage/progress, latest qualification loss, throughput
and refreshed ETC. Check earlier for nonfinite/numerical failure, disk pressure,
deadline risk or previously declared stale-event warnings. Setup/compile warning:
no progress for 10 minutes. Runtime warning: stale events for 5 minutes, excluding
bounded 180-second profiler calls. End further work early enough to retrieve and
verify available evidence. Partial evidence cannot pass the full-cohort verifier.

## Retention and teardown

The compressed input archive is about 160 MB; [bundle.json](bundle.json) records
its final exact size/hash. It includes frozen runtime and retained checkpoint/cache
identities. Existing checkpoints remain locally retained.

Retain exact pre-gate inputs, tokens/indices and head weights; raw timing JSONL;
numerical checks/losses; screen failures/selections; real-operand support/MMA counts;
compiler register/spill logs; environment/GPU/source identities; Nsight reports or
access failures; comparison tables and transfer inventories. Binary snapshots stay
outside Git. Run068 retains existing training diagnostics and final checkpoints.
Estimated return payload is 0.3-1 GB, dominated by snapshots/profiler reports.

Copy all agreed files, verify byte counts/SHA256 against the remote inventory,
then delete the owned Pod and re-list account Pods/volumes. Report remaining
resources and posted/provisional charges. Never delete foreign resources.

Approval reason: [AGENTS.md](../../../AGENTS.md) explicitly requires separate
launch confirmation: **Design approval is not launch approval.** The current
confirmation accepts artifact retention; the concrete cloud launch awaits approval.
