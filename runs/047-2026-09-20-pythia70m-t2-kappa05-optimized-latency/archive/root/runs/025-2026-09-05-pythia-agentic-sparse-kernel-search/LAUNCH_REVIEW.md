# Calibration implementation and launch review

Status: design confirmed; CPU preparation implemented and verified; **not
launched**. This document defines the calibration gate only. It does not
declare a winning kernel, finish the search evaluator, or authorize the
remaining $35 after calibration.

## Implemented files

- `00_prepare.py`: inventory all 36 retained model-only checkpoints and their
  canonical counts; verify train/validation hashes; extract 64 training blocks
  with seed 2500. No new training or random initialization.
- `p0.py`, `kernels/twell_pythia.cu`: documented exact Sakana TwELL descendant
  and four-site adapter. `SOURCE_AUDIT.md` states compatibility changes.
- `measurement.py`, `01_calibrate.py`: non-default-stream primitive checks;
  rotating paired B=1,T=2048 full-logit timing; complete 338-block validation;
  fixed logit/loss gates; dense operator profile; separate BF16 activation
  diagnostics; durable samples/progress/errors and memory/throughput records.
- `02_package.py`: explicit input/source allowlist, per-file hashes,
  optional exclusive archive creation, and remote verification.
- `00_setup_remote.sh`, `03_start_calibration.sh`: pinned CUDA environment;
  detached worker; independent GNU timeout including child compiler processes.
- `05_upstream_control.sh`: unmodified official H100 supported positive control
  in its separate compatible environment, not historical Pythia raw ELL.
- `06_billing_guard.ps1`: separate LOCAL Pod stop guard, no Pod credentials;
  retained disk permits recovery before deletion. It must run on an awake
  machine. The worker timeout is NOT a billing guard.
- `trial_record.py`: immutable proposal/result files, source snapshots and
  evaluator hash checks. CPU dummy record/reload/complete tests exist; this is
  not an unattended model-API controller or a completed search evaluator.

## Exact local verification

Executed with Python 3.12.10, torch 2.11.0+cpu, transformers 5.12.1:

1. Full checkpoint/cache preparation: PASS, 36 checkpoints; 18 development
   endpoints / 18 untuned interior thresholds. `prelaunch/input_manifest.json`.
2. Focused tests: 23 PASS. Signed/all-zero/dense/ragged packing, N=128 and
   other widths, bias, changing inputs/workspace reuse, all four canonical
   topologies with mixed A7 gates, native restoration, paired input rotation,
   numerical rejection, shifted-loss coverage, transfer exclusions, immutable
   trial records, evaluator-tamper detection and record reload.
3. Full bootstrap: **242 PASS** after additions. Earlier test-only clock
   resolution failures were repaired: deterministic timer clock in the unit
   test; validation progress safely reports null throughput if elapsed=0.
4. Retained 14M A0/A1-H/A4-OL1(k=.5)/A7-OL1(k=.5), B=1,T=8 CPU smoke:
   adapter-dense and restored-native outputs bit-identical; mathematical
   backend logit gates pass. `prelaunch/local-smoke.json`.
5. Bash syntax checked for all three shell scripts. PowerShell guard parsed
   and dry-run executed with no API calls. No live stop/SSH-disconnect test
   has run; it belongs to the pilot readiness checks.

CUDA compilation, compute-sanitizer, GPU numerical gates, peak VRAM, real
latency and full-validation ETC are **not yet measured**. Local GPU is a 12 GB
RTX 5070 Ti Laptop; the repository env is CPU-only and no local GPU job is
proposed. Batch-1 on a 32 GB rented GPU should fit one 410M checkpoint with
headroom (BF16 model ~0.81 GB, plus transforms/workspaces/full logits and
comparison temporaries), but this is an estimate, not a measured fit.

## Sequential pilot envelope

Live control-plane snapshot on 2026-09-05: balance about $42.64, zero Pods,
zero endpoints. Existing 100 GB volume `9luykg5yc3` remains untouched. Quotes:

| Stage | GPU / live community rate | Lease maximum, including setup and transfers | GPU maximum |
| --- | --- | ---: | ---: |
| First, portable P0 calibration | RTX 5090 32 GB / $0.69 per hour (low stock) | 2 hours | $1.38 |
| Second, U0 and portability | H100 SXM 80 GB / $2.69 per hour | 1 hour | $2.69 |
| Both | Sequential, no idle second Pod | 3 GPU-hours | $4.07 |

**Combined pilot ceiling $5, included within the $40 total.** Reserve $0.93
for disks, prorated existing storage, price uncertainty, and teardown. This
is a stop budget, not a promise that every calibration condition completes.
At most eight endpoint checkpoints are in the portable pilot: 14M A0/A1-H;
70M A0/A1-H/A7(.5); 410M A0/A1-H/A7(.5). If the budget/time gate is hit, save
partial measurements and report incomplete coverage. Never call it a pass.

Start only the RTX 5090 first. H100 follows after transfer/setup readiness and
a remaining-budget check. A scarce or more expensive GPU is not silently
substituted at the full planned duration. RTX PRO fallback still requires
recalculating remaining hours inside the same $40; no second study budget.

Proposed Pod: one GPU, COMMUNITY, public TCP SSH only (`22/tcp`), official
`runpod-torch-v280` image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
CUDA >=12.8, 40 GB container disk + 80 GB persistent Pod volume at `/workspace`.
Pinned image manifest digest from Docker Hub:
`sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Use it in the lease record and image selection before creation. Local Pod storage
avoids assuming a community GPU can attach the existing DC-specific volume.
Expected disk allocation is ample for the 5.83 GB pilot input plus two venvs,
compiler caches and logs. Do not put evidence on disposable container storage.

Published running disk rates imply about $0.05 for 120 GB across three hours;
the pre-existing 100 GB volume adds about $0.03 during those hours (30-day
proration). Stopped disk remains billable until deletion. Sources:
[RunPod storage rates and persistence](https://docs.runpod.io/pods/storage/types).
Actual lease/account billing wins over these estimates; recheck immediately
before creation. Full 32h RTX5090 +4h H100 is still $32.84 compute before
storage, reserve and any incremental agent charges. No paid external API is
configured by these scripts.

## ETC interpretation

Preparation is local and already complete. Budget 10-25 minutes for the first
5.83 GB transfer plus installs/compilation, then up to 90 minutes of portable
calibration with a 10-minute retrieval margin. At 100 Mbps, the payload alone
takes about 7.8 minutes; at 20 Mbps about 39 minutes. These are transfer-rate
scenarios, not measured throughput. If setup is slow, reduce worker seconds
from the remaining lease time; do not extend the two-hour lease automatically.

The harness measures full-validation paired seconds/block and full-logit
timing at each size. Use those measurements to project all 36 final cases,
three implementations, repeated processes, diagnostics, ablations, H100
transfer and rebuild/setup costs. Protect the final evidence budget before
any search. No defensible per-candidate/full-study ETC exists before this gate.

## Execution playbook (after launch approval)

1. Re-query balance, Pods/endpoints, GPU price/stock; verify two registered SSH
   keys and working local private-key access without revealing keys. Record
   exact pinned image digest, rate, creation UTC and absolute stop deadline locally.
   Check direct TCP SSH is available. Do not launch to diagnose missing auth.
2. Build/reverify `prelaunch/run025-calibration.tar`; transfer only the explicit
   allowlist (~5.83 GB). No credentials, optimizer states, full training cache,
   or untuned-checkpoint weights. Prefer direct SSH/SCP; do not start a relay
   or publish a URL as an unapproved workaround. Archive/file hashes must match
   after remote extraction and before checkpoint loading.
   Inventories describe exact payload bytes. If reconstructing files from
   a fresh Git checkout instead of this archive, regenerate and verify the
   transfer inventory first; do not bypass a line-ending hash mismatch.
3. Create one named `run025-*` Pod with the stated resources. Immediately arm
   the independent hidden PowerShell stop guard at creation+2h (H100 +1h).
   Verify it is alive. CLI 2.12.0 live help does NOT expose the terminate/stop
   deadline flags from the skill example; do not assume they were installed.
   Read-only inspection and timed worker shutdown remain separate controls.
4. Run `00_setup_remote.sh` detached under `timeout 1500s`, log under
   `artifacts/runtime`. Install only into the declared persistent env. Confirm
   Python/torch/CUDA versions, nvcc, and a real GPU operation. Bootstrap a
   primitive gate under compute-sanitizer before the whole-model loop:
   `compute-sanitizer --tool memcheck --error-exitcode 99 <venv-python> 01_calibrate.py --primitive-only --attempt memcheck-001 --seconds 900`
   (also apply an outer `timeout --kill-after=30s 900s` to handle hangs).
   This includes the historical M=256,K=512,N=128 regression.
5. Start `03_start_calibration.sh <venv-python> <new-attempt> <seconds>` with
   seconds=min(5400, remaining-lease-seconds-600). Verify the worker survives
   SSH disconnect and log/status files advance. Its GNU timeout kills a hung
   process group. A CUDA/build failure is recorded, not hidden by retries.
6. Monitor every 10 minutes, or the projected completion window if sooner.
   Check sooner for compile >12 minutes, no progress >10 minutes, numerical
   failure, nonfinite output, OOM, transfer/checksum failure, or projected
   cost exceeding the reserved pilot envelope. Report condition/block progress,
   current validation loss (N/A before validation), tokens/sec, ETC, elapsed
   billable cost and reserve required. No training loss/gradient run is implied.
7. Copy finalized JSON/JSONL, profiles, errors, pip/nvcc/nvidia metadata and
   exact sources after each condition and at worker termination. Hash-verify
   locally. Do not copy an actively rewritten `status.json` as a final artifact.
   Original model weights already remain local; no new weights are produced.
8. Stop early at completion. Delete only after local hash verification. If the
   stop guard fires, disk persists for recovery; retrieve before deleting.
   Confirm zero unintended Pods/endpoints and reconcile actual spend. Keep
   the pre-existing network volume unchanged. Guard/API/network failures can
   overrun estimates: keep the control machine awake and respond to warnings.
9. On H100, reproduce U0 with `05_upstream_control.sh <new-attempt>` under a
   remaining-time timeout; record its official CSV and source/model manifests.
   Use the small `02_package.py --phase primitive --build-tar` payload and
   `00_setup_remote.sh primitive` on H100; it contains no Pythia weights.
   Portable compatibility runs with the Pythia interpreter:
   `01_calibrate.py --primitive-only --attempt h100-p0-001 --seconds 900`.
   Do not launch a second full eight-model suite
   blindly into the one-hour H100 envelope.
10. Present the calibration measurements and full-study cost projection. Seal
    P0/dense/evaluator only after corrections and CUDA verification. Implement
    the stronger compiled/graph dense comparison and search scoring before
    K001, then preserve the untouched final-test split.

## Post-hoc inventory and remaining scope

Approved diagnostic minimum: exact/near-zero counts and activation RMS/L2 at
a/m/h/z/q_post/k_post/v; weight norms; canonical logical counts/analytic ceiling
and workload identity; gate thresholds; full-logit error; complete validation
loss; timing raw samples; hardware/software/compile/profile and source history.
Calibration counts are train-split BF16 diagnostics, distinct from canonical
validation counts. Complete frozen-model counts and packing/attention/component
performance attribution still need the final evaluator, not an inference from
this pilot. Gradient conflict/OL1 boundary metrics are N/A to inference and
remain in source training artifacts. All original final checkpoints are retained.

Before renting, ask whether any additional post-hoc measurement is needed and
request explicit pilot launch approval. The confirmed design authorizes this
implementation, not an unreviewed full search or new agent API expenditure.
