# Authorized execution envelope

On 3 October 2026, after design approval, the user explicitly asked to continue
while AFK through analysis/figures/tables/results, emphasized full authorization
to launch experiments on RunPod, and asked for responsible use of available
funding. No additional confirmation is required for this bounded run.

## Implementation and checks

The eight configs vary a/m unilateral versus bilateral gates, Ph versus Pall,
and kappa=0/.1. h/z stay unilateral. `training.py` adapts the frozen Run004
recipe trainer with declared pressure capture and timing that includes batch
assembly/staging. `optimizer_boundary.py` adapts the corrected Run015 OL1
boundary to audit Ph or Pall capture names on every microbatch. No shared
scientific code or historical run input is changed.

`diagnostics.py` adds pre-gate a/m moments/signs, all four gate rejection energy,
sequence-local row/8x16/16x16 occupancy, and actual learned LN affine values.
It retains existing full-validation post activations, weight norms, logical
products and integer analytic topology ceiling with an explicit unit. Training
checkpoints, dynamic scaler, task/pressure gradients and OL1 geometry follow
the approved inventory. Scientific completion requires zero skipped updates.

Local evidence: 22 focused tests passed, including threshold equality/gradients
(also FP16 rounded-cutoff equality),
six-layer hook placement after learned LN affine, all eight actual pressure
boundaries, wrong capture rejection, count-first tile/energy pooling,
checkpoint/topology round trips, optimizer/scaler/RNG serialization, cache
schedule identity and incomplete-validation rejection. Full bootstrap: 242
passed. Real14M CPU mechanics passed all eight conditions with the identical
approved initial parameter hash, T16/MB1/GAS1/FP32. These checks do not establish
production CUDA FP16 memory fit or ETC. Bash syntax and PowerShell AST parse
pass; CLI2.14.0 live `pod get/stop` and SSH help are verified.

The exact MB32/GAS32/T2048 workload historically used about57--62GiB reserved;
the local RTX5070Ti Laptop has12GB and cannot fit it. Historical Run041 H100
calibration measured five updates after one warmup, median3.381--3.393s/update,
about618--620k tokens/s and61.413GB reserved of85.017GB. Four scientific
conditions took2,426--2,430s each. The changed a/m gates, Pall and extra
diagnostics still require their own target preflight.

Each condition's remote preflight uses the exact architecture, precision,
microbatch/accumulation, cache order, pressure path and full validation. It
measures five complete boundaries after one warmup, checkpoint reload and all
final diagnostics. Require no overflow/skips, exact initial/schedule/capture
identities, and at least10% memory headroom. All eight must pass before any
fresh scientific condition starts. Median>6.8s/update triggers placement review.

## Hardware, price and deadline

Primary: one Secure Pod `run057-t4-gate-sign-001`, eight H100 SXM80GB, one
independent condition per GPU. Live catalog reads immediately before launch
show USD3.49/GPU-hour, eight-GPU LOW stock in EUR-NO-2 on hostCUDA13.0.
The pinned CUDA12.8 image/runtime remains compatible with that host driver.
H200 is USD4.59 and B200 USD6.79, with no eight-GPU Pod stock in the same reads.
One-GPU H100 stock is HIGH, H200 MEDIUM, B200 LOW. H100 has direct matched
14M throughput evidence; the choice avoids extra setup/cache copies while
running all conditions concurrently. Availability and actual fit win over the
GPU name. If an eight-GPU placement fails, up to eight independent H100 Pods
may run the same conditions within the same aggregate24GPU-hour/US$90 cap.
There is no automatic substitute to a differently priced GPU.

Fleet GPU rate USD27.92/hour. Maximum running duration is three hours from
creation, covering setup, upload, preflight, training, diagnostics and retrieval;
GPU maximum USD83.76. Running40GB container+50GB persistent disk storage adds
aboutUSD0.04 over three hours for one Pod, or belowUSD0.31 for eight Pods,
using a730-hour month estimate. Responsible incremental operational cap:
USD90, including bounded infrastructure retries within remaining time/cost.
Expected end-to-end1--2h (USD28--56 compute), not yet a new calibration.

Image `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`;
Python3.12, torch2.11.0+cu128, Transformers5.12.1; fixed setup dependency versions.
Request at least4vCPU/16GB CPU RAM per GPU. Store code, logs, caches and outputs
under `/workspace`. No new network volume. Existing100GB EUR-IS-1 shared
volume and the pre-existing EXITED Run052 Pod remain outside this run's scope.
The observed old Pod conflicts with historical deletion text and is not reused.

Storage pricing is USD0.10/GB/month while running; a stopped persistent volume
is USD0.20/GB/month. No provider ingress/egress fee. [Current RunPod pricing](https://docs.runpod.io/pods/pricing).
Deadline action is **stop and preserve the volume**, not deletion of unverified
evidence. A credential-free remote process guard stops the detached workload;
an independent hidden Windows guard performs provider stop. Credentials stay
on the workstation. GPU costs stop at the deadline; any stopped storage retained
for recovery is reported separately and removed after verified retrieval.

## Transfer inventory, ETC and monitoring

Upload the actual committed Git bundle (~155MB), the56,279,344-byte random
step-zero replay, and one immutable train/validation int32 cache plus metadata
(5,969,620,336 token-file bytes). Upload overlaps environment setup. Verify
every cache/replay hash remotely. Pinned HF rebuild is an unchanged-input
fallback only if transfer throughput makes it faster; verify the same hashes.
Sharing one cache removes seven redundant transfers and preparations.

Retain96 model checkpoints and24 recovery states across eight conditions,
including all final checkpoints; all manifests/configs/events, full-validation
metrics, sign/energy/occupancy/weight/logical diagnostics, preflight rows and
control logs. Historical payload projects~8.1GB, compressed~7.5GB; exact bytes
and SHA256 inventory will replace these estimates. Run041 transferred3.759GB
in191.453s (~19.6MB/s), an approximate six-minute download anchor for this run,
not a guaranteed network speed. Every archive and extracted member is checked
locally, followed by per-attempt/cohort scientific verification before deletion.

Monitor every five minutes, idle between checks. Updates report steps, latest
finite task/pressure losses, throughput, memory/disk, and ETC including remaining
diagnostics/transfer. Check at the projected completion window when sooner.
Warnings: nonfinite losses/diagnostics, overflow/skipped updates, changed
identities/capture names, reserved memory>90%, disk>85%, no new event for ten
minutes, sustained throughput below half the matched H100 anchor, or projected
completion outside the remaining envelope. Diagnostics phases are identified
explicitly so a training-event gap is not mistaken for a stalled optimizer.

On completion: seal immutable evidence, copy and verify all artifacts locally,
terminate newly created Pod(s), audit Pods/volumes and refresh posted billing.
Then generate paired quality/structural tables and PDF figures with observation
notes. Claims remain one-seed joint a/m evidence, without measured speedup or
automatic manuscript edits. Tile pressure stays on the backlog.
