# Run066 launch packet — awaiting launch approval

The user approved the four-condition design and the post-hoc inventory on
8 October 2026. This packet requests launch approval; no Pod has been created
and no scientific training has started. See [README](README.md) and
[resolved recipe](config.yaml) for the scientific contract.

## Implemented scope and verification

Four independently initialized 14M conditions add one 8x16 corner gate after
the learned affine final LayerNorm, before the dense vocabulary projection.
Each uses matched h/z/head kappa 0 or .1. Pressure is either six post-gate h
tensors or those six plus the head input, equally averaged (six/seven terms).
No weights or logits are sparsified. Each trains 712 updates from the identical
original random parameter bytes and MiniPile block order.

Retained Run060 same-kappa endpoints are copied, hash-verified and re-evaluated
without a head gate or further training. Their input provenance is in
[controls inventory](inputs/controls/inventory.json). `17_controls.py` uses an
identity head tap and the same full validation and head diagnostics.

Runtime is adapted within this run from Run060, with source provenance in
[derived sources](prelaunch/derived-sources.json). `head_site.py` owns placement,
capture and checkpoint restoration; `optimizer_boundary.py` checks exact capture
sets. `diagnostics.py` adds head moments, occupancy/corner/energy, native dtypes,
final LayerNorm affine values and separately named head logical opportunities.
`03_verify.py` reconciles all four attempts and both retained controls.

Focused tests: **39 passed**. Bootstrap suite: **242 passed**. Real 14M CPU
smoke: **four passed**, original initialization SHA matched, one complete OL1
boundary each, exact checkpoint logits and exact causal prefix logits. Tests
exercise threshold equality and signed interiors in FP32/FP16/BF16, actual
projection operands, six/seven tensor means and gradient scope, faulty capture
rejection before an optimizer update, checkpoint/recovery state and pooled
element/row/tile/logical counts. Receipts: [checks](prelaunch/checks.json),
[CPU smoke](prelaunch/local-smoke.json).

These CPU checks do not establish CUDA memory fit, production FP16 numerics or
new training throughput. GPU preflights are mandatory before fresh training.
All four conditions each run six exact MB32/GAS32/T2048 updates; retain five
post-warmup end-to-end timings, full validation, full diagnostics, peak memory,
and checkpoint hashes/topology. A B1/T2048 production-FP16 roundtrip must match
logits and loss exactly. Preflight updates are discarded; scientific training
starts from the original initialization only after all four preflights pass.

Reject source/cache/order/initialization drift, skipped or overflow updates,
nonfinite values, failed coverage/corner checks, roundtrip differences, reserved
memory above 90% or median complete update above 6.8 seconds (twice the prior
H100 timing). No silent batch, gate, pressure or tolerance changes.

## Resource, live price and flexible envelope

Local GPU: RTX 5070 Ti Laptop, 12,227 MiB. Run060's exact MB32/GAS32 production
workload reserved 61,467,525,120 bytes (57.24 GiB), versus 85,017,493,504 bytes
on H100 (72.3%). The local GPU cannot fit the proposed decomposition.

The [live catalog and account inventory](prelaunch/catalog-and-inventory.json)
were read on 8 October 2026. Secure two-GPU H100 SXM hosts report **LOW**
availability in AP-IN-1, AP-JP-1 and US-GA-2. Four-GPU H100/H200 hosts report
**NONE**. Two-GPU H200 hosts report **LOW** in EUR-IS-4. Prices are USD3.49
per H100 GPU-hour and USD4.59 per H200 GPU-hour. A100 SXM80GB is USD1.59 but
historical repository throughput is about 5.7 seconds per update, versus 3.38
on H100, making H100 preferable for time to an answer.

Propose **one Secure on-demand Pod**, `run066-vocabulary-tile-001`, **two H100
SXM80GB**, with two successive waves of two conditions. The first wave uses
Ph, the second Ph+head. Conditions keep independent RNGs, optimizer states,
checkpoints and logs. Preflights also use two waves, then an all-four barrier.
Two control evaluations run concurrently after training. No distributed
training or condition-specific batch change is involved.

Use the pinned image digest in config, Python3.12/Torch2.11.0+cu128/
Transformers5.12.1, 40GB container disk and a 50GB Pod volume at `/workspace`.
Use current registered SSH credentials; two public keys are registered.
No network volume is created or attached. Prefer US-GA-2 for the H100 placement
if allocatable, otherwise another eligible center. If the two-H100 offer fails
to allocate, this packet includes a two-H200 fallback under the same recipe,
runtime, duration and total ceiling. Do not choose a different GPU or add Pods.
Refresh capacity and the actual offer immediately before provisioning.

Estimated end-to-end completion: **2–3 hours**, with uncertainty. Run060
measured five full post-warmup updates per condition: medians 3.382154/3.382143
seconds, ranges 3.379180–3.390408 / 3.380588–3.388119. Two 712-update waves
therefore give an 80.3-minute training baseline. Allow approximately 20–30%
uncertainty for the new head/pressure path, 15–30 minutes for setup/cache rebuild,
and 10–30 minutes for preflights, diagnostics, checkpoints, transfer and
verification. New GPU preflights will replace this extrapolation; H200 speed
for this workload is unmeasured. The ceiling remains a flexible backstop.

**Maximum running duration: four hours from provider start. Maximum incremental
total: USD60**, including retries only within this aggregate envelope. At the
current quotes, four hours costs USD27.92 GPU time on two H100s, or USD36.72 on
two H200s. Running disks total 90GB at USD.10/GB/month, about USD.05 for four
hours using a 730-hour month. Expected H100 compute is USD14–21 for 2–3 hours.
Do not create an offer whose full billable envelope exceeds USD60. No paid
resource is needed for the preparation or approval wait.

[RunPod's pricing documentation](https://docs.runpod.io/pods/pricing) states
per-second Pod compute/storage billing and no ingress/egress fees. The retained
50GB stopped Pod volume costs USD10/month, about USD.0137/hour, until deletion.
If the backstop fires, stop and preserve unverified artifacts for recovery;
report this continuing storage charge and retrieve promptly. The four-hour
limit is not a scheduled early termination at the nominal ETC.

## Persistence, transfer and final products

All checkout, caches, events and artifacts live under `/workspace`. A detached
pipeline survives terminal loss. A credential-free remote workload deadline
guard stops its process group; a separate hidden workstation guard performs
provider stop at the same deadline. Record provider-start time, exact quote,
Pod identity, absolute deadline and guard PIDs before training.

Upload a bundle of the actual committed Git revision (historically about
160–200MB), 56.3MB original random replay, and **112,563,120 bytes** of retained
control checkpoint inputs. Check source/replay/control SHA256 remotely. Rebuild
the pinned HF MiniPile/tokenizer cache once on the Pod, matching train/validation
identities and the approved schedule SHA. A source bundle receipt is prepared
after the implementation commit; this file is not a claim that upload occurred.

Retrieve all four resolved configs/manifests/events and complete loss trajectory;
74 activation-statistics rows per pass, 13 gate energy/corner rows, 25 occupancy
rows, 13 normalization affine records; full head/block diagnostics at steps
0/32/128/712; training gradient-conflict/OL1 metrics; final named weight norms;
block and separately labeled head logical counts; control metrics and matching
diagnostics; runtime, preflight, pipeline and guard logs.

Retain **48 model checkpoints** plus **12 optimizer/scaler/RNG recovery states**
(the recovery states accompany checkpoints256/512/712), about 4.05GB raw across
four conditions from the parent inventory, roughly 3.7–4.1GB compressed plus
diagnostics/logs. Retain both controls and the untrained replay locally. No
clipping sweep is approved; exact source/cache/checkpoint identities support
later checkpoint-based diagnostics. Gradient interactions are recorded now.

Primary head sparsity/tiles and head logical opportunities come from the
production SDPA activation pass at validation batch32. Legacy block logical
enumeration uses eager attention at batch1; its extra head observation is
separately labeled. Different attention/rounding paths must not be mixed.
Actual pre/post gate dtypes are recorded; native gate-output zeros are not a
claim about every extra zero introduced by internal matmul dtype conversion.
The original block-only R_model numerator and HZ analytic ceiling stay named
with their old scope. Logical opportunities do not establish runtime speedup.

After completion: run the scientific verifier remotely; seal terminal artifacts
with byte/SHA256 inventories; retrieve the archive resumably; verify archive
and every member locally; run the cohort verifier locally. Only then terminate
the owned Pod and confirm absence. Re-list Pods/volumes and reconcile posted
billing. Never terminate before the agreed evidence is verified locally.
Existing EXITED Run052 Pod `qa00sm05lmrs7w` and shared100GB volume
`9luykg5yc3` are outside this launch and remain untouched.

Produce a six-row comparison (two controls and four new conditions), loss/
perplexity deltas, native head exact/near zeros, removed energy and empty tiles,
with PDF figures and observation Markdown beside this run. No latency/speedup
conclusion or manuscript edit is included. No numerical acceptable-loss bound
was approved; present the measured trade-off for review.

## Monitoring and approval

Read-only status checks every five minutes, with idle waits between checks.
Report setup/preflight/control stage or training wave, active step, current task
loss, tokens/s, GPU memory/utilization, disk and refreshed ETC including queued
conditions and transfer. Use the projected completion window when sooner.
Warn for stale progress beyond ten minutes, any skipped/overflow update,
nonfinite values, memory above90%, free disk below10GB, process/coverage/hash
failure, or refreshed ETC entering the final30 minutes of the running deadline.
No unapproved scientific rescue changes; preserve failure evidence.

The user has accepted the diagnostic inventory. **Launch approval remains
pending**. [AGENTS.md](../../AGENTS.md) requires a separate launch confirmation:
“Design approval is not launch approval.” Approval of this packet authorizes
the exact workload and resource/cost envelope above.
