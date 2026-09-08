# Run 031: signed activation distributions for Figure 05-v3

Status: completed and verified on 8 September 2026 after explicit design and
launch approval. All seven checkpoints passed full validation and signed-count
verification. The Pod was deleted after local hash verification; zero Pods
remain and the existing network volume is retained. The local GPU was unused.

The [complete signed data](results/README.md),
[Figure 05-v3](../../analyses/018-2026-09-08-results-materials/figures/05-v3-activation-density-grid.pdf)
and [observation/caption](../../analyses/018-2026-09-08-results-materials/observations/O011-activation-density-v3.md)
are available. Original Figure 05 and Figure 05-v2 are unchanged.

## Results and closeout

All 2,366 blocks completed in 69.94 seconds after environment setup. Maximum
absolute validation-loss discrepancy from retained eager results is 0.0000688543,
below 0.0005. Integer partitions, pooled counts and 108 empty-gate-region checks
pass. The three-checkpoint CUDA calibration used at most 1.78 GB reserved memory;
all six remote focused tests and both CUDA dtype boundary controls passed.

At kappa=.5, pooled FFN exact-zero fractions are 99.31% for A4-OL1 and 93.64%
for A7-OL1. Attention exact-zero fractions are 0.22% and 95.60%, respectively:
the figure separates A4's sharp central nonzero peak from A7's zero point mass
and symmetric threshold gap. These are descriptive complete-recipe comparisons.

All 28 transferred files match their remote SHA-256 values and pass local
verification. Pod `57n9792o4ns8ss` and its deadline guard are removed. Confirmed
Pod-free time was 19:45:30 UTC; estimated GPU cost is USD0.206, within the USD2
cap. Posted billing had no records yet. Setup on the network volume dominates
the roughly 17-minute Pod lifetime. See `transfer-receipt.json`, `closeout.json`
and `results/scientific-verification.json`. No manuscript or finding was changed.

## Question and approved scope

How do A0, A4-OL1 and A7-OL1 reshape their signed activation distributions?
Use the seven Pythia-14M step-712 checkpoints from Analysis 018 Figure 05:
A0 from Run 004; A4-OL1 from Run 015 and A7-OL1 from Run 014, each at
kappa=0,.05,.5. Exact source attempts, hashes, training identities and original
activation summaries are in `input-manifest.json`. Its inherited `evaluate`
field belongs to Run 030's reuse catalog; this run evaluates all seven entries.

The approved [design](../../analyses/018-2026-09-08-results-materials/ACTIVATION-DENSITY-V3.md)
connects this descriptive diagnostic to the manuscript's distribution case
study. No training, optimizer, gradient pass or additional clipping is used.
Source seed 1234, trained gate configurations and weights remain fixed. A4
uses one-sided gates at a,m,h,z; A7 additionally uses symmetric gates at
post-RoPE q,k and v. The original OL1 pressure is inactive at evaluation.

Evaluate FP32 parameters under FP16 CUDA autocast, eager uncached attention,
B=1 and T=2048. Use all 500 MiniPile validation documents: 338 complete blocks,
692,224 input tokens and the excluded 1,444-token tail per checkpoint. This
is 2,366 block evaluations in total. Recompute loss and the original activation
moments/counts, recording discrepancies; a loss difference above 5e-4 fails
verification and prevents use in the figure.

## Measurements and intended figure

Capture post-gate signed h,m,q_post,k_post,v, retaining all 30 layer/site rows
per checkpoint. Accumulate integer histogram counts over [-8,8] at nominal
.001 spacing using stored float32 edges. Retain underflow/overflow, extrema,
dtypes, exact-zero and near-zero counts, finite/nonfinite counts and moments.
Exclude exact zeros from the continuous histogram and preserve their point
mass separately. Reject nonfinite activations or any count partition mismatch.
FP16 values near a nominal threshold can fall on an adjacent float32 bin;
do not infer gate leakage from a bin straddling that threshold.

Pool counts and totals before dividing: FFN h,m have 80%/20% element weights;
attention q,k,v have equal weights. Density is count/(total * actual bin width),
so its integral equals the nonzero mass in range. It is not renormalized to
one. Retain per-site counts to audit effects hidden by pooling.

Analysis 018 will own `figures/05-v3-activation-density-grid.pdf`: three kappa
rows, two activation-group columns, signed x, density y, colored outlines and
low-opacity fills. Show exact-zero mass separately, +kappa for FFN, and +/-kappa
for A7 attention. Choose the displayed range from measured tail coverage.
Keep the original and v2 PDFs. The observation will disclose pooling, finite
precision, omitted tails and the descriptive comparison of complete recipes.
No manuscript text or finding is promoted by this task.

## Implemented files and verification

- `01_prepare.py`: verify and package seven checkpoints, validation tokens,
  source modules and this run's scripts; exclude training data and optimizer state.
- `density.py`, `02_evaluate.py`: signed counts, pooled densities and immutable
  evaluation attempts, coverage, loss reconciliation and durable progress logs.
- `03_verify.py`: check input hashes, completed coverage, site counts and exact
  pooled reductions; produce per-attempt transfer inventories.
- `04_setup.sh`, `05_launch.sh`: pinned environment and detached CUDA smoke/full
  execution, with separate attempt directories and a 30-minute process timeout.
- `06_stop_guard.ps1`: independently stop this named Pod at the compute deadline,
  preserving its volume. The process timeout alone does not stop GPU billing.

CPU verification passed 248 tests (242 bootstrap + 6 focused) in 8.87 seconds.
Tests cover signed boundaries in FP16/FP32, tails, zero exclusion, streaming
pooling, nonfinite/count failures, serialization and actual NeoX post-gate hooks.
The collector also completed three real-checkpoint CPU FP32 smokes, one full
2,048-token block each for A0, A4-OL1(.5), A7-OL1(.5), and the independent
artifact verifier passed. These partial CPU measurements are excluded from
scientific results and are not CUDA performance estimates. The local GPU was
hidden with `CUDA_VISIBLE_DEVICES=''` throughout.

## Approved RunPod launch record

Live MCP quotes at 19:22 UTC on 8 September 2026: one Secure RTX PRO 4500
Blackwell, 32 GB, USD0.72/hour; if unavailable, one Secure RTX 5090, 32 GB,
USD0.99/hour. Both advertise low availability in EUR-IS-1. No existing Pods
were found. Reuse the existing 100 GB network volume `9luykg5yc3`; create no
new persistent volume. Pod name `run031-activation-density-001`, 20 GB container
disk, SSH only, `/workspace` on that volume. Pin the previously exercised image
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
The exact digest in the launch receipt must match the verified Run 030 image.

Use Python 3.12, torch 2.11.0+cu128, transformers 5.12.1 and the pinned packages
in `requirements.txt`. Work under `/workspace/run031`; preserve unrelated volume
content. Verify the bundle inventory before execution, record environment and
GPU details, and rerun the focused tests remotely.

First evaluate eight validation blocks for each of A0/A4(.5)/A7(.5), with the
exact FP16 CUDA histogram workload. Record peak reserved memory and per-model
elapsed time. Proceed to the seven full passes only if counters are valid,
memory has at least 20% headroom and the refreshed full-evaluation estimate
plus transfer allowance fits the remaining approved hour. A 14M checkpoint
has about 56 MB of weights; 32 GB is expected to provide ample inference
headroom, but CUDA histogram timing and peak memory are not yet measured.

Provisional total ETC: 15-30 minutes including setup, upload, calibration,
evaluation and verified retrieval. Prior Run 030 measured about 12 seconds per
14M validation pass on RTX 5090, but signed histograms add uncalibrated work;
this historical timing is not the ETC for the new collector.

Approved maximum: one hour of GPU time, deadline from Pod creation, with a
USD2 total incremental spending cap including container storage and contingency.
At the quoted rates, one hour costs at most USD0.99 GPU plus approximately
USD0.003 container storage. The already-retained network volume continues
at about USD7/month and is not a new resource for this run. Storage rates are
from [RunPod storage documentation](https://docs.runpod.io/pods/storage/types).
Arm the scoped stop guard in a hidden PowerShell process immediately after
creation. Normal teardown occurs sooner, after verified retrieval; if interrupted,
the deadline stops compute and leaves artifacts on the existing volume.

Monitor at 60-second intervals, or sooner near projected completion. Report
completed checkpoints/blocks, current validation loss, tokens/second, refreshed
ETC, GPU memory and disk. Investigate stale progress (>120 seconds), nonfinite
values, identity or loss failures, CUDA errors, and a projected deadline overrun.
No automatic budget extension or additional worker is proposed.

## Transfer and retention

Upload only the explicit `bundle-inventory.json` entries: seven checkpoint
directories, validation tokens, source modules and run scripts. The bundle
receipt records exact size/SHA-256; checkpoints and tokens are ignored by Git.
Retain histograms, source/cache/code identity, coverage/loss, exact/near-zero
counts, RMS/L2, dtypes, timing, extrema and tails. Existing checkpoint files and
weight diagnostics remain retained. Gradient interaction cannot be reconstructed
here and remains in the training records; logical/clipping measurements already
exist and are not repeated by this diagnostic.

Retrieve the three-checkpoint CUDA smoke, seven complete histogram artifacts,
terminal manifests, events, environment/GPU records and logs. Generate remote
transfer inventories, copy outputs locally, then verify all SHA-256 values and
scientific invariants. Only then delete the Pod, stop its local guard, and
re-list Pods/volumes. Retain the existing network volume. Commit compact signed
measurement artifacts with their provenance when the completed run is verified;
never commit checkpoints, datasets, secrets or files still being written.

Local commands (CPU-only preparation/verification):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
.venv/Scripts/python.exe -m pytest tests runs/031-2026-09-08-signed-activation-density/test_density.py -q
.venv/Scripts/python.exe runs/031-2026-09-08-signed-activation-density/01_prepare.py --bundle
.venv/Scripts/python.exe runs/031-2026-09-08-signed-activation-density/03_verify.py --inputs
```
