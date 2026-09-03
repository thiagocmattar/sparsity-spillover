# Run 021 - Pythia-410M A0 learning-rate screen with resolved worker LR

## Status

The scientific design is the already-approved Run 020 design. Run 021 exists
only because Run 020 failed before optimizer boundary 1: its inherited loop
read the intentionally null top-level learning rate instead of the
condition-resolved value. Run 020 remains unchanged and terminally failed.

Run 021 forwards a copied execution config containing the selected condition's
concrete peak/minimum learning rates into that same inherited loop. A regression
test invokes the actual `run_worker` dispatch for both worker IDs and captures
the config received at the inherited boundary. The remote preflight repeats the
same contract check before launch. No other scientific field differs from the
approved screen.

Run 021 is complete and valid. Both new arms completed all 712 optimizer
boundaries from the same pinned initialization and schedule, passed complete
validation and terminal verification, and were retrieved with matching archive
and file-level hashes. The predeclared training-only selector retained the Run
019 `3e-4` baseline. Both RunPod GPU Pods were deleted after retrieval; no new
TEAL evaluation was run because the selected baseline already has a complete,
verified ten-point Run 019 frontier. See [RESULTS.md](RESULTS.md).

## Question

Was Run 019's unexpectedly weak Pythia-410M A0 endpoint materially caused by
using the canonical 410M peak learning rate (`3e-4`) inside this unusually short
one-MiniPile-pass budget?

Run 021 compares the verified Run 019 `3e-4` A0 result with two new, independent
from-scratch A0 runs at peak learning rates `6e-4` and `1e-3`. The minimum LR is
always ten percent of peak. This is a small recipe-selection experiment, not a
model-scale or intervention ablation.

## Matched design

| Arm | Peak LR | Minimum LR | Execution |
| --- | ---: | ---: | --- |
| Run 019 baseline | `3e-4` | `3e-5` | Reuse pinned completed evidence |
| Run 021 arm 1 | `6e-4` | `6e-5` | New full pass from scratch |
| Run 021 arm 2 | `1e-3` | `1e-4` | New full pass from scratch |

The two new arms otherwise exactly match Run 019 A0:

- Pythia-410M architecture: 24 layers, width 1,024, FFN width 4,096, 16
  attention heads, 50,304 vocabulary entries, and 405,334,016 parameters;
- the same locally generated random-pretraining initialization, model seed
  1234, strict-load parameter SHA-256
  `76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`,
  and restored post-initialization CPU RNG state;
- the same seed-1234 complete-block permutation and schedule SHA-256
  `d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`;
- 712 optimizer boundaries, global batch 1,024, sequence length 2,048, and
  1,493,172,224 scheduled input tokens. This covers all 728,374 complete
  MiniPile training blocks and wraps 714 blocks to complete the final batch;
- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0.1 with bias/norm
  exclusions, gradient clipping at 1.0, dynamic FP16, FP32 parameters/state,
  zero dropout, one-percent warmup, and GPT-NeoX pre-step cosine semantics;
- A0/GeLU, no activation gate, no activation pressure, SDPA/flash training, and
  no activation checkpointing.

Released Pythia weights are never loaded. Each new arm starts from the same
hash-pinned random initialization; neither continues from the Run 019
checkpoint.

## Validation, diagnostics, and retained artifacts

Every new arm evaluates all 500 MiniPile validation documents: 338 complete
2,048-token blocks and 692,224 input tokens, with the 1,444-token tail excluded
and reported. The canonical endpoint pairs loss and `R_model` from the same
eager logical-product pass.

The run records every boundary's task loss, effective LR, dynamic loss-scale
state, global task-gradient L2 norm before and after clipping, clipping flag,
throughput, memory, overflow and skip status. Final eager diagnostics retain
integer-pooled exact/near-zero activation counts and RMS/L2 statistics at
`a,m,h,q_post,k_post,v,z,attention_output`, complete logical-product integers,
`R_block`, `R_model`, the analytic A0 reach ceiling, and all-parameter weight
statistics. A complete final recovery checkpoint retains model, optimizer,
loss scaler, schedule identity, and Python/NumPy/Torch CPU/CUDA RNG states.

Because A0 has no auxiliary pressure gradient, OL1 conflict and trust-budget
metrics are not applicable. The post-hoc TEAL protocol clips `a,m,h,z` at all
ten target sparsities from 0.0 through 0.9. It runs only for the selected new
arm; if the Run 019 baseline wins, its already-verified TEAL frontier is reused.

## Predeclared selection and stopping rule

Selection uses the mean task loss over optimizer boundaries 649--712. Validation
loss and `R_model` are reported afterward and cannot select the LR. An arm is
eligible only if all 712 boundary losses, LRs, and gradient norms are finite and
no update overflowed or was skipped.

The hash-verified Run 019 baseline has a fixed selection metric of
`4.4937172935`, final task loss `4.4483484887`, 56 clipped boundaries, paired
validation loss `4.5474564377`, and `R_model=0.0001104792`.

- If `3e-4` or `6e-4` has the lowest eligible training metric, select it.
- If `1e-3` is the stable best arm, stop with status `deferred_unbracketed` and
  propose one fresh A0 `1.5e-3` full pass in a new numbered run. Do not silently
  promote an optimum at the edge of the grid.
- If the baseline remains best or both higher-LR arms fail, the higher-LR
  explanation is refuted within this grid.

After a bracketed choice, its complete-validation behavior determines the
interpretation. A stable higher LR with lower training and validation loss
supports the LR-undertraining hypothesis. Lower training loss without a
commensurate validation improvement indicates optimization/generalization
tension. Little improvement while the curve remains descending points toward
the one-pass token budget rather than LR alone.

The other eleven 410M intervention runs are not part of Run 021 and will not be
launched until the A0 result is reviewed and the user approves the final recipe.

## Manuscript connection and limits

This run determines whether the 410M point can be used in the manuscript's
cross-scale `R_model`--validation-loss comparison under a defensible short-budget
recipe. It does not establish an optimal 410M learning rate, a compute-optimal
scaling law, or a multi-seed uncertainty estimate. It is one-seed recipe
selection on the operational MiniPile contract.

## Proposed execution envelope

Run 019 measured the exact A0 workload at 5.78 hours for training and complete
endpoint work on an RTX PRO 6000. Its full Secure Pod lifecycle, including
setup, TEAL, retrieval, and teardown, was 8.85 billable hours and $18.4931.
Run 021 proposes two isolated RTX PRO 6000 Pods in parallel, Community first
and Secure fallback, with A100 SXM, H100, and H200 allowed only as the approved
availability fallbacks. Automatic termination is 10 hours for RTX PRO, 12 for
A100, 10 for H100, and 8 for H200, matching the measured SKU-specific ETCs.

At the 2026-09-03 catalog snapshot, RTX PRO 6000 was $1.69/GPU-hour Community
and $2.09/GPU-hour Secure. Expected two-Pod cost is approximately $30--$37;
the 10-hour RTX operational guard is $33.80--$41.80. The largest currently
possible fallback envelope is two Secure H200s for eight hours, or $73.44,
plus small Pod disk charges. The final prelaunch refresh showed an $81.21 account
balance,
$0.01/hour current spend from the already-retained network volume, and no GPU
Pods or Serverless endpoints. Prices, capacity, and balance must be refreshed
immediately before launch.

The measured workload reserves about 22 GiB on RTX PRO 6000 and the full
Run 019 calibration reserved under 35 GiB, so the configured 80 GB minimum has
ample headroom. The exact workload does not fit the local machine with required
headroom; RunPod is the proposed execution lane.

Each Pod stores its checkout, input cache, log, attempt, and approximately
4.86 GB final checkpoint below `/workspace`. Training runs detached and writes
one event per boundary. Read-only monitoring occurs every 30 minutes, shortened
to the projected completion window when ETC falls below 30 minutes. Warnings
are any non-finite value, overflow/skip, stale event over 60 minutes, unexpected
initialization/schedule hash, disk pressure, or ETC approaching the applicable
per-SKU backstop.

Each attempt is verified remotely, packaged with an inventory, retrieved, and
SHA-256 checked locally before selection. If the result is unbracketed or the
Run 019 baseline wins, both new Pods are deleted immediately. If a new arm is
selected, its Pod is retained only long enough to receive the hash-checked
selection record and run the ten-point TEAL evaluation; the other Pod is
deleted immediately. The selected Pod is deleted after its TEAL artifacts are
retrieved and verified. This avoids a third Pod and a redundant checkpoint
upload.

An independent local RunPod CLI 2.12.0 guard is armed by exact Pod ID
immediately after creation and retries deletion at the applicable SKU deadline
if normal cleanup has not already completed. A final RunPod inventory must show
zero unintended GPU Pods or endpoints.

## Workflow

1. Complete local focused tests, strict initialization verification, shell
   syntax checks, and the full bootstrap suite.
2. Refresh RunPod inventory, capacity, prices, balance, and exact maximum cost.
3. Obtain explicit launch approval.
4. Create and preflight two Pods; verify source, caches, runtime, GPU, and two
   independent strict initialization loads before optimizer construction.
5. Launch `a0-lr-6e-4` and `a0-lr-1e-3` detached.
6. Monitor every 30 minutes with progress, loss, gradients, throughput, ETC,
   accrued cost, and required balance.
7. Verify, retrieve, and hash-check both attempts, then apply `05_select.py`
   locally while the deadline guards remain armed.
8. Delete both Pods if selection is unbracketed or the Run 019 baseline wins.
   Otherwise delete the non-selected Pod, transfer the selection record to the
   selected Pod, run and retrieve its ten-point TEAL frontier, and then delete
   it.
9. Confirm zero GPU Pods/endpoints and consolidate the A0 result before
   proposing any of the other eleven
   interventions.
