# Run 007 - Local Pythia-14M joint A4-Z threshold plus all-site OL1

## Status

Design and local launch approved on 2026-08-29. Implementation, tests, exact
calibration, and the locked launch packet are complete. Scientific execution
will use that immutable packet.

## Question and hypothesis

This run asks whether fixed strong OL1 pressure combined with every condition
in Run 006's joint A4-Z one-sided-threshold sweep improves the model-wide
quality--logical-opportunity frontier. Run 006 is the threshold-only matched
comparator at each `kappa`; Run 005 is a secondary `h`-only OL1 comparator.

OL1 may move surviving post-threshold activations toward the gate boundary and
increase exact zeros and `R_model` relative to Run 006 while its global conflict
projection and trust budget limit task interference. The hypothesis is
supported by a new nondominated condition with higher logical opportunity and
no material validation-loss penalty. It is weakened or refuted by negligible
additional opportunity, quality degradation, instability, or a vanishing
pressure gradient after aggressive thresholding.

## Conditions and intervention

Five serial conditions use topology `A4-Z = {a,m,h,z}` in all six layers:

| Order | Gate threshold | Pressure |
| ---: | ---: | --- |
| 1 | `kappa=0` | post-threshold OL1, `lambda=1`, budget `1` |
| 2 | `kappa=0.01` | post-threshold OL1, `lambda=1`, budget `1` |
| 3 | `kappa=0.05` | post-threshold OL1, `lambda=1`, budget `1` |
| 4 | `kappa=0.1` | post-threshold OL1, `lambda=1`, budget `1` |
| 5 | `kappa=0.5` | post-threshold OL1, `lambda=1`, budget `1` |

At every selected site, the gate keeps `x` when `x >= kappa` and otherwise
returns exact zero. Equality survives; surviving inputs receive identity input
gradients and rejected inputs receive zero input gradients.

OL1 pressure targets the gated outputs at all four sites in every layer. Its
objective is the unweighted mean of 24 tensor-level mean absolute activations:
`a`, `m`, `h`, and `z` across six layers. Thus each site-layer tensor has equal
weight even though `h` has a different width. Pressure on pre-gate latent
values is not part of this run.

The task gradient is globally clipped to norm `1.0` and alone drives AdamW and
its moments. The pressure gradient is preconditioned with AdamW's task second
moment, globally projected only under conflict, capped at a weighted
pressure-to-task direction ratio of `1.0`, and applied after the task step.

## Matched training and data

Everything other than OL1 matches Run 006: pinned randomly initialized
Pythia-14M, no released weights, model seed 0, data-order seed 0, exact training
schedule hash, MiniPile cache identities, sequence length 2,048, global batch
64 as microbatch 4 by accumulation 16, BF16 CUDA with FP32 parameters and
optimizer state, zero dropout, and peak LR `1e-3`.

Each condition executes 581 updates and 76,152,832 input tokens. The cohort
totals are 2,905 updates and 380,764,160 input tokens. AdamW uses betas
`(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, one-percent linear warmup,
and cosine decay to ten percent of peak.

Every validation pass covers all 500 MiniPile validation documents: 338
complete blocks and 692,224 input tokens, with the 1,444-token tail reported
and excluded. Validation runs after update 1 and from the reloaded final
checkpoint; the independent logical-product pass covers the same workload.

## Diagnostics and retention

Every optimizer boundary records task and pressure losses, clipped task and
pressure gradient norms, dot/cosine/conflict, OL1 adaptive-direction norms,
projection, raw and final correction ratios, trust scale, LR, throughput,
wall time, and peak GPU memory. These interaction metrics cannot be recovered
from a checkpoint.

Final checkpoints record count-first exact-zero and near-zero counts at `0`,
`1e-3`, and `1e-2`, plus RMS/L2 moments for `a`, `m`, `h`, `q_post`, `k_post`,
`v`, `z`, and post-`W_o` attention output. All named parameter norms and all six
actual-operand logical-product counters are retained. There is no clipping
frontier. All five final checkpoints are saved without optimizer state.

`R_block`, `R_model`, and the A4-Z `R_model_max = 12.833152%` ceiling are
logical-product opportunities, not removed FLOPs or measured speedup.

## Manuscript relationship and interpretation limits

The run tests the proposed combination of architecture-wide thresholding and
conflict-aware pressure. Pressure at signed attention/branch sites as well as
`h` is broader than the manuscript introduction's strategic-FFN-pressure
wording; this is an explicitly approved experimental variant. The previously
approved operational A4-Z topology includes pre-`W_o` site `z`, which remains
absent from the current manuscript topology table.

The design has one seed, one small model, and a short local horizon. Joint
gating and joint pressure do not attribute effects to individual sites. At high
thresholds, pressure is computed only on surviving values and may have little
gradient support. Logical opportunities do not imply sparse-kernel speedup.

## Approval record

- Design approved by the user on 2026-08-29: all five Run 006 thresholds,
  post-threshold OL1 pressure at all four `{a,m,h,z}` sites, `lambda=1`, trust
  budget `1`, exact Run 006 token budget, and a 70-minute local envelope.
- Local launch explicitly approved by the user on 2026-08-29 at
  `2026-08-29T22:28:05.1761202Z`.

## Calibrated local fit and ETC

The non-evidence pre-launch calibration exercised both endpoint conditions
(`kappa=0` and `kappa=0.5`) for eight exact optimizer boundaries each, timing
six after warmup. Each endpoint also completed ordinary full validation,
count-first activation diagnostics, eager logical-product diagnostics, all
parameter statistics, and a checkpoint hash/reload round trip.

The resulting five-condition ETC is 3,788.1 seconds (63m08s) at the median and
3,893.7 seconds (64m54s) at p90. The p90 estimate leaves 306.3 seconds (5m06s)
below the locked 4,200-second ceiling. Median optimizer-boundary time was
1.2542 seconds, equivalent to 104,506 training input tokens/second. Peak Torch
allocation was 7,209,447,424 bytes and peak reservation was 8,315,207,680 bytes
on the 12,820,480,000-byte RTX 5070 Ti Laptop GPU. After the measured 2,252 MiB
desktop baseline, the conservative residual headroom was 2,143,879,168 bytes.

Calibration task losses and sparsity values are infrastructure smoke results,
not scientific evidence. Both endpoint correction ratios respected the OL1
trust budget. The calibration artifact and locked execution definition are in
`prelaunch/`; 41 focused tests and all 108 bootstrap tests pass, and both
PowerShell scripts parse. No `attempts/` directory exists.

On launch, the five conditions run serially in a detached local process. Each
retains its final checkpoint (about 56.3 MB; about 281.4 MB total), full
validation and diagnostic artifacts, boundary event stream, logs, hashes, and
verification result; optimizer states and predictions are not retained. There
is no cloud resource, bill, storage, transfer, or teardown step.

Monitoring uses one read-only check after each 1,800-second PowerShell
`Start-Sleep`, with updates at launch, about 30 minutes, about 60 minutes, and
completion. Every update reports progress, task loss, throughput, and refreshed
ETC. Warning conditions include non-finite metrics, OL1 ratio above budget,
less than 1 GiB conservative GPU headroom or OOM, a stale event stream,
throughput more than 25% below calibration, ETC above 4,200 seconds, early exit,
traceback, or identity/coverage/artifact mismatch.

## Execution log

- `2026-08-29T22:28:33.821767Z`: launched the approved immutable packet as
  detached local process 13892. The launcher recorded Git, config, run-code,
  schedule, and calibration provenance before starting the first condition.
- The first process completed `kappa=0` but stopped at `kappa=0.01`, boundary
  145, after a CUDA memory-allocation error. The completed and failed attempts
  remain immutable. Infrastructure retry 001 reuses the locked condition code,
  retains the failed attempt, and restarts only conditions without a completed
  attempt; its rationale and controller are recorded under
  `artifacts/infrastructure-retry-001/`.
- Infrastructure retry 001 completed replacement attempts through `kappa=0.1`
  but stopped with another CUDA allocation failure at `kappa=0.5`, boundary
  356. Four conditions have unique completed attempts. The run is paused before
  a final-condition retry because its projected combined active compute exceeds
  the approved 70-minute envelope.
- The user subsequently approved an 80-minute combined active-compute envelope
  and one unchanged restart of only `kappa=0.5`, followed by terminal
  verification.
- That restart created attempt 007 but stopped with a CUDA allocation failure
  at boundary 33. Cumulative active attempt time is 63m00.8s. Four conditions
  remain complete, `kappa=0.5` remains incomplete, and terminal verification
  has not run. A further retry requires explicit authorization.
