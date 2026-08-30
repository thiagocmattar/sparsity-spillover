# Run 005 - Local Pythia-14M ReLU OL1 dose response

## Status

Completed and terminally verified on 2026-08-29. All five conditions reached
581 updates and 76,152,832 training input tokens. Evidence is valid with the
detached-process Git-provenance limitation described under `Results`.

## Question and hypothesis

This run asks how the strength of conflict-aware orthogonal L1 pressure at the
ReLU FFN hidden activation `h` changes targeted activation sparsity, untargeted
activation geometry, logical zero-product opportunity, and language-model loss
over a matched short local pretraining horizon.

A dose-ordered increase in exact or near-zero mass at `h`, accompanied by a
bounded validation-loss cost, supports the intended intervention. Little
movement at `h`, severe quality degradation, or trust-budget saturation that
makes nominally different weights operationally indistinguishable limits or
refutes that interpretation. Responses at `m`, `q_post`, `k_post`, `v`, and the
post-`W_o` attention output are descriptive spillover evidence, not proof of a
causal route.

## Conditions and matched design

The five serial conditions are a ReLU/no-pressure control followed by ReLU OL1
at `h` with `lambda` in `{0.01, 0.1, 0.5, 1.0}`. All conditions use topology
`A1-H`; standard ReLU replaces GELU directly at `h` in every layer. OL1 uses a
fixed relative trust budget of `1.0`.

Everything else is matched: pinned Pythia-14M architecture and MiniPile caches,
random initialization, seed-0 parameter identity, seed-0 realized block order,
2,048-token sequences, batch decomposition, optimizer, LR schedule, validation
coverage, diagnostics, and final-checkpoint retention. Released Pythia weights
are never loaded.

## Optimizer and operational OL1 definition

Peak LR is `1e-3`. AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay
`0.1`, a global task-gradient clip of `1.0`, one-percent linear warmup, and
cosine decay to ten percent of peak. Parameters and optimizer state are FP32;
CUDA forward/backward uses BF16 autocast; dropout is zero.

At every accumulated boundary OL1 computes task and pressure gradients
separately. The clipped task gradient alone drives AdamW and its moments. The
pressure gradient is preconditioned by AdamW's task second moment; a globally
conflicting component is removed; the weighted correction-to-task direction
ratio is capped at `1.0`; and the correction is applied after AdamW. Pressure
is the unweighted mean across the six per-layer mean-absolute `h` tensors.

The manuscript describes the trust budget as optional, while the operational
bootstrap requires the approved positive value. The operational clipped task
gradient drives both AdamW and its adaptive task direction.

## Budget, data, and full validation

The locked decomposition is global batch 64 as microbatch 4 by accumulation
16. The production-shaped control/OL1 calibration selected 581 common updates,
76,152,832 training input tokens per condition and 380,764,160 across the five
conditions. Its conservative p90 inclusive ETC is 3,297.05 seconds (54m57.0s),
inside the 3,300-second planning envelope and the user-approved 3,600-second
hard ETC ceiling. The estimate includes cache verification, setup, training,
validation, diagnostics, checkpoint save/reload/hash, and terminal headroom.

Every validation pass covers all 500 MiniPile validation documents: 338
complete blocks and 692,224 input tokens, with the 1,444-token tail reported
and excluded. Validation runs after update 1 and from the reloaded final
checkpoint. The logical diagnostic independently covers the same complete
workload using eager uncached attention.

## Diagnostics and retention

Every optimizer boundary records task and pressure losses, raw gradient norms,
dot/cosine/conflict, task-gradient clipping, OL1 adaptive direction norms,
projection, raw and final correction ratios, trust scale, LR, throughput, wall
time, and peak GPU memory.

The reloaded final checkpoint records count-first exact-zero and near-zero
counts at `1e-3` and `1e-2`, RMS/L2 moments for `h`, `m`, `q_post`, `k_post`,
`v`, and attention output immediately after `W_o`; all named parameter norms;
and the six actual-operand logical-product counters. `R_block`, `R_model`, and
the integer-count `A1-H` `R_model_max` are logical opportunities, not removed
FLOPs or measured speedup. No clipping frontier is included.

Every final model checkpoint is retained without optimizer state. This is
expected to require about 56 MB per condition and preserves checkpoint-
reconstructible diagnostics. Training-time gradient interaction cannot be
reconstructed and is therefore collected now.

## Manuscript relationship and limits

The run exercises the methodology's OL1 conflict projection, adaptive
directions, trust ratio, ReLU `h` pressure, spillover measurements, and
model-wide logical-product definitions. It does not compare OL1 against naive
L1, establish an architecture-wide topology advantage, measure kernel speedup,
or support a general scaling claim. One seed and a sub-hour cohort remain
descriptive.

## Approval record

- Design approved by the user on 2026-08-29: pinned Pythia-14M, ReLU, OL1 only
  at `h`, lambda grid `{0.01, 0.1, 0.5, 1.0}`, trust budget `1.0`, peak LR
  `1e-3`, matched ReLU/no-pressure control, logical-product counters, and a
  maximum 60-minute local ETC.
- Launch approved by the user on 2026-08-29 for the locked 581-step, five-
  condition, local p90 54m57.0s envelope in `prelaunch/launch-plan.json`.

## Implementation and verification

Run-local code now validates the five-condition matrix and immutable inputs,
executes serial attempts, logs every optimizer boundary, reloads each retained
checkpoint, performs complete activation and eager logical-product validation,
checks all artifact identities, and publishes a terminal cross-condition
verification. The launcher uses a hidden detached Windows process with durable
stdout/stderr logs; the bounded monitor reports condition state, latest attempt
event, event age, process state, and stderr tail.

Focused Run 005/OL1/logical tests pass 28/28. The full bootstrap suite passes
90/90. Both PowerShell scripts parse without errors. The non-evidence
calibration exercised eight exact boundaries for the control and `lambda=1`
OL1 paths, excluding two warmup timing samples per class. Losses and all OL1
metrics were finite; complete activation and logical coverage, 76 parameter
rows, checkpoint save/hash/reload, and the `A1-H` ceiling identity passed.

The repository `.venv` currently contains CPU-only Torch `2.11.0+cpu`. The
known local CUDA interpreter used for calibration and proposed launch is Python
3.12 with Torch `2.11.0+cu128`, CUDA runtime 12.8, and Transformers 5.12.1 at
`C:/Users/thima/AppData/Local/Programs/Python/Python312/python.exe`.

## Calibrated local launch packet

The calibration file is `prelaunch/calibration-20260829-185146.json`; the exact
execution definition is `prelaunch/launch-plan.json`. At 581 common updates the
median inclusive ETC is 3,244.16 seconds (54m04.2s) and p90 is 3,297.05 seconds
(54m57.0s). The final seed-0 schedule consumes 37,184 distinct complete blocks
without wrap and has SHA-256
`c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4`.

Calibration peak Torch memory was 7,119,201,792 bytes allocated and
7,763,656,704 bytes reserved on the 12,820,480,000-byte RTX 5070 Ti Laptop GPU.
With 3,130 MiB used by the desktop after calibration, conservative headroom is
about 1.65 GiB. Avoid other GPU-heavy work during execution.

Five final checkpoints are estimated at 281,402,180 bytes total, plus small
events and diagnostics. No optimizer states, cloud resources, transfer, or
billable storage are involved. Monitoring is every 60 seconds, warning on
non-finite values, OL1 budget violations, OOM/low headroom, a stale event stream,
more than 25 percent throughput degradation, ETC above 3,600 seconds, early
process exit, stderr errors, or identity/coverage mismatches.

## Where we stopped

The scientific cohort and append-only terminal verification are complete. The
local GPU process exited and released its allocation; no cloud or other
billable resources were created.

## Results

The five serial conditions completed 2,905 optimizer updates and 380,764,160
training input tokens in 3,284.44 seconds (54m44.4s), inside the approved
3,600-second ceiling. Median step throughput ranged from 112,188 to 147,963
tokens/s. All 15 complete validation passes covered 338 blocks and 692,224
input tokens, excluding the declared 1,444-token tail.

| Condition | Final validation loss | `h` exact zero | `h` near zero (`1e-3`) | `R_block` | `R_model` | Conflict rate | Projection rate | Trust saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 5.479057 | 74.166% | 74.253% | 10.592% | 3.173% | n/a | n/a | n/a |
| OL1 `lambda=0.01` | 5.448441 | 76.898% | 76.983% | 10.982% | 3.289% | 44.923% | 68.330% | 0.000% |
| OL1 `lambda=0.1` | 5.370173 | 84.315% | 84.381% | 12.042% | 3.607% | 52.496% | 95.009% | 0.172% |
| OL1 `lambda=0.5` | 5.324622 | 90.620% | 90.669% | 12.942% | 3.876% | 58.003% | 100.000% | 0.172% |
| OL1 `lambda=1.0` | 5.325123 | 93.588% | 93.624% | 13.366% | 4.003% | 62.651% | 100.000% | 3.442% |

Within this one-seed short horizon, stronger OL1 pressure produced a monotonic
increase in targeted `h` exact-zero mass and logical opportunity through
`lambda=1.0`. Validation loss improved through `lambda=0.5` and was effectively
flat from `0.5` to `1.0` at the reported precision. These are descriptive
results, not a general quality or scaling claim. `R_block` and `R_model` remain
logical-product opportunities rather than measured speedups.

Five final model checkpoints were retained and content-hash verified
(281,402,180 bytes total). The terminal verification is recorded at
`artifacts/verification.json` with evidence label
`valid_with_provenance_limitation`. The original verifier deliberately rejected
the null `git_commit` and `git_dirty` fields produced inside all detached
attempt manifests. An external launch sidecar preserves the launch commit and
dirty state, while exact config, run-code content, schedule, initialization,
data, environment, checkpoint, diagnostic, and artifact identities all pass.
The attempts were not rewritten. After launch, monitoring cadence was reduced
to 30 minutes at the user's request.

## Figure 01

`figures/01-h-vs-site-near-zero-grid.pdf` plots the count-first epsilon-`1e-3`
near-zero trajectories at `q_post`, `k_post`, `v`, and `m` against targeted
`h`, using the same four-panel, shared-zero-scale design as Run 002 Figure 02.
The terminal diagnostics already included all five sites, so no checkpoint
replay or new scientific measurement was required. The reduction and complete
interpretation record are in `artifacts/figure_data_sitewise.json` and
`observations/O001-h-vs-site-near-zero-grid.md`.
