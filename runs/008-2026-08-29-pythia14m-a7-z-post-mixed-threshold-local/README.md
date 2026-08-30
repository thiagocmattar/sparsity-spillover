# Run 008 - Local Pythia-14M A7-Z-POST mixed threshold dose response

## Status

Design approved on 2026-08-29. Implementation, focused tests, full-suite audit,
production-shaped endpoint calibration, and the locked launch packet are
complete. The user granted local launch approval on 2026-08-30; execution
status is recorded below.

## Question and hypothesis

This run asks whether adding symmetric threshold gates at the actual
`q_post`, `k_post`, and `v` attention operands improves the
quality--logical-opportunity frontier relative to Run 006's one-sided
thresholding at `a`, `m`, `h`, and `z`.

At least one nonzero `kappa` would support the hypothesis if it creates a useful
increase in attention exact zeros and `R_model` without severe validation-loss
degradation. The hypothesis is weakened if every nonzero condition is unstable
or dominated by the corresponding Run 006 condition. Formal cross-run
reduction belongs in a future numbered analysis; this run retains the exact
matched identities required for it.

## Conditions and matched design

Five serial conditions use one common threshold at all seven active sites in
every layer:

| Order | Condition | `a,m,h,z` | `q_post,k_post,v` |
| ---: | --- | --- | --- |
| 1 | `kappa=0` | one-sided | symmetric |
| 2 | `kappa=0.01` | one-sided | symmetric |
| 3 | `kappa=0.05` | one-sided | symmetric |
| 4 | `kappa=0.1` | one-sided | symmetric |
| 5 | `kappa=0.5` | one-sided | symmetric |

The one-sided gate keeps `x >= kappa`; the symmetric gate keeps
`abs(x) >= kappa`. Equality survives, comparisons are detached, surviving
inputs receive identity input gradients, and rejected inputs receive zero input
gradients. At `kappa=0`, each symmetric gate is the exact identity, including
its gradient, and does not allocate a redundant tensor.

`q_post` and `k_post` are after partial RoPE and are the operands consumed by
causal QK scores. `v` is consumed by PV. `z` is the concatenated PV context
immediately before `W_o`. The operational topology is
`A7-Z-POST = {a,m,h,q_post,k_post,v,z}`. No L1 or OL1 pressure is used.

Everything else matches Run 006: pinned randomly initialized Pythia-14M,
seed-0 parameters, seed-0 realized block order, MiniPile caches, 2,048-token
sequences, global batch 64 as microbatch 4 by accumulation 16, optimizer, LR
schedule, validation coverage, diagnostics, and checkpoint retention. Released
Pythia weights are never loaded.

## Optimization, budget, and validation

Peak LR is `1e-3`. AdamW uses betas `(0.9,0.95)`, epsilon `1e-8`, weight decay
`0.1`, global task-gradient clipping at norm `1.0`, one-percent linear warmup,
and cosine decay to ten percent of peak. Parameters and optimizer state are
FP32; CUDA forward/backward uses BF16 autocast; dropout is zero.

Each condition executes 581 updates and 76,152,832 input tokens. The cohort
total is 2,905 updates and 380,764,160 input tokens. The schedule hash is
`c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4`,
identical to Run 006.

Every complete validation pass covers all 500 MiniPile validation documents:
338 complete blocks and 692,224 input tokens, with the 1,444-token tail
reported and excluded. Each condition has step-one, reloaded-final activation,
and independent eager logical-product coverage, for 15 complete passes.

## Diagnostics, ceiling, and retention

Every optimizer boundary records task loss, clipping, LR, throughput, wall
time, and peak GPU memory. Pressure/conflict/OL1 fields are prohibited.

The reloaded final checkpoint records count-first exact-zero and near-zero
counts at `0`, `1e-3`, and `1e-2`, RMS/L2 moments for `a`, `m`, `h`,
`q_post`, `k_post`, `v`, `z`, and post-`W_o` attention output, all named
parameter norms, and the six actual-operand logical-product counters. No
post-hoc clipping frontier is included. All five final checkpoints are retained
without optimizer state, estimated at 281,405,115 bytes total.

All six block-operation families are reachable. Because `v=0` already closes
through PV into `W_o`, selecting `z` adds no distinct all-zero reach. The exact
ceiling is `5,638,717,440 / 18,825,609,216`, or
`R_model_max = 29.95237697384911%`, for one full 2,048-token sequence.
`R_block`, `R_model`, and this ceiling are logical-product opportunities, not
removed FLOPs or measured speedup.

## Manuscript and operational relationship

This run tests the methodology's one-sided and symmetric activation operators
and its architecture-wide quality--logical-compute direction. It adds the
smallest operational per-site gate mapping needed for one-sided `a,m,h,z` and
symmetric post-RoPE Q/K/V gates while preserving the legacy uniform-gate
interface. The manuscript topology table still omits `z` and this combined
seven-site topology. No manuscript TeX is changed by this run.

## Implementation and verification

Shared methodology now registers `A7-Z-POST`, validates exact per-site gate
coverage, serializes mixed gates through checkpoints, and verifies every
realized layer/site gate. A zero symmetric threshold has an exact no-allocation
identity path. Real GPT-NeoX integration tests confirm post-RoPE Q/K placement,
signed surviving values, one-sided branch/context values, and equality between
captured `z` and the exact `W_o` input.

The final focused shared/Run 008 suite passes 39/39. Both PowerShell scripts
parse. The full bootstrap audit reports 122 passed and one pre-existing Run 007
state-expectation failure: its test expects `awaiting_launch_approval`, while
Run 007's append-only packet records `approved_for_launch` from its prior
user-approved retry. Run 008 has no failing tests, and the prior run was not
modified.

The launcher requires `launch_approved: true`, records Git/config/code/schedule/
calibration identities before detaching, and refuses any relaunch after an
attempt exists. The monitor defaults to 1,800 seconds and shortens its final
sleep to the refreshed completion window.

## Calibrated local launch packet

The production-shaped calibration is
`prelaunch/calibration-20260830-010451.json`; the locked but unapproved packet
is `prelaunch/launch-plan.json`. Both `kappa=0` and `kappa=0.5` executed eight
exact optimizer boundaries with the first two timing samples excluded. Both
completed ordinary, activation, and eager logical validation plus checkpoint
save/hash/reload. Each produced 48 activation layer rows, eight pooled sites,
and 76 weight rows.

For the fixed 581-step budget, median inclusive ETC is 3,024.27 seconds
(50m24.3s) and p90 is 3,115.69 seconds (51m55.7s), including 60 seconds of
terminal headroom. P90 leaves 424.31 seconds (7m04.3s) under the 3,540-second
planning envelope and 484.31 seconds (8m04.3s) under the 3,600-second hard
ceiling.

Peak Torch memory was 6,269,864,448 bytes allocated and 7,788,822,528 bytes
reserved on the 12,820,480,000-byte RTX 5070 Ti Laptop GPU. With 2,378 MiB used
by the desktop after calibration, conservative headroom is 2,538,143,744 bytes
(about 2.36 GiB). Avoid other GPU-heavy work during execution.

The first `.venv` calibration invocation stopped before model construction
because that environment has CPU-only Torch. The first CUDA calibration then
OOMed before publishing an artifact because `kappa=0` symmetric identity gates
were allocating redundant tensors. The mathematically identical no-allocation
identity path was added and tested; the final two-endpoint calibration then
completed. Neither stopped calibration created a scientific attempt.

## Interpretation limits

The run has one seed, one small model scale, and a short local horizon. The
joint Q/K/V addition cannot attribute an effect to any one attention operand.
Thresholded training changes forward values and gradient support. Cross-run
comparisons assume the locked common initialization, data order, and training
recipe. Logical opportunity is not hardware speedup.

## Approval record

- Design approved by the user on 2026-08-29: retain Run 006's one-sided
  `{a,m,h,z}` gates, add symmetric `q_post,k_post,v` gates, and use the same
  `kappa` for both gate families.
- Launch approved by the user on 2026-08-30 for the locked local execution
  definition and 60-minute hard envelope.

## Where we stopped

Implementation, testing, complete endpoint calibration, and the locked local
launch packet are complete. The approved local cohort subsequently completed
and terminally verified valid; detailed execution status follows.

## Execution status

The user granted launch approval on 2026-08-30. The detached cohort started at
2026-08-30T10:00:41Z as process 8396. Scientific attempt
`001-20260830-100058-5096ccb1` began with `kappa=0`; its first optimizer event
and complete step-one validation were healthy. At the first read-only check it
had reached step 133 with task loss 6.860099, throughput 133,013 tokens/s, and
step-one validation loss 10.751086 over all 338 complete blocks.

PowerShell `Get-Content` blocked while reading the actively written event file
and left two read-only monitor processes consuming host RAM; those monitor-only
PIDs were terminated without touching the cohort. Subsequent monitoring uses a
nonblocking Python file reader. This is a monitoring-infrastructure note, not a
scientific retry or change to the locked run.

## Execution completion

The cohort completed and terminally verified at 2026-08-30T10:47:30Z. Driver
wall time was about 46m49s, earlier than the calibrated 51m56s p90 window. All
five conditions completed 581 updates and 76,152,832 input tokens. The verifier
reconciled five final checkpoint hashes, 15 complete validation passes, the
common initialization and schedule identities, every diagnostic, and
380,764,160 total training tokens. `artifacts/verification.json` has status
`verified` and evidence label `valid`; retained checkpoints total 281,405,129
bytes.

| kappa | final validation loss | R_model | R_block | exact-zero q_post | exact-zero k_post | exact-zero v | exact-zero z | median tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.657176 | 0.078823 | 0.263160 | 0.000000004 | 0.000000015 | 0.000000002 | 0.696083 | 140,398 |
| 0.01 | 5.656776 | 0.084122 | 0.280853 | 0.008247 | 0.014631 | 0.019307 | 0.735389 | 143,019 |
| 0.05 | 5.662928 | 0.102549 | 0.342374 | 0.040421 | 0.065405 | 0.089862 | 0.838275 | 144,090 |
| 0.1 | 5.648046 | 0.120822 | 0.403381 | 0.065180 | 0.122371 | 0.170288 | 0.935149 | 143,775 |
| 0.5 | 6.036204 | 0.275723 | 0.920539 | 0.999388 | 0.999546 | 0.999507 | 1.000000 | 136,098 |

Within this one-seed run, `R_model` increases monotonically across the threshold
grid. `kappa=0.1` has the lowest observed final validation loss, while
`kappa=0.5` approaches the analytic logical-opportunity ceiling with a much
higher validation loss. These are standalone Run 008 observations; a formal
matched comparison with Run 006 belongs in a numbered cross-run analysis.
