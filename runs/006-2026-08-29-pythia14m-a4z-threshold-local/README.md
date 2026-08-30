# Run 006 - Local Pythia-14M joint A4-Z threshold dose response

## Status

Design approved on 2026-08-29. Implementation, tests, production-shaped local
calibration, and the locked launch packet are complete. Scientific execution
remains prohibited until a separate explicit launch approval is recorded.

## Question and hypothesis

This run asks how jointly increasing a one-sided threshold at `a`, `m`, `h`,
and the new pre-`W_o` attention-context site `z` changes task quality, sitewise
exact sparsity, and model-wide logical-product opportunity over the exact Run
005 token budget.

Increasing `kappa` should mechanically enlarge the rejected region at every
selected gate and may increase measured `R_block` and `R_model`. A useful
quality--logical-compute response requires at least one nonzero threshold to
increase logical opportunity without severe validation-loss degradation or
training instability. Because all four sites change jointly, the run cannot
attribute a response to any one selected site.

## Conditions and matched design

Five serial conditions apply the same fixed gate in all six layers at topology
`A4-Z = {a,m,h,z}`:

| Order | Condition | Gate |
| ---: | --- | --- |
| 1 | `kappa=0` | `x` if `x >= 0`, otherwise zero |
| 2 | `kappa=0.01` | `x` if `x >= 0.01`, otherwise zero |
| 3 | `kappa=0.05` | `x` if `x >= 0.05`, otherwise zero |
| 4 | `kappa=0.1` | `x` if `x >= 0.1`, otherwise zero |
| 5 | `kappa=0.5` | `x` if `x >= 0.5`, otherwise zero |

Equality survives and the comparison is detached, so surviving inputs receive
identity input gradients and rejected inputs receive zero input gradients.
`kappa=0` is forward-value equivalent to standard ReLU at all four selected
sites; at exact equality its approved identity-gradient convention is explicit.
No L1 or OL1 pressure is used in any condition.

Everything else matches Run 005: pinned Pythia-14M architecture, random
initialization, seed-0 parameters, seed-0 realized block order, MiniPile
caches, 2,048-token sequences, global batch 64 as microbatch 4 by accumulation
16, optimizer, LR schedule, validation coverage, diagnostic coverage, and
checkpoint retention. Released Pythia weights are never loaded.

## Operational `z` definition and manuscript relationship

`z` is the concatenated attention context produced by `PV`, shape `[B,T,D]`,
immediately before `attention.dense`/`W_o`. It is not the post-`W_o`
`attention_output` diagnostic. The implementation exposes an identity capture
tap and optional gate after head concatenation and before the exact tensor is
consumed by `W_o`.

The manuscript architecture-map artifact includes this `z` site, while the
current methodology prose and topology table do not yet list it. This run
operationalizes the map as new topology `A4-Z`; it does not edit manuscript
TeX. The all-zero reach is `a->QKV`, `m->W1`, `h->W2`, and `z->W_o`. For
Pythia-14M at `T=2,048`, the integer ceiling is
`2,415,919,104 / 18,825,609,216`, or `R_model_max = 12.833152%`.

## Optimization, budget, and validation

Peak LR is `1e-3`. AdamW uses betas `(0.9,0.95)`, epsilon `1e-8`, weight decay
`0.1`, global task-gradient clipping at norm `1.0`, one-percent linear warmup,
and cosine decay to ten percent of peak. Parameters and optimizer state are
FP32; CUDA forward/backward uses BF16 autocast; dropout is zero.

Each condition executes 581 updates, 76,152,832 input tokens, and the exact
Run 005 realized training schedule. The cohort total is 2,905 updates and
380,764,160 input tokens. Every validation pass covers all 500 MiniPile
validation documents: 338 complete blocks and 692,224 input tokens, with the
1,444-token tail reported and excluded. Validation runs after update 1 and
from the reloaded final checkpoint; an independent eager logical-product pass
covers the same complete workload.

## Diagnostics and retention

Every optimizer boundary records task loss, task-gradient clipping, LR,
throughput, wall time, and peak GPU memory. Pressure/conflict/OL1 metrics are
inapplicable and must not appear.

The reloaded final checkpoint records count-first exact-zero and near-zero
counts at `0`, `1e-3`, and `1e-2`, plus RMS/L2 moments for `a`, `m`, `h`,
`q_post`, `k_post`, `v`, `z`, and the post-`W_o` attention output. It also
records all named parameter norms and the six actual-operand logical-product
counters. `R_block`, `R_model`, and `R_model_max` are logical opportunities,
not removed FLOPs or measured speedup. No clipping frontier is included.

All five final checkpoints are retained without optimizer state, expected to
require about 281 MB total. This preserves checkpoint-reconstructible future
diagnostics.

## Interpretation limits

The run has one seed, one small model scale, and a short local horizon. The
joint topology does not isolate site-specific causal effects. Thresholded
training changes both the forward values and gradient support. Exact zeros and
logical opportunities are not hardware speedups, and no sparse kernel is
benchmarked.

## Approval record

- Design approved by the user on 2026-08-29: joint one-sided thresholding at
  `{a,m,h,z}`, `kappa in {0,0.01,0.05,0.1,0.5}`, no L1/OL1 pressure, all other
  Run 005 training settings and token budget matched, complete diagnostics,
  logical counters, and all final checkpoints retained.
- Launch approval: not yet requested or granted.

## Implementation and verification

Shared methodology now exposes and gates `z` after PV head concatenation and
immediately before the exact tensor consumed by `W_o`. Topology metadata,
activation capture, checkpoint reload, and architecture ceilings include the
new site. A real GPT-NeoX integration test asserts bitwise equality between the
captured `z` tensor and the `W_o` input. The logical counter's existing
attention-output-projection pre-hook therefore observes the gated operand
directly.

The focused shared/Run 006 suite passes 32/32. The full bootstrap suite passes
99/99. Both PowerShell scripts parse without errors. The launcher refuses to
run without a launch-approved packet, records Git/config/code/schedule
provenance before detaching, and refuses any relaunch after a numbered attempt
exists. Monitoring defaults to 1,800 seconds and reports progress, loss,
throughput, and refreshed ETC.

## Calibrated local launch packet

The production-shaped endpoint calibration is
`prelaunch/calibration-20260829-204815.json`; the locked execution definition is
`prelaunch/launch-plan.json`. It exercised eight exact optimizer boundaries at
both `kappa=0` and `kappa=0.5`, excluding two warmup timing samples from each.
Both endpoints completed full ordinary, activation, and eager logical
validation; 48 activation layer rows, eight pooled sites, 76 weight rows, and
checkpoint save/hash/reload passed.

For the fixed 581-step budget, median inclusive ETC is 2,872.19 seconds
(47m52.2s) and p90 is 2,893.51 seconds (48m13.5s), including 60 seconds of
terminal headroom. This leaves 706.49 seconds (11m46.5s) under the 3,600-second
ceiling. The calibration reproduced Run 005's schedule hash
`c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4`
and initialization hash
`8e511149aac21f3fcacccb9968299bf3938473ff38f4d53cb99f2e9e1a2403bc`.

Peak Torch memory was 6,269,864,448 bytes allocated and 7,786,725,376 bytes
reserved on the 12,820,480,000-byte RTX 5070 Ti Laptop GPU. With 2,884 MiB used
by the desktop after calibration, conservative headroom is about 2.01 GB.
Avoid other GPU-heavy work during execution.

Each final checkpoint calibrated at 56,280,470 bytes, for approximately
281,402,350 bytes across the cohort. No optimizer states, cloud resources,
transfer, or billable storage are involved.

## Where we stopped

Implementation, tests, smoke, and exact local calibration are complete. Await
explicit launch approval for the locked five-condition local cohort.

## Execution completion (supersedes the pre-launch status above)

The user granted launch approval on 2026-08-29. The first infrastructure-only
detachment attempt stopped before creating a scientific attempt because Windows
PowerShell could not enumerate duplicate case variants of `Path`; its record is
under `artifacts/launch-attempts/001-start-process-environment-collision/`.
After replacing only the detached-process mechanism, locking the new run-code
hash, and rerunning the full suite (99/99 passed), the cohort started at
2026-08-29T21:11:17Z and verified at 2026-08-29T21:59:45Z. Total wall time was
about 48m28s, below the 60-minute ceiling.

All five conditions completed 581 updates and 76,152,832 input tokens. The
verifier reconciled five retained checkpoint hashes, 15 complete validation
passes, the common initialization and schedule hashes, all diagnostic coverage,
and 380,764,160 total training tokens. `artifacts/verification.json` has status
`verified` and evidence label `valid`.

| kappa | final validation loss | R_model | R_block | exact-zero a | exact-zero m | exact-zero h | exact-zero z | median tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.661601 | 0.079207 | 0.264443 | 0.492311 | 0.504059 | 0.800789 | 0.710152 | 138,706 |
| 0.01 | 5.656006 | 0.080692 | 0.269400 | 0.497273 | 0.509610 | 0.819225 | 0.738115 | 136,631 |
| 0.05 | 5.673034 | 0.086014 | 0.287171 | 0.506076 | 0.521434 | 0.898973 | 0.843148 | 137,620 |
| 0.1 | 5.651189 | 0.091279 | 0.304748 | 0.523409 | 0.539994 | 0.966644 | 0.938568 | 136,726 |
| 0.5 | 6.036731 | 0.104624 | 0.349301 | 0.680493 | 0.685429 | 0.999990 | 1.000000 | 136,470 |

These are descriptive one-seed results for the approved joint intervention;
they do not attribute effects to individual sites or establish runtime speedup.
