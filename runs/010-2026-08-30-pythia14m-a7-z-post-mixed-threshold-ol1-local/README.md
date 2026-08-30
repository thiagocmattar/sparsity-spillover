# Run 010 - Pythia-14M A7-Z-POST mixed thresholds with all-site OL1

## Status

Design approved by the user on 2026-08-30. Implementation, focused verification,
the full bootstrap suite, and production-shaped local calibration are complete.
The user approved local launch at `2026-08-30T11:24:50.9193657Z`. The detached
cohort completed and terminal verification marked all five conditions valid at
`2026-08-30T12:27:27.919599Z`.

The next free identifier is 010 because Run 009 already belongs to the separate
full-pass, `h`-only OL1 experiment. This run does not alter that record.

## Question and comparison

This run asks whether Run 007's operational orthogonal-L1 correction improves
the quality--logical-sparsity frontier when applied after every threshold gate
in Run 008's A7-Z-POST topology. Run 008's five completed no-pressure
conditions are the matched comparators; they are not rerun here.

Support means a lower complete-validation loss at comparable `R_model`, a
higher `R_model` at comparable loss, or a new nondominated point relative to
Run 008. Domination by the corresponding Run 008 conditions, numerical
instability, or ineffective pressure weighs against the hypothesis. One seed
and one model size permit descriptive, not population-level, claims.

## Conditions and operational sites

All five conditions use the common threshold grid
`kappa in {0, 0.01, 0.05, 0.1, 0.5}` and A7-Z-POST sites
`{a,m,h,q_post,k_post,v,z}` in every one of six layers.

- `a`, `m`, `h`, and `z` use the one-sided operator
  `x * 1{x > kappa}`.
- `q_post`, `k_post`, and `v` use the symmetric operator
  `x * 1{|x| > kappa}`.
- `q_post` and `k_post` are after RoPE.
- `z` is concatenated `PV` context immediately before `W_o`; it is the
  approved operational interpretation of the user's `p` shorthand, not the
  attention-probability tensor `P`.

The same kappa is used by both gate families within a condition. Kappa zero is
the exact identity for signed symmetric sites and ordinary ReLU-equivalent
thresholding at the one-sided sites.

## OL1 boundary

Every condition applies `orthogonal_l1` at the seven post-threshold outputs.
The pressure scalar is the unweighted mean of 42 tensor-level means: seven
sites times six layers. Means are taken within each tensor before the global
mean, so tensor size does not change its weight. Lambda is `1.0` for this one
global objective and the trust/step budget is `1.0`.

At every optimizer boundary, task and pressure gradients are accumulated
separately over the same 16 microbatches. The task gradient alone is globally
clipped to norm 1 and drives AdamW and its moments. The pressure direction is
preconditioned by AdamW's task second moment, only the globally conflicting
component is projected away, its weighted direction-to-task ratio is capped at
1, and the correction is applied after AdamW. This is Run 007's operational
OL1 definition, not naive L1.

## Matched training contract

- pinned Pythia-14M architecture config and tokenizer revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`;
- random initialization, model seed 0 and data-order seed 0;
- MiniPile, 2,048-token sequences, global batch 64 as MB4/GAS16;
- 581 boundaries and 76,152,832 input tokens per condition;
- five conditions, 2,905 boundaries and 380,764,160 input tokens total;
- BF16 autocast with FP32 parameters, zero dropout, CUDA;
- AdamW, peak LR `1e-3`, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay
  `0.1`, 1% warmup, cosine decay to 10% of peak, task clip norm 1.

The schedule identity must equal Runs 006--008:
`c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4`.

## Validation, diagnostics, and retention

Each ordinary and diagnostic validation pass covers all 500 validation
documents: 338 complete 2,048-token blocks and 692,224 input tokens. The
1,444-token tail is excluded and reported. Ordinary validation runs after
boundary 1 and at the final boundary; final activation and logical diagnostics
run from the reloaded checkpoint.

The run retains count-first exact-zero and near-zero counts at `0`, `1e-3`, and
`1e-2`, activation RMS/L2 moments for all seven sites plus post-`W_o`
`attention_output`, all named weight norms including bias and normalization,
all six logical-product operation families, and every-boundary OL1 interaction,
projection, and trust measurement. `R_block`, `R_model`, and the integer
A7-Z-POST `R_model_max` are logical opportunities, not measured speedup.

Every final model checkpoint is retained; optimizer state and predictions are
not. There is no clipping-frontier diagnostic. Gradient interaction is
captured during training because it cannot be reconstructed later.

## Prelaunch verification and local fit

The focused requirement suite passes 31/31 and the complete bootstrap suite
passes 141/141. All Python files compile and both PowerShell scripts parse. The
calibration exercised kappa 0 and 0.5 with eight exact MB4/GAS16 OL1 boundaries
each (six timed after warmup), six complete validation/diagnostic passes, two
checkpoint save/hash/reload round-trips, 48 activation-layer rows and eight
pooled activation rows per endpoint, all 76 named parameter tensors, and both
logical-product endpoint summaries. All boundary quantities were finite and
the final pressure/task correction ratio respected the trust budget.

The governing non-evidence calibration is
`prelaunch/calibration-20260830-111454.json`. Its median ETC is 3,854.81 seconds
(64m14.8s) and p90 ETC is 3,895.85 seconds (64m55.9s), including one terminal
minute. This fits the 4,140-second planning envelope and 4,200-second hard
ceiling by 244.15 and 304.15 seconds respectively. The earlier `111314`
calibration used the same scientific and execution definition with a
provisional 90-minute envelope and is superseded by the tighter packet.

On the RTX 5070 Ti Laptop GPU, calibration peaked at 7,261,024,256 bytes
allocated and 8,334,082,048 bytes reserved. Against the 12,820,480,000-byte
device and a 961 MiB post-calibration desktop baseline, conservative remaining
headroom is 3,478,716,416 bytes (3.24 GiB). BF16 is supported. Local storage had
about 1.398 TB free at planning time.

## Local execution and approval boundary

The intended command is a detached, serial five-condition local CUDA run via
`04_launch.ps1`. The launcher rejects missing approval, identity drift, and
existing attempts. Each condition owns one immutable numbered attempt.

Run 007 peaked at 7.21 GB allocated and experienced OOMs under changing desktop
load. Run 010's matched MB4/GAS16 calibration fit with 3.24 GiB conservative
headroom, but desktop GPU load remains a warning condition and no automatic
microbatch change is authorized.

The proposed monitor interval is 30 minutes, with warnings for nonfinite
quantities, trust-ratio violations, OOM or less than 1 GiB conservative GPU
headroom, a stale event stream, throughput more than 25% below calibration,
ETC above its eventual hard ceiling, early exit, traceback, or identity and
coverage mismatches.

Calibration is non-evidence. `prelaunch/launch-plan.json` records the immutable
scientific/config/code/schedule/calibration identities and remains
`approved_for_launch` with `launch_approved=true` and the approval timestamp.

## Manuscript relationship

This run exercises the manuscript's AdamW-relative conflict-aware pressure idea
at an architecture-wide operational topology. It is broader than strategic
pressure placement and includes the operational `z` boundary. No manuscript
text or result claim will be changed without separate user approval.

## Completed result

The run completed 2,905 optimizer boundaries and 380,764,160 training input
tokens in 62m00s wall time. All 15 complete validation passes covered 338
blocks and 692,224 tokens while reporting the excluded 1,444-token tail. Five
final checkpoints total 281,405,129 bytes. Their content hashes, all attempt
transfer inventories, config/code/schedule/initialization identities, topology,
pressure definition, diagnostics, and logical counters passed terminal
verification. No optimizer states or predictions were retained.

| kappa | final validation loss | `R_block` | `R_model` | final train loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.649200 | 0.259548 | 0.077741 | 5.769070 |
| 0.01 | 5.645979 | 0.300136 | 0.089898 | 5.767919 |
| 0.05 | 5.669881 | 0.423879 | 0.126962 | 5.791913 |
| 0.1 | 5.678580 | 0.615236 | 0.184278 | 5.799648 |
| 0.5 | 6.029251 | 0.999999 | 0.299524 | 6.135406 |

Exact-zero fractions from the reloaded final checkpoints are:

| kappa | `a` | `m` | `h` | `q_post` | `k_post` | `v` | `z` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.497732 | 0.507149 | 0.796302 | 0.000000 | 0.000000 | 0.000000 | 0.562393 |
| 0.01 | 0.501261 | 0.510617 | 0.816824 | 0.044327 | 0.049777 | 0.040295 | 0.613259 |
| 0.05 | 0.512303 | 0.522174 | 0.892571 | 0.156521 | 0.180113 | 0.237144 | 0.785649 |
| 0.1 | 0.530639 | 0.540391 | 0.959488 | 0.411269 | 0.450106 | 0.530646 | 0.914920 |
| 0.5 | 1.000000 | 0.999995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The lowest observed validation loss is 5.645979 at kappa 0.01. Logical
opportunity rises monotonically across this grid; kappa 0.5 effectively reaches
the analytic A7-Z-POST ceiling `R_model_max=0.2995237697`, with validation loss
6.029251. The trust budget saturated at every boundary for kappa up through
0.1; at kappa 0.5 it saturated for 12.39% of boundaries and the final pressure
ratio fell to 0.0606 as the pressure objective approached zero. These are
within-run descriptions. A formal matched comparison with Run 008 belongs in a
separate numbered analysis.
