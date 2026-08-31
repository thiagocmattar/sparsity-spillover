# Run 016 - Pythia-70M selected ladder promotion

## Status

Implemented and locally verified; not launched. The user approved the scientific
interpretation on 2026-08-31. A separate launch approval is still required.
The next permitted billable action is one non-evidence A40 preflight, not the
12-condition scientific cohort.

The viewer named in the request,
`research/investigations/027-trading-data-representations/viewer-2026-08-19.html`,
is not present in this checkout. The approved step identities below are therefore
anchored to the user's request and the existing Run 004/014/015 implementations.

## Question and decision rule

Does the relationship between measured logical-product opportunity (`R_model`)
and complete-validation loss observed at Pythia-14M persist when the selected
ladder is promoted to randomly initialized Pythia-70M?

This is a promotion, not a broad ablation. Evidence is descriptive at one seed
and one larger scale. Persistence means that the promoted endpoints/frontiers
retain a useful loss--`R_model` tradeoff; loss collapse, absent measured
opportunity, or a qualitatively reversed ordering would refute that expectation.
`R_model` is not removed FLOPs or measured speedup.

## Twelve matched conditions

All conditions share model/data seed 1234, the same initial parameter draw, the
same training-block schedule, 712 optimizer boundaries, and one MiniPile pass.

- `a0-gelu`: A0 GeLU control.
- `a1h-relu`: A1-H ReLU control.
- `a4-ol1-kappa-{0,0p01,0p05,0p1,0p5}`: A4-Z with one-sided gates at
  `a,m,h,z`, OL1 pressure at the same four sites, `lambda=1`, trust budget 1.
- `a7-ol1-kappa-{0,0p01,0p05,0p1,0p5}`: A7-Z-POST with one-sided gates at
  `a,m,h,z`, symmetric gates at `q_post,k_post,v`, OL1 pressure at all seven
  sites, `lambda=1`, trust budget 1.

There is no released-weight loading. The Hugging Face revision pins only the
architecture configuration. The official `config.json` is vendored in this run
folder and tied to that revision, so construction does not fetch released
weights or depend on a mutable Hub response. Parameters are recreated in FP32
and initialized with the Pythia `small_init`/`wang_init` mapping.

## Model and recipe contract

The pinned architecture is 70,426,624 parameters, 6 blocks, hidden width 512,
MLP width 2,048, 8 attention heads, sequence length 2,048, and vocabulary 50,304.
The official Pythia recipe supplies LR `1e-3`, minimum LR `1e-4`, Adam betas
`(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, warmup fraction `0.01`,
gradient clipping at 1, zero dropout, and dynamic FP16. Run 016 keeps FP32
parameters/optimizer state and maps the original NeoX run to Transformers 5.12
and PyTorch 2.11; it is not a bitwise framework reproduction.

The original 70M multi-GPU recipe enables activation checkpointing. The approved
Run 016 proposal starts with checkpointing disabled and microbatch 4 with 256
accumulation steps, preserving global batch 1,024. If the exact A40 preflight
does not fit with 10% VRAM headroom, or if checkpointing appears necessary,
execution stops for review rather than silently changing this setting.

The 14M tokenizer cache is deliberately reused. Pythia-14M and Pythia-70M have
the same 50,304-token vocabulary, and both cache files are verified by byte size
and SHA-256 before use. The training schedule consumes 1,493,172,224 input
tokens per condition; it wraps 714 blocks after the 728,374 complete source
blocks. The 1,464-token training tail is excluded.

The locally verified schedule SHA-256 is
`d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`.
The pinned Torch 2.11/CUDA 12.8 seed-1234 FP32 initialization SHA-256 is
`724ad04e7233a747e3383e040077a2fc202e34293bf5e36e041a3efd2c5aac17`;
the A40 preflight must reproduce it.

## Validation and diagnostics

Step 1 and the reloaded final checkpoint each evaluate all 500 validation
documents: 338 complete 2,048-token blocks and 692,224 input tokens, excluding
the 1,444-token tail.

The final checkpoint additionally records count-first, per-layer and pooled
per-site activation statistics at `a,m,h,q_post,k_post,v,z,attention_output`:
exact zero, `abs(x)<=0.001`, `abs(x)<=0.01`, RMS, L2, and finite/non-finite
counts. All named parameter norms are stored. Actual-operand integer counters
cover all six logical operation families and yield `R_block` and `R_model`.

Every OL1 boundary verifies the exact pressure capture identity (24 tensors for
A4, 42 for A7) and stores task/pressure norms, dot product, cosine, conflict,
raw/final pressure-to-task ratios, trust scale, clipping, and overflow behavior.
These training-time interactions cannot be reconstructed later.

For one full 2,048-token Pythia-70M workload, the analytic declared-topology
ceilings are:

| Topology | Reachable products | Model products | `R_model_max` |
|---|---:|---:|---:|
| A0 | 0 | 104,293,466,112 | 0.000000% |
| A1-H | 12,884,901,888 | 104,293,466,112 | 12.354467% |
| A4-Z | 38,654,705,664 | 104,293,466,112 | 37.063401% |
| A7-Z-POST | 51,545,899,008 | 104,293,466,112 | 49.423901% |

These are analytic all-zero reach ceilings in integer logical-product units,
not observed zero rates. The model denominator includes 51,545,899,008 block
products and 52,747,567,104 dense LM-head products.

## A0/A1 post-hoc TEAL frontiers

After both final checkpoints pass training verification, `06_teal_posthoc.py`
applies the exact Analysis 005/006 uniform protocol to A0 and A1-H only:

- sites `a,m,h,z` at their actual matrix inputs;
- separate empirical threshold per site and model layer;
- calibration on the first 10 complete source-order training blocks;
- target sparsities `p in {0.0,0.1,...,0.9}`;
- full 338-block validation at batch one for every point;
- per-layer/pooled activation counts and moments, all logical counters,
  `R_block`, `R_model`, loss, throughput, and durable progress;
- `p=0` must reproduce the source final loss within `5e-4`.

This is evaluation-only uniform TEAL-style clipping, not block-wise greedy
allocation and not a sparse-kernel speed measurement. Weight diagnostics are
hash-linked because clipping does not change weights.

## Reuse decision

Reusing the 14M work is safer than rebuilding the full lifecycle, but importing
the 14M scientific assumptions is not. Run 016 therefore:

- reuses Run 004's hash-verified cache handling, schedule arithmetic, attempt
  lifecycle, event logging, full-validation evaluator, and generic diagnostics;
- owns and validates the 70M initializer instead of importing the 14M shape;
- implements one dynamic OL1 boundary for both the corrected Run 015 four-site
  behavior and Run 014 seven-site behavior; it does not import Run 012;
- derives layer counts and pressure capture identities from the actual model;
- replaces the inherited narrow timer with deferred cache slicing/staging so
  measured step time includes the complete recurring optimizer update;
- implements scale-generic TEAL mapping rather than the Analysis 005 six-layer
  literal.

The final recovery checkpoint alone is retained: FP32 model, optimizer, dynamic
loss scaler, schedule step/hash, and Python/NumPy/Torch CPU/all-CUDA RNG states.
The initial parameter hash replaces an extra step-0 weight copy. Expected final
recovery storage is about 0.81 GiB per condition and about 9.7 GiB for the cohort.

## Compute assessment (live snapshot 2026-08-31)

The local RTX 5070 Ti Laptop GPU exposes 12,227 MiB total VRAM and is not an
acceptable exact-fit target for the A7 FP16 component-gradient boundary with
headroom. Local CPU construction and a short forward pass validate the exact
70M graph, but cannot calibrate CUDA memory or ETC.

`prelaunch/smoke-20260831-224431.json` records the attempted local CUDA smoke.
The installed Torch 2.11/CUDA 12.8 runtime reports no Flash Attention kernel on
this GPU, so the smoke correctly stopped before constructing an unrepresentative
math-attention boundary. The separate exact CPU graph test remained finite.

RunPod currently lists Secure A40 48 GB at `$0.44/GPU-hour`, CUDA 12.8
available, global availability `LOW`, and a Secure maximum count of 10. Its
only listed locations are `EU-SE-1` and `CA-MTL-1`, both `LOW`. Therefore 12
A40s concurrently are unavailable even before account quota and transient stock
are considered. Run 016 uses condition-level parallelism but stages four
sentinels followed by up to eight remaining conditions; the eight may be split
into two groups of four if stock does not support one wave.

The retained 100 GB Standard network volume `9luykg5yc3` is in `EUR-IS-1`, where
the A40 is not currently offered, so it is explicitly not part of this launch.
Its separate continuing cost is approximately `$7/month`. Each Pod instead uses
a 25 GB volume disk and a 30 GB container disk, receives a hash-verified cache
copy, and is deleted only after transfer verification.

The exact science ETC is intentionally not extrapolated from 14M. The billable
preflight measures five complete A0 and A7 boundaries including cache slicing
and staging, a full validation pass, activation diagnostics, eager logical
diagnostics, checkpoint serialization, memory, and throughput. It then produces
the science ETC and uncertainty basis.

Cost guards at the current price are:

- one 1.5-hour A40 preflight: `$0.66` compute plus at most about `$0.012`
  prorated 55 GB Pod storage, maximum about `$0.672`;
- later 12-condition science cohort, only if separately approved: 96 guarded
  GPU-hours = `$42.24` compute plus about `$0.73` prorated Pod storage,
  maximum about `$42.97` before any unrelated retained-volume charge.

Prices and stock must be refreshed immediately before creation. RunPod documents
per-second Pod compute/storage billing and no ingress/egress fee; the provider
billing record remains provisional until refreshed after teardown.

## Preflight and launch gates

The preflight passes only if both exact A0 and A7 dimensions complete five
finite, non-overflowing boundaries; all initial hashes match; full coverage and
diagnostics reconcile; and peak reserved VRAM stays at or below 90% of the
actual A40 total. It creates no scientific attempt.

After a passed preflight, the repository must be updated with measured ETC,
transfer throughput, and the final wave/cost envelope. A second explicit launch
approval is then required for the four-condition scientific sentinel. The
remaining eight conditions require the sentinel identities, losses,
throughput, memory, checkpoints, and diagnostics to verify before proceeding.

Monitoring is every five minutes. Warnings are: event age over 10 minutes,
non-finite loss/gradient metric, any skipped boundary, capture mismatch,
reserved VRAM over 90%, unexpected schedule/hash/runtime, disk risk, or ETC
past the guard. Artifacts are copied and hash-verified locally before each Pod
is terminated; a final resource listing must show no unintended Pods.

See `DEPLOYMENT_PLAYBOOK.md` for the exact staged control flow.

## Approval record

- 2026-08-31: user approved the detailed Run 016 design interpretation,
  including the selected steps, one-seed matched promotion, complete validation,
  OL1 diagnostics, A0/A1 TEAL protocol, final recovery retention, and staged
  A40 preflight/condition-parallel plan.
- No preflight or scientific launch approval has been given.
