# Run 028: all-site sparse kernel search for Pythia-14M

Status: implementation and calibration. This is not a measured success.

## Authorization and objective

The user explicitly authorized iterative implementation, all launches and runs,
and a new USD 20 total cloud envelope on 2026-09-06. No additional design or
launch confirmation is required. The objective is a faster specialized kernel
with coverage of all six block matmul families counted by R_model, and verified
evidence that higher achieved R_model can produce additional speedup. An
unfavorable result does not authorize changing checkpoints or numerical gates,
omitting failures, or declaring a partial-coverage kernel the completed goal.

## Fixed scientific contract

- Reuse all 35 Run027 checkpoint identities: 30 main-study and five explicitly
  historical A4+OL1@h endpoints. Six-layer Pythia-14M, D128, FFN512, H4 x d32,
  vocabulary 50304. These are existing seed-1234 random-pretraining endpoints
  at optimizer boundary 712, not released pretrained weights or new training.
- No optimizer updates, initialization, training data order, weights, gates,
  thresholds, pressure sites/method, activation topology, or checkpoints change.
  Original training settings and realized interventions remain in provenance.
- BF16 parameters/logits, B1 T2048, full vocabulary logits, uncached causal
  inference. TF32 and reduced-precision BF16 reductions disabled. Default
  cuBLAS workspace. Compare stock eager and the frozen Run027 K019+K018 path.
- Development: existing 64 seed-2500 training-cache blocks. Final timing:
  64 validation blocks selected by seed2504, seven randomized paired passes,
  three fresh processes on one GPU. This is an adaptive follow-up to prior
  observed checkpoint-level results, not a new untouched training-seed holdout.
- Every final deployment must pass all 500 validation documents / 338 complete
  2048-token blocks, 692224 inputs, 691886 prediction targets; excluded tail
  1444 tokens. Unchanged bounds: elementwise atol=.25, rtol=.02; relative L2
  <=.02; absolute pooled loss difference <=.001 nat/token. Failures stay visible.
- Canonical R_model retains Run027's source FP16-autocast integer counts and
  dense LM-head denominator. New native/candidate BF16 eligible and executed
  skip counters are separate quantities. No relabeling precision or counts.

## Implementation search and evidence

Scope is a->QKV, m->W1, h->W2, z->Wo, post-RoPE Q/K->QK, and P/V->PV.
The LM head remains a declared dense denominator, not part of the zero-product
numerator. Explore shape-specific sparse projections and fused exact attention;
all-zero rows/tiles may use algebraic shortcuts only with correct causal softmax
normalization. Nonzero operands cannot be dropped. Data-dependent decisions,
packing, prefix construction and sparse metadata are inside timing.

First candidate investigates explicit sparse QK/PV and exact zero-query
shortcuts. All-site projection integration, numerical repairs, and shape/fusion
tuning remain required work, not assumed complete by the first prototype.
Record immutable source snapshots and new attempt directories for each trial.
Freeze the chosen implementation/policy before the final cohort timing. Keep
same-implementation skip-disabled controls and component ablations; the control
is not presumed to be an optimized dense competitor.

Success needs qualified speed improvement over both stock and the previous
kernel, coverage/skip evidence for every requested operation, and a reproduced
positive R_model-speedup relationship (including within-family contrasts, not
just topology-confounded points). Report all 35 endpoints, timing uncertainty,
quality failures and exclusions. No manipulated axes, selected-only positive
curve, or universal speed law. A single seed limits scientific generalization.

This addresses manuscript methodology eq:r-model-measured and the runtime
realization of logical opportunity. No manuscript edit or finding promotion is
authorized by this implementation task.

Retain full raw timings, full-validation losses/errors, a/m/h/q/k/v/z exact and
near-zero counts, RMS/L2, weight norms, row/tile occupancy, per-operation logical
and actual skip counters, source/environment hashes, clipping/gate identities,
and original final checkpoints/cache identities. Gradient conflict is a
training-only measurement linked to the original runs; it cannot be recovered
from these inference measurements. No training is launched.

## Budget and execution

New task envelope USD20 including setup, idle development, failed attempts,
storage and transfers. Existing 100GB network volume remains untouched and is
not a new task resource. Initial discovery: zero Pods and zero endpoints.
Current RTX5090 quote is community USD0.69/h (32GB); refresh before creation.
Run027's representative four-mode workload peaked near 3.9GB, so this fits with
headroom. Use the same CUDA GPU family remotely, avoiding impact on the user's
desktop and obtaining the pinned Linux CUDA development environment.

Initial development lease: at most four hours, 40GB container +40GB /workspace
Pod volume; reserve at least USD5 for final measurements/retrieval. Stop compute
at the lease deadline to preserve evidence if disconnected, then retrieve,
verify hashes and delete the Pod. No lease may exceed the remaining envelope.
Keep all mutable work under /workspace and jobs detached with persistent logs.
Monitor after predicted short-probe completion or each five minutes for long
jobs; report stage, loss, throughput and refreshed ETC. Warning conditions:
exceptions/nonfinite outputs, stale events >5min, disk >85%, GPU memory >90%,
deadline or budget risk. No repeated polling of unchanged active jobs.

Transfer the hash-checked existing Run027 model/data bundle plus new run sources;
return every terminal trial, source snapshot, diagnostics, and raw timings.
Verify both archive and per-file hashes locally before deleting cloud storage.
Final publication PDFs, captions and observations will be run-local. Estimated
search ETC is initially 2-4 GPU-host hours, to be recalibrated after first probes;
the scientific objective is not guaranteed within that estimate or budget.

## First implementation and infrastructure evidence

- Local mathematical tests plus the full bootstrap suite: 245 passed. These
  tests do not establish CUDA/full-model equivalence.
- Lease `r7ex2sb18ax2yp`, community RTX5090, driver580.159.04, USD0.69/hour,
  began 16:57:06 UTC on 2026-09-06. Independent stop guard PID21920 is armed
  for 20:57:06 UTC. Pinned Torch2.11.0+cu128 is installed and real CUDA tensor
  operations pass. The base image's Torch2.8 is deliberately not used.
- K020 synthetic probe `k020-synthetic-001`: 12 cases at lengths17/129/2048,
  signed dense, zero Q, mixed queries, and zero V. All results are finite;
  zero V outputs are bitwise equal. Explicit QK/PV traversal and prefix-row
  counters are retained. This is a component probe, not full-model qualification.
  At T2048, zero-query prefix timing is about0.0200ms against native0.0426ms;
  dense-input K020 is about2.03ms against native0.0488ms. This motivates further
  specialization; it is not a trained-model speedup claim.
- Returned synthetic archive SHA256
  `9584d72525281b96cf6ff10f69abf168e8039af96ee6773ee6abfca066cc67cc`:
  all12 files /80541 bytes verified locally.
- The input relay stalled at82%. No incomplete checkpoint was used. The first
  receiver lacked the full relay suffix and used the image's old CLI; it was
  terminal before a corrected v2.12 receiver. A coarse SFTP repair was cancelled
  after measuring only0.4MB/s. Fine-block SSH repair now retains matching blocks
  and transfers314090304 bytes, with whole-archive SHA256 required at completion.
  An attempted partial gzip recovery found no valid requested checkpoints;
  a subsequent scoped recovery verified every A0 c01 input/provenance hash.
  This permits c01 development only, not a claim that the bulk archive is ready.
- K021 adds sparse a/m projections to K020 attention and inherited K018 h/z.
  `k021-projection-001` completed 10 synthetic cases (384/512 output widths,
  requested zero fractions 0/0.5/0.7/0.9/0.99). All outputs are finite and
  enabling/disabling sparse traversal is bitwise equal. At 90% zeros, median
  host times are 0.01245/0.01415ms versus native 0.02139/0.02005ms, respectively.
  This isolates a sparsity effect, but is not trained-model qualification.
- The first A0 full-model trials use 8 development training blocks and 3 paired
  timing passes. K020 and K021 obtain 0.1736x and 0.1654x stock speedup; both
  fail the fixed elementwise logit gate, although losses differ by only
  -0.000321/+0.000138 nat/token and all outputs are finite. Neither candidate
  is qualified. The preceding kernel passes this small development subset;
  this does not override its full-validation results in Run027.
- Returned archive `evidence-002`, SHA256
  `60cfa175c321949beec7c1820f2d9f28a14e98fe4e67b8ac65a3eeef0cb65afa`,
  contains 38 files /215378 bytes, all verified locally, including terminal
  projection and A0 trial records and source snapshots. No failed record is
  overwritten by a subsequent attempt.
- K022 changes attention scheduling to one 256-thread CTA per query, with
  parallel key scores and eight-way sparse V reductions. It preserves exact
  Q/K/V zero skipping and the zero-query causal-prefix shortcut; it does not
  drop nonzeros or change gates. It composes K021 a/m and K018 h/z projections.
  `k022-synthetic-001` completed all 12 cases with finite outputs. For the
  mixed T2048 input, its prefix-enabled non-rounded-P median is 0.1554ms,
  versus native 0.0462ms; it remains slower. Dense attention is 1.795ms,
  versus native 0.0480ms. The fully zero-query shortcut takes 0.02245ms,
  versus native 0.04758ms. These are not interchangeable with trained results.
  Archive `evidence-003`, SHA256
  `ec1c4e1b281d4a628aa6f65ebad73180b7080b85303fc67890a27a066d62cea9`,
  has 41 files /264330 bytes verified locally.
- Fine-block transfer repair completed: original archive SHA256
  `044dd4115f1bd4066dba11ae4e1b8305e3e939f6376b38562186b1df95ef9f19`
  matches in full. `15_install_inputs.py` then installed only the 248 declared
  input/provenance files and verified every hash for all 35 conditions. No old
  source code was extracted over the current implementation. The terminal
  receipt is `launch-control/rtx5090-001/inputs-complete-001.json`.
- `14_high_r.sh` is running sequential 8-development-block, 3-pass trials for
  c25/c30 on K020/K021/K022, followed by real-operand component probes. The
  sequence completed; K022 c30 reaches 2.288x stock versus previous 1.822x,
  but fails the elementwise logit gate on 6/8 development inputs. The loss
  difference is +0.00007675 nat/token. K022 c25 is only 0.5237x stock and also
  fails the elementwise gate. Neither is a qualified result. Real operands
  confirm c25 has no entirely zero query rows in the two-input probe, while
  c30 ranges from about25% (first layer) to 99.9-100% (last layer). Both kernel
  schedules retain matching actual QK/PV term counts on these cases.
  Archive `evidence-004`, SHA256
  `4e055f5b84b4d0cc769bdd893479806b0415e69a8c989ab3c1e1b27c9ab987a9`,
  has 143 files /1211469 bytes verified locally. The full local bootstrap plus mathematical tests
  were rerun after K022 implementation: 245 passed in 7.61s.
- A precision diagnostic identifies native Flash Attention dispatch and
  compares alternative backends and threshold crossings on the same captured
  operands. K023 tests a different arithmetic schedule: round the unnormalized
  exponential probabilities before sparse PV, then divide by the normalizer.
  It retains the same sparse traversal, prefix shortcut and gate thresholds.
  This is a new candidate, not a modification to executed K020/K021/K022.
  All-site qualified speedups and the final scientific figure remain open.

## Precision and attribution follow-up

`k022-c30-precision-001` confirms that direct SDPA reproduces the captured
reference attention output bitwise on all 12 layer/input cases. Unrounded K022
changes the 0.5 z-gate decision for one element in input0/layer0 and two in
input1/layer0. A one-BF16-step attention difference can therefore become a
0.5 post-gate difference. No tolerance or trained threshold was changed.
The normalized-BF16-P trial remains unqualified. Archive `evidence-005`
SHA256 `c486b133fed666f4c0e79c22d771e884acf53a8227e3a6a7ef5a93483d6247f3`
contains 155 files /1265928 bytes, verified locally.

K023's unnormalized-BF16-P c30 trial passes four of eight development input
logit gates, but fails the other four. Its 2.295x timing is therefore unqualified;
loss delta is +0.00003101 nat/token. Archive `evidence-006`, SHA256
`98daccc604c993297ebdb32185ef95f85e063548ac23a7e1e6eedbd3cf53e4a0`,
contains 194 files /1682367 bytes, verified locally.

K024 tests a reverse online-softmax tensor-core fallback with exact-zero
query/K/V **tile** bypasses. It does not skip individual zeros within a mixed
tile, and its counter unit is issued FMA-equivalent terms including padding,
not the scalar traversal count of K020-23. Its development speedups are 0.738x
(c01), 1.935x (c25) and 1.919x (c30); all fail the elementwise gate. This candidate
is neither qualified nor evidence that all logical opportunities are realized.
Archive `evidence-007`, SHA256
`712d8895df49d71807ed5fba2a6362526d4cf429bbe7e442dd8b2a386fff3839`,
contains 248 files /2040740 bytes, verified locally.

K025 is explicitly an attribution control, not an all-site candidate. It uses
the same direct attention wrapper with native SDPA, plus K021/K018 projections.
The all-four-projection skip toggle isolates savings within that wrapper.
Comparisons against this control are required before attributing a gain over
the earlier K019/K018 implementation to sparse attention.

Both K025 development endpoints pass all eight input gates with identical
pooled loss to stock. Speedups are 2.086x (c25) and 2.140x (c30); disabling all
four projection skips gives 1.248x and 1.223x, respectively, in separate paired
development trials. These timings are not the final 64-input/three-process
evaluation. Full-validation follow-up covers all 338 blocks: c30 passes all
gates, loss delta -2.823e-9 nat/token; c25 fails on blocks74/118/119 despite
loss delta -4.359e-6. Preserve this full-coverage failure rather than promote the
passing eight-input result. The development timing in the c30 full-validation
process is 2.132x stock against previous 1.810x. It still has dense attention.
Archive `evidence-008`, SHA256
`dc57e03430b7a4ec5802ad99344c4641bf54d1144e004492f7e49d1603b3cfca`,
has 318 files /2505822 bytes verified locally. Archive `evidence-009`, SHA256
`6141a28a9c8e2c42e16d620fc8054a7b55851266f12f45f4a2b0ca61d108767f`,
has 330 files /2703470 bytes verified locally, including both full-validation
trials and their closed logs. All current GPU jobs are terminal.

For the next precision-matched specialization, the installed Torch source
identity is `70d99e998b4955e0049d13a98d77ae1b14db1f45`; GitHub's immutable tree
identifies Flash Attention submodule `e2743ab5b3803bb672b16437ba98a3b1d4576c50`.
The [pinned D32 dispatcher](https://github.com/Dao-AILab/flash-attention/blob/e2743ab5b3803bb672b16437ba98a3b1d4576c50/csrc/flash_attn/src/flash_fwd_launch_template.h)
uses a 128x128, four-warp forward tile. Matching its actual arithmetic schedule
is a remaining implementation avenue; it has not been implemented or qualified
in this run. Preserve upstream licenses if headers are reused.

Next work remains the original all-six-operation objective: implement and
qualify a precision-matched sparse attention specialization, compare it with
the matched dense-attention control, diagnose projection rounding failures,
then gather per-site diagnostics and the repeated 35-variant final evidence.
Do not treat K025 as completing the attention requirement, or plot unqualified
2.3x timing as a valid achieved speedup. Latest CPU bootstrap/math rerun:
245 passed in 7.30s. No manuscript or final-figure change has been made.
