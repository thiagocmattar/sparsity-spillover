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

## Native-dispatch audit and first qualified all-site candidate

The preceding ordinary-forward dispatcher inference was incomplete. The
`dispatch-001` CUDA trace shows that native BF16 B1 H4 T2048 D32 actually uses
`flash_fwd_splitkv_kernel<32,64,256,4,...,Split=true>` and a two-split combine.
At T129 it uses the ordinary 128x128 forward kernel. A CPU operator name alone
did not reveal this distinction. No historical measurement is overwritten.

K026 ports the ordinary forward schedule with exact-zero MMA-atom bypasses.
Its first two attempts failed before execution (upstream header order, then a
`forward` name collision); both source snapshots and logs are retained.
Attempt003 compiles and passes sparse/no-skip bitwise equality in all36
component cases, but differs from the native T2048 split schedule. Full338
validation fails for both c25 and c30; c30's 2.319x timing is unqualified.

K027 changes to the observed two-split 64x256 schedule, native split reduction
and combine, and no fast-math compiler option. All28 synthetic/captured
components are bitwise equal to native, with and without MMA skipping.
The high-R c30 endpoint passes every full338-block gate. Its preliminary
8-development-input x3-pass speedup is 2.274x versus previous1.816x; pooled
loss difference is -2.823e-9 nat/token. c25 still fails the projection-related
blocks74/118/119, as K025 did, despite 2.262x timing. These are not the final
64-validation-input, three-process study estimates.

K027's separate c30 attention-skip-disabled trial also passes, at 2.289x stock.
This does not establish an attention-specific speed benefit: matched multi-mode
attribution and further specialization are required. Sparse-MMA counters are
issued 16x8x16 atoms, including padding; the dense two-split control issues
147456 atoms per QK or PV operation (301989888 FMA-equivalent terms), not the
268566528 causally valid scalar products of the R_model denominator.

`projection-precision-001` captures native a/m operands on c25 validation
blocks74/118/119 and c30 block0. It records238 differing output entries across
48 layer/site cases. Native outputs are closer to the FP64 dot on192 entries,
the sparse output on43, with3 ties. Separately rounding the BF16 matmul before
bias does not match native. These are selected-case diagnostics, not general
error-frequency estimates. K028 tests FP64 recomputation within8 FP32 ULPs of
a BF16 rounding midpoint; c25 remains unqualified, c30 qualifies at2.257x.
It has not improved the qualification coverage or demonstrated a speed gain.

K029 is testing an exact-grid prefix shortcut for wholly zero64-query tiles,
while retaining K027's sparse MMA fallback. It checks actual inputs, not the
checkpoint label or threshold: K must be finite, and each V must be zero or
have magnitude in[0.5,64]. Under this sufficient condition, each <=1024-term
BF16 sum is an integer multiple of2^-8 within FP32's exact integer range.
The shortcut retains the native split normalization and combine; it is not
the earlier rounded/unnormalized prefix algorithm. Prefix work, safety tests,
and extra launches are inside timing. Bypassed QK atoms, PV prefix reuse, and
query-split rows are recorded separately from zero-operand MMA skipping.

The dependency inventory pins Flash `e2743ab5b3803bb672b16437ba98a3b1d4576c50`
and its nested CUTLASS `7127592069c2fe01b041e174ba4345ef9b279671`, with retained
BSD licenses. Torch's separate CUTLASS pin is
`0d2b201e8c1c4a03efa6e9c468161916e2334725`. K027's equality claim rests on the
measured bitwise checks, not on claiming all dependency identities are equal.

Returned and locally hash-verified evidence (archive SHA256):

| Bundle | Files / bytes | SHA256 |
|---|---:|---|
| evidence-010 |340 /3081130|`0e4f84fbdcf670c486a2b3b455ae64ee13689acb47eabb34a9081b863f2fb229`|
| evidence-011 |351 /3399186|`40155150b60fe95564d24b9f5edb778430f3a78457558a1b4b40184cf9b3b722`|
| evidence-012 |365 /3914695|`b5836dc1f178709eff2bbe17ede6d748fefdec41cdf629923961a938cd4905fa`|
| evidence-013 |426 /4670256|`e05baf0de81ce9e1e1dc3d78e220787ec5ab28e51086ae61a66d98dff0dd4f1e`|
| evidence-014 |476 /5737191|`203d85c6e0d684cbe51597a5e2c038ecd01e83be1baf825788a89729b78609fe`|
| evidence-015 |534 /6590044|`1f45311a47a52b4a726e71e01adf1a48f7e0434413292332cf87b04bc47934ed`|

Latest local bootstrap/math test run:248 passed in7.20s. Native matching on
components and one qualified endpoint do not complete the objective: measured
attention-specific benefit, broader qualification, a frozen repeated35-variant
study, all-site diagnostics and the final scientific PDF remain required.

K029's30 components are bitwise equal to native in all three modes (dense,
sparse+prefix, sparse without prefix); both QK and PV issued+skipped+bypassed
atom counters reconcile to147456 in all90 cases. It passes c30's full338-block
validation, with the same loss delta as K027 and maximum absolute logit
difference0.25 (within the unchanged elementwise gate). Its preliminary2.237x
timing does not improve K027. c25 remains unqualified. Evidence-016 contains
585 files /7756457 bytes, locally verified; archive SHA256
`38fd797d953ef784d149d48fdb051c2d59250583f03220cd48df85d32342ac03`.

K030 moves the exact-grid shortcut inside the zero-query CTA, eliminating the
global prefix buffer and its extra launch. It uses the same sufficient grid
bound and split normalization. If the grid/finite check fails, every tentative
partial output is overwritten by the normal attention computation. The
successful-reuse counters do not count this failed-shortcut overhead; all such
work remains included in timing. `36_attribution.py` pairs native, previous,
sparse, attention-dense, no-prefix and all-skips-disabled modes on the same
development inputs, followed by full validation of all six. K030 and these
multi-mode results are still being evaluated, not frozen as a final policy.

K030 completed: all30 components match native and the skip toggles bitwise.
The six-mode c30 test passes all338 blocks, but sparse attention is 1.5% slower
than its matched attention-dense control (2.266x versus 2.299x stock). c25
continues to fail the same three projection-related blocks in all new modes.
K031 changes only the exact-prefix memory mapping: lanes load contiguous
features, warps own token slices, and shared memory combines exact-grid sums.
Only causally needed K/V are checked. All30 components again match native
and the skip toggles bitwise. Full338 qualification is unchanged: c30 passes,
c25 fails. On32 development inputs x7 paired passes, c30 achieves2.2566x
stock versus previous1.8294x; attention skipping gives only1.00274x over the
same implementation with attention skips disabled, and prefix reuse gives
0.99850x over sparse MMA alone. This is not an established attention benefit.
All-skips-enabled versus all-skips-disabled is1.77962x on c30. These timings
remain development evidence, not the final held-out three-process graph.

Evidence-017 (636 files /9988953 bytes) and evidence-018 (687 files /12214868
bytes) are locally archive- and per-file-SHA256 verified. Their archive hashes
are `fc9e8bde722657f841e8166daac663fad9296460cc5a8ca08981c0eff1ea47e4`
and `d2df39f5e7b70755b8bdc7b20749d76840936e9a316d17879b64e4398ac88b7f`,
respectively. All K026--K031 processes and included logs are terminal. Latest
local bootstrap/math suite:248 passed in7.23s. At18:25 UTC the existing Pod
remains available for the approved search, about USD1.02 compute since creation
at USD0.69/hour, under the USD20 total ceiling; posted billing is not final.
The independent stop deadline remains20:57 UTC. Next work tests tensor-core
projection arithmetic without changing checkpoints, gates, or tolerances.

K032 tests a/m projection accumulation using BF16 `mma.m16n8k16` in forward
K16 order and one final BF16 bias rounding. It skips wholly zero16x16 input
fragments, not individual scalar zeros, while retaining K031 attention and
K018 h/z. `42_tensor_projection_probe.py` tests synthetic inputs and the same
48 selected native-operand cases, and records native projection CUDA dispatch.
`43_k032.sh` then runs both six-mode/full338 endpoint tests. This is not a
policy selected by checkpoint identity. CPU/bootstrap suite248 passed in7.25s;
expected existing-Pod runtime about two minutes, cost roughly USD0.03 or less,
with a40-minute controller hard limit inside the existing four-hour Pod guard.
Source, failed attempts, logs, full quality/timing results remain retained.

An additional matched CUDA-graph control (`44_graph_controls.sh`) tests K031
and K032 at c25/c30 with every comparator captured, including stock. It keeps
resident-input full50304-logit work, all gates, full338 numerical coverage,
and the same32 development timing identities x7 passes. Input staging is
equally excluded; capture/setup is disclosed outside steady-state timing.
This separates Python launch overhead from sparsity-dependent GPU savings;
graph candidate versus eager stock will not be mislabeled as skip-only gain.
Expected four-process duration about two minutes after compilation, under
USD0.04, with a60-minute controller limit and the existing Pod guard.

K032's six synthetic plus48 captured projection cases are bitwise native,
including skip/no-skip equality. Both full338 endpoints pass, pooled loss
identical to native, at2.2613x (c25) and2.2513x (c30) eager stock. Evidence-019
contains743 files /13961538 bytes verified locally; archive SHA256
`616bce212ee8a8865c13e1f40a380958cf71d56a5adf63a6f0c569be4dc718ff`.
The projection profiler identifies native a/h/z WMMA kernels and an m16816
tensor-op kernel; names alone are not performance estimates.

The initial graph attempt fails before timing because Transformers5.12.1
creates an explicit causal mask during capture, while the custom attention
requires the ordinary unmasked causal path. `45_graph_forward.py` exposes the
same fixed B1/T2048 uncached/unpadded forward to every graph comparator,
including native, preserving implicit causal SDPA. An additional unmodified
eager-stock runner anchors all338 validation blocks. No graph claim is valid
unless that anchor passes. `46_graph_controls.sh` uses new attempt002 folders;
the original failed attempt remains retained. K033 additionally prototypes
tensor-core h/z with exactly the two rounded linear outputs and two rounded
residual additions; it is not yet executed or qualified.

K032's full-quality artifacts confirm maximum logit difference0 on every
block at both c25 and c30, in all new modes. The initial graph failure and
closed controller log are returned in evidence-020:756 files /14070423 bytes,
archive SHA256 `002ba8fc49e82cf6121c9719a480e5d9c718fe45f7c8a50bd5ea8a37704df0af`.
K033's next stage (`47_joint_tensor_probe.py`, `48_k033.sh`) covers12 synthetic
gate/sparsity cases, then full338/seven-mode graph comparisons at c01, c11,
c25 and c30. Gates, initialization, checkpoints and numerical bounds remain
unchanged. Source-level test suite248 passed in7.43s; CUDA arithmetic is not
claimed tested by that suite. Estimated existing-Pod duration three minutes
after compilation, about USD0.04, with a70-minute hard controller timeout
bounded additionally by the existing20:57 UTC Pod stop deadline.

Completed graph002 controls retain the negative performance result: K032
c25/c30 qualify, including the eager-stock anchor, but achieve only1.256x/
1.278x against matched graph-native, versus previous1.583x/1.582x. c30 sparse
latency0.643ms exceeds attention-dense0.628ms; Python overhead was not the only
problem. Evidence-021 contains839 files /17966339 bytes verified locally;
archive SHA256 `6e87aa15d60d5b3f6ccf967a4cdb393059e572214ac7dae519d68d13801c6d59`.

K034 improves projection data reuse with32x64x64 shared-memory tiles and a
bank-swizzled layout. It retains K16 MMA order, exact gates, four-projection
zero-fragment skipping and the rounded joint residual epilogue. Shared weight
loads still occur when a fragment is zero; the skip counter must not imply
these memory transactions are eliminated. K031 attention remains unchanged.
`49_shared_projection_probe.py` and `50_shared_joint_probe.py` precede the
same four full338/seven-mode endpoint tests in `51_k034.sh`. CPU tests248 pass
in7.01s. Expected duration three minutes plus first compilation, about USD0.05,
with an80-minute controller limit and the existing independent Pod deadline.

K034 qualifies at all four endpoints but is slower: c30 reaches0.948x graph
native, sparse0.869ms versus all-skips-disabled0.722ms. Resource inspection
finds no register spills (LOCAL0); shared reuse did not compensate for the
fragment-test/control cost. It is not promoted. K035 therefore retains the
faster qualified K033 projections, and tests a segmented coalesced attention
prefix:128 CTAs compute local64-token prefix sums, and zero-query CTAs reuse
them with an exact-grid sum of preceding segments. Safety checks are per
causally needed segment. Prefix preparation stays inside timing; no threshold
or checkpoint identity dispatch is introduced. `52_segmented_prefix_probe.py`
and `53_k035.sh` cover30 components and both full338/seven-mode graph endpoints.
CPU suite250 passes in7.36s, including tile-map and segmented-prefix tests.
Expected runtime three minutes including compilation, about USD0.04; a
55-minute controller timeout remains inside the existing Pod stop guard.

K033 and K034 are bitwise identical to native on every full-validation block
at c01, c11, c25 and c30 (not merely within tolerance). Their12 joint primitive
cases are also bitwise native and skip-toggle equal. Evidence-022 has939
files /21889819 bytes, archive SHA256
`3b14ba90f5ae97e8ec398416fb3a3fff429ea260836b2494b149e987e1629363`;
evidence-023 has1063 files /25961107 bytes, archive SHA256
`a7b8fb2be8e90442365b5dca4fffcb08dad4ee3dcdb3cbc83e514db8c32cf9a5`.
Both archives and their per-file inventories are verified locally.

K035 qualifies at both endpoints but still loses to attention-dense: c30
sparse0.698ms versus attention-dense0.681ms, graph-native speedup1.177x.
It does not establish the desired attention benefit. K036 tests the pinned
CUTLASS32x32x64 projection pipeline with a custom exact-zero MMA operator,
bias broadcast and final BF16 rounding, retaining K033 h/z and K035 attention.
The dense ablation uses the same pipeline with the ordinary MMA operator.
`54_cutlass_projection_probe.py` precedes four full338/seven-mode graph
endpoints in `55_k036.sh`. CPU suite250 passes in7.07s; first GPU compile and
full checks are estimated at four minutes /USD0.05, with a75-minute controller
cap inside the existing independent Pod stop deadline. This is still search,
not a frozen final policy or a claimed positive sparsity trend.

Evidence-024 contains1131 files /28666463 bytes verified locally, archive
SHA256 `ab42b50c834995bfe9e0b14784ec6a6a3abac8ed9b10f08fafd10de6af36a439`.
All30 K035 components are bitwise native and skip-toggle equal. In the selected
c30 training input0, pooling six layers' counters gives380673 QK issued atoms
and265922 PV issued atoms out of884736 each:57.0%/69.9% eliminated or replaced
by exact-prefix reuse. This is a single BF16 development input, not the full
FP16 canonical R_model estimand or the final all-site diagnostic.

K036 attempt001 fails compilation: CUTLASS's eight-element epilogue access
produces an invalid thread map for the16x16 warp tile. No timing exists.
Attempt002 changes only epilogue access width to four elements, retaining the
same scalar computation, bias, MMA order and input policy. The original
source snapshot/error log remains retained; `56_k036_build_retry.sh` writes
new attempt002 folders with the same bounded execution envelope.

K036 attempt002 completes: all four full338 endpoints have maximum logit
difference0 in the no-prefix mode. Its paired graph-native speedups are
1.0961x/1.1573x/1.3812x/1.3860x at c01/c11/c25/c30. Matched all-skips-enabled
versus all-skips-disabled ratios are0.96312x/0.97167x/1.01771x/1.02407x,
respectively. Thus sparse overhead still hurts low-sparsity controls, while
the high endpoints show a modest positive net skip benefit. Attention alone
remains slower (c25 0.98745x, c30 0.99284x); these results do not establish
an attention-specific speed win. Previous graph execution is still faster at
c25/c30 (~1.583x graph-native), while it fails c01/c11's numerical gate.
All limitations and failures must remain visible in the final evidence.

Evidence-025 contains1137 files /28845172 bytes, archive SHA256
`09e85badd01d21ab7fe658bbec14c60e1d386d8ffa71d722127424bbc6fb317f`;
evidence-026 contains1277 files /33088620 bytes, archive SHA256
`6b0e1378e1a71b52020c89b81bc3a7f8341600f53e03350b5a3d59e6dc128d79`.
Both archives and all inventory entries are verified locally. Latest local
CPU/bootstrap run:250 passed in7.11s. All included trials/logs are terminal.

`57_k036_development.sh` extends identical seven-mode/full338 checks to the six
remaining registered development conditions c08/c15/c21/c23/c26/c28. It retains
32 training timing inputs x7 passes, not final validation timing. Expected
duration about three minutes, approximately USD0.04, with a95-minute controller
cap inside the existing20:57 UTC stop guard. The likely final policy disables
prefix reuse uniformly, not by checkpoint; this is not yet frozen. The original
35-checkpoint x3-process final study, all-site diagnostics and scientific PDF
remain outstanding; no manuscript or finding is promoted.

The final-study harness is implemented for preflight, not launched as final
science yet. It uses K036/no-prefix uniformly, eight matched modes (stock,
previous and new eager; the same three graphed; graph all-skips-disabled and
graph attention-skips-disabled),64 validation timing identities x7 passes,
three fresh processes per checkpoint, and full338 numerical coverage in every
process. The eager-native anchor remains the numerical reference. All35
checkpoints and failures remain in the matrix. `58_study_diagnostics.py` counts
actual BF16 operands once per checkpoint: all seven activation sites, row
occupancy, norms/near-zeros, native weights, operand-derived projection MMA
counts and instrumented attention MMA counts. Its V-only PV scalar count is
explicitly a lower bound, not a replacement for canonical FP16 R_model.

`59_study.py` records all raw paired samples and flat source snapshots to avoid
Windows path-length problems. `60_freeze_study.py` seals source provenance,
not authorization. `62_study_smoke.sh` first tests two8-block/4-timing-input
eight-mode cases including diagnostics under a separately named preflight
policy. The final policy is not sealed before this harness smoke passes.
CPU/bootstrap/math/study unit tests:253 passed in7.50s before final packaging;
the latest source-map-only change is not a numerical change. Estimated smoke
duration under30s, bounded by ten minutes /USD0.12 at the existing Pod rate.

All ten registered K036 development conditions pass full338 validation with
bitwise-identical logits in no-prefix mode. Evidence-027 is verified locally:
1458 files /39589398 bytes, archive SHA256
`36cb8e862a42b7c5bb35fe16f2185baa928912753772f33b8b75e82b2073cfde`.
The first study preflight rejects the raw vendor inventory's SHA before model
loading: local inventory is CRLF and remote inventory is LF. Normalizing only
line endings makes the inventories identical; all900 file paths, sizes and
hashes match. No GPU measurement occurred. The failed smoke wrapper also
stops at this source check. Code-029/archive and preflight-policy-001 preserve
that rejected identity. The retry seals the parsed dependency inventory and
verifies each actual dependency file, rather than its platform-specific JSON
formatting. `63_study_smoke_retry.sh` uses preflight-policy-002 and new smoke
attempt002 identities; the final policy remains unsealed.

Study smoke002 completes at both c01/c30: all eight modes pass the8-block
smoke, all new modes are bitwise native, and every diagnostic denominator
and attention counter conservation check passes. It is not full-validation
evidence. Evidence-029 is verified locally:1574 files /41760298 bytes,
archive SHA256 `19a29e6a5360c912e25642495f29f7c03ec77cb26efc2c7e5329e7bd76c946b0`.
Evidence-028 retained the preflight001 failures:1453 files /39522825 bytes,
archive SHA256 `a6b0123ab6f1f744536cd676f3d39b7df649a52342f6ebdefbbdd27065d1f6a2`.
The final policy now seals K036/no-prefix uniformly. `64_final_matrix.py`
executes35 variants x3 fresh processes in fixed seed2504 randomized order
within each replicate. Every process times64 validation identities x7 passes
and validates all338 blocks. Diagnostics cover all338 blocks in replicate1.
Numerical failures remain in the matrix, not grounds for selective fallback.
Each leaf has a600s timeout; the whole controller has a90-minute timeout,
additionally bounded by the20:57 UTC independent Pod stop guard. At19:22 UTC,
the existing RTX5090 Pod is running atUSD0.69/h, with40GB container and40GB
Pod-volume storage. Expected duration70min /USD0.81 additional GPU cost;
all new work remains within the user-approvedUSD20 envelope. Logs, raw timing,
quality, counters, manifests, source snapshots and all failures will be
copied and hash-verified before Pod termination. No new weights/data are
created; all35 original checkpoints are already verified locally. Monitor
read-only approximately every60s, with immediate attention to failures,
missing progress, cost/deadline risk or inconsistent counters. Local suite:
253 passed in7.38s. No manuscript or finding is promoted by this launch.

The final controller launched asPID23896 under the90-minute timeout. Code-031
archive SHA256 `51f1311e76468f8ffa7129d92412295d170920b36dc3b3ae68c00d04d6ad3099`
and all161 source-overlay files /1854842 bytes were verified remotely before
launch. Final policy SHA256 is
`fa4415ad8fe8e6d0c852b60b7d095602a829b7d879c363219d7ff59860001441`;
all51 source and900 dependency records passed remote verification. Commit
`f536ad5` preserves the harness, frozen policy, ten-condition development
evidence and both smoke attempts. The reducer and figure scripts are present
but the final reduction/PDFs await all105 processes. Local suite256 passes
in7.17s, including paired-process reduction and smoke counter audits.

Source-audit clarification: the immutable diagnostic's compact phrase
"weight loads remain" is overbroad. K036 a/m CUTLASS loads weights before
the zero-fragment MMA test; K033 h/z loads weights inside the nonzero branch,
so its skipped fragments can also avoid those weight loads. The recorded
MMA counters themselves remain correct and are not memory-traffic counters.
The figures/captions use this more precise interpretation without rewriting
the frozen diagnostic or its source identity. Read-only monitor67 imports
only the standard library to minimize monitoring overhead during timing.

While that fixed matrix runs, K037 is prepared locally as a separate candidate,
not a modification to the frozen51 files. It retains K036's four projections
and K035's native two-split attention. Its only attention change returns from
the MMA helper before checking B fragments when all A fragments are zero,
and before the MMA loop when all B fragments are zero. The surviving MMA
sequence, gates, precision and zero-work definition are unchanged. This tests
whether redundant fragment checks explain part of the still-negative attention
benefit; a timing regression or changed outputs/counters refutes this candidate.
No new pruning or checkpoint-specific implementation selection is introduced.

`68_k037_components.py` compares30 synthetic/captured development cases with
native and K035, checking bitwise outputs and identical issued/skipped counts.
`69_candidate_comparison.py` compares K036/K037 and dense/skip controls on32 training
inputs x7 passes, with an eager-native anchor and full338 validation at the
four registered development endpoints c01/c11/c25/c30. All gates, source
checkpoints, data, seed, optimizer provenance and post-hoc scope remain fixed;
there is no training. `70_k037_development.sh` refuses to run before the105-case
matrix is terminally complete. Expected existing-Pod cost after the matrix is
about five minutes /USD0.06, with a20-minute controller cap /USD0.23, within
the approvedUSD20 envelope and subject to the existing stop deadline. It has
not been deployed or launched. Final-study observations remain unmodified by
this new development proposal, and all failures will be retained.

K038 is also prepared locally, isolating output-projection tiling rather than
attention: each warp reuses one gated h/z activation fragment for two adjacent
N8 MMA atoms (M32N16 CTA,64 threads), instead of reloading and rechecking it in
two separate output tiles. This halves that duplicated activation/check work
but lowers resident warp parallelism; either a gain or regression is plausible.
Every output accumulator keeps its original K16 order and BF16 rounding.
`71_k038_joint_probe.py` covers12 synthetic joint cases, followed by the same
four development endpoints and full338/six-mode comparison through
`69_candidate_comparison.py`. `72_k038_development.sh` likewise refuses to run
until the frozen matrix completes. Estimated additional five minutes /USD0.06,
with its own20-minute outer timeout /USD0.23 proposed inside the approved
envelope. Neither K037 nor K038 is deployed/launched yet. The local suite is
258 passed in7.27s, including exact output coverage for the new M32N16 mapping;
this is not CUDA performance or numerical qualification.

At20:28 UTC the same Pod price is reverified atUSD0.69/h and the independent
stop deadline extends to22:57:06 UTC, six total hours /USD4.14 GPU maximum
plus existing storage, within the approvedUSD20 envelope. New hidden guard
PID32840 is armed before original PID21920 is stopped. The extension helper's
immediate process-exit check races Windows asynchronous termination and throws
before its receipt write; subsequent read-only CIM checks confirm old PID absent
and new PID/command line/ARMED log correct. The explicit extension receipt
retains this infrastructure outcome. No second guard or Pod is created.

The frozen105-process K036 matrix completes in4309.38s with no infrastructure
failure. All35 variants and all new eager/graph/skip-disabled controls are
bitwise-native on all338 validation blocks in every process; previous K19+K18
qualifies only6/35 variants. Evidence-030 is locally hash-verified:7911 files
/305125906 bytes; archive SHA256
`ff4154c069274a04d027ba803db592aae932df4e795bce5b5d381ddbc0ad7d9f`.
The reducer audits all sources, pairings, gates, and pooled counts. Figures01/02
and their observations now preserve the entire cohort and all negative controls.
Both vector PDFs are rendered and visually inspected. At c30,R_model27.48268%,
matched graph speedup is1.37988x, net skip contribution1.01912x, and isolated
attention skipping0.98851x. The previous qualified graph remains faster.
This is a complete characterization, not completion of the optimization goal.

Code-032 SHA256
`fe10844e173ef83ace8c029ac2d214ec44a492b41ca92e29d5e4eb34ae08bec1`
and all179 overlay files /2004426 bytes are verified remotely after the fixed
matrix ends; all51 frozen source hashes remain unchanged. K037 launches under
20-minute timeout PID46319. Its detached parent-shell PID46318 initially keeps
the SSH output channel open; stopping only that verified wrapper leaves the
separate-session timeout/worker running under PID1. K037 completes30 component
cases with bitwise-native outputs and unchanged counters, followed by all four
development endpoints passing full338 checks in all six compared modes.
K038 subsequently launches under its separate20-minute timeout PID47503 after
K037 terminal evidence is archived/transferred; its outcomes are not yet known.
Local suite before these launches:258 passed in7.14s. Neither probe changes the
fixed K036 policy or its figures. Standard-library monitor74 is read-only.

K037 is numerically qualified but rejected for performance: its matched
K036/new ratios at c01/c11/c25/c30 are0.8520/0.8603/0.8553/0.8971. Early
returns alone do not reduce actual latency here. Evidence-031 is locally
verified:8188 files /307248427 bytes, archive SHA256
`c03c503dc886a465a521797dbf4b3d6fd9cd1813beb16f18be40105b7a6449ea`.

K038 passes all four full338 endpoints. Native/new graph speedups are
0.99666/1.05402/1.42511/1.43126 at c01/c11/c25/c30; K036/new ratios are
0.91113/0.91168/1.03396/1.03484. It helps sparse endpoints by3.4% but regresses
low-sparsity endpoints. Its no-skips/new ratios at c25/c30 are1.16171/1.16797,
while attention-only ratios remain0.98403/0.98953. These development results
do not establish that it beats the previous qualified high-sparsity graph,
and are not substituted for the complete K036 matrix or used to omit variants.

The next bounded probes K039/K040 change only h/z output tiling: four/eight
N8 MMA atoms per warp reuse the same gated A fragment, yielding M32N32 and
M32N64 CTAs respectively. More reuse may help the sparse endpoints but can
increase registers and reduce parallelism; numerical mismatch or a regression
against K038 refutes a candidate. Every output retains forward K16 accumulation
and the original BF16 linear/parallel-residual rounding. No gates, pruning,
model/data/seed/checkpoint/optimizer provenance, numerical tolerance, or
per-checkpoint dispatch rule changes. All six matmul paths remain present.
Scripts75/76 add12 synthetic cases and seven-mode,32-training-input x7-pass
timing with full338 validation at c01/c11/c25/c30, including both K036 and K038
graph comparators. Script77 executes each candidate separately. Proposed
existing-Pod envelope per candidate: expected three minutes /USD0.035,
20-minute timeout maximum /USD0.23, within the22:57 UTC guard andUSD20 budget.
All raw failures, timing, sources and quality are retained and transferred
before teardown. These two probes are implemented but not yet launched.

Evidence-032 is now locally verified:8435 files /311664200 bytes, archive
SHA256 `9fae658ea3dedc5cb1fa15633b680635b3311665a2d389813297f568e1cd0115`.
Code-033 SHA256
`fbd7e9172fa0cb58ff47dc831078534d848306925edcd7514c430f4cdc5d762d`
and all189 source-overlay files /2039774 bytes are verified remotely; the
original51-file frozen policy still verifies unchanged. Latest local suite:
260 passed in7.83s. K039 launches under20-minute timeout PID48726 after the
K038 evidence is closed and copied; K040 still awaits sequential launch.

K039 completes all12 components and all four full338/seven-mode endpoints.
Matched K038/new ratios at c01/c11/c25/c30 are1.00303/1.00357/1.00081/1.00205,
too small to call a strong improvement from one development process. Its c30
native/new ratio is1.43449 and no-skips/new1.16459. Evidence-033 is locally
verified:8690 files /316672889 bytes, archive SHA256
`7a08d6f5879ce2b39a381fd241abcf79e2fd2ec585cea1601c5ab93f7a6f39f5`.
K040 launches sequentially under20-minute timeout PID49776 after K039 closes
and its evidence transfers. Its outcomes are still pending.

K041 is a separately prepared attention specialization. Instead of requiring
an entirely zero Q or K operand fragment, it forms each fragment's16-bit
inner-dimension support mask and skips QK MMA atoms when those supports do not
overlap. This is necessary and sufficient for all scalar products in that
finite-input MMA atom to be structurally zero. Nonzero products and their
accumulation order remain unchanged. PV retains the original operand-empty
predicate; all four projections are K036, isolating the attention change.
It does not skip softmax merely because a score is zero, and does not add
pruning, weight changes,2:4 restrictions, or checkpoint-dependent dispatch.

The mapping is checked against the pinned CUTLASS BF16 traits and NVIDIA's
[m16n8k16 fragment layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type);
the mask reduction uses the documented
[warp OR intrinsic](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions).
CPU tests cover all256 A singleton coordinates against all128 B singleton
coordinates, signed zero,100 mixed patterns, and deliberately disjoint supports.
Script78 adds a disjoint-support GPU case to the prior30 component cases,
requiring bitwise-native output, unchanged PV counts, conserved total QK atoms,
and non-decreasing skipped QK counts. Script79 uses the same four development
checkpoints,32 training timing inputs x7 paired passes, six matched modes and
full338 validation. All original data/model/seed/gate/precision/quality/post-hoc
contracts remain fixed. Extra mask instructions or register pressure may
outweigh additional skipped work; no performance gain is assumed.
Script80 is prepared for sequential execution after K040, expected about
three minutes /USD0.035, with20-minute outer cap /USD0.23 on the existing Pod,
inside the22:57 UTC stop guard andUSD20 approved budget. All raw outcomes and
sources will be retained. Latest local tests:262 passed in8.00s. K041 has not
been deployed or GPU-tested, and no new final policy has been selected.

K040 completes all four full338 checks, but its K038/new ratios at c25/c30
are0.98779/0.98938; it is rejected as an improvement over the narrower output
tile. Evidence-034 is locally verified:8945 files /321677225 bytes, archive
SHA256 `3a7db6361128a9d88dab2d43643b02b4e03604514137eaa35616a0309f40fa0f`.
Code-034 SHA256
`b394d1a91398d4c350d38245fd3a5165c9514a99d646d9ccce80f77c6c63be5d`
and all198 overlay files /2156069 bytes are verified remotely; all51 original
frozen sources remain unchanged. K041 launches sequentially under20-minute
timeout PID50979 after the K040 archive is closed and transferred. Its31
component checks pass, including the disjoint-support counter assertion;
full-model checks are still running.

K042 is prepared as a separate a/m reuse probe: the CUTLASS CTA changes from
32x32x64 to32x64x64 and the warp from16x16x64 to16x32x64. The same16x8x16
instruction, K-order, bias epilogue, gates and zero-A criterion remain. It
isolates input-projection tiling atop K036; it does not yet combine K039 or
K041, or claim to eliminate additional logical zero products. Larger output
tiles may reduce duplicated loads but increase register pressure and reduce
parallelism. Scripts81/82 retain54 synthetic/captured training primitive
checks and the four registered development endpoints with32 training timing
inputs x7 passes, six matched modes and full338 numerical coverage. Script83
will run only after the current GPU probe closes. All scientific input,
validation, quality and post-hoc contracts are unchanged. Proposed existing-Pod
envelope is expected three minutes /USD0.035, capped at20min /USD0.23, under
the22:57 UTC guard andUSD20 budget. All raw results and failures will be
retained and verified locally before teardown. K042 is not yet GPU-tested.

K041 completes all four full338 endpoints but regresses against K036:
K036/new ratios are0.98163/0.98245/0.98334/0.98216 at c01/c11/c25/c30.
It is rejected as a speed improvement, while its extra zero-work recognition
and unchanged outputs/counter invariants remain useful negative evidence.
Evidence-035 is locally verified:9217 files /327248241 bytes, archive SHA256
`d058ed8644701ef4a64665ac363d26be3f628aae5b0bd16c49908dc89c08b9ec`.
Code-035 SHA256
`239b2dcc933286fca06cee1649c569757da0be00d9628c8a547498569d1b8afb`
and all203 overlay files /2175704 bytes are remotely verified, with the51-file
original policy unchanged. K042 launches sequentially under20-minute timeout
PID52276; all54 primitive cases and four full338 endpoints pass. Its K036/new
ratios at c01/c11/c25/c30 are1.01963/1.01105/1.02023/1.01957; c30 native/new
is1.41036. This is an input-tiling gain, not new logical opportunity capture.
Latest prelaunch local suite:262 passed in8.14s. Guard PID32840 and theUSD0.69/h
Pod rate are reverified after the original20:57 deadline; the new22:57 deadline
is active. No final follow-up policy is selected yet.

Scripts84/85 prepare a short CUDA-kernel profile of c30 training input0 for
native, previous, K036/K038/K039/K041/K042, ten warm graph replays each with
full logits and unchanged precision/shape. Every implementation must have
already passed its full338 c30 check, and its source hashes are verified.
CUPTI traces and per-kernel durations identify bottlenecks; they are explicitly
not substitutes for the uninstrumented paired study or end-to-end speedups.
No new weights, gates, sparse pattern or final-cohort selection is introduced.
The profile will run after the current trial evidence is closed and transferred,
expected under one minute /USD0.012, bounded by ten minutes /USD0.12 within
the same22:57 stop guard andUSD20 budget. It has not yet been launched.

Profile001 completed in6.94s and is included in locally verified evidence037:
9591 files /345903485 bytes, archive SHA256
`f3f782ea403d226bf01d1fbbe47578de6807afed3220de41c00256d4fb7ca381`.
Across six layers the joint h/z kernels consume about37.6us for previous,
88.4us for K036, and69.3us for K038/K039. K042 input projections consume49.0us,
close to previous47.6us, versus61.4us for K036. These summed instrumented
kernel times identify a bottleneck, not a replacement end-to-end benchmark.

K043 tests a narrowly bounded hybrid output projection atop K042 inputs and
K035 attention. Each h/z row with at most two nonzero values uses scalar
products with coalesced transposed copies of the same weights; other rows
retain the native-order BF16 MMA accumulation. The hybrid requires finite,
normal-range activation/weight values; unsafe values take the MMA path.
BF16 products fit FP32 exactly within the checked exponent range, and two
summands avoid reassociation among three or more nonzero terms. Numerical
qualification is still mandatory; this reasoning alone is not qualification.
The M16N64 CTA masks simple rows from its fallback MMA, preserves separate
linear/bias rounding and parallel-residual rounding, and exposes compiled-out
work counters. Counters distinguish MMA bypass (including SIMT substitution)
from the scalar products actually executed. This is not new pruning or a
change to the model, gates, data, seeds, precision or tolerances.

Scripts86/87/88 cover39 synthetic gate/range/mixed-density cases, independent
MMA/SIMT count conservation, count-disabled and skip-disabled equality, then
the four registered development endpoints using32 training inputs x7 paired
passes and all338 validation blocks. Timing retains native, frozenK036,
all-skips-disabled and attention-skips-disabled controls. No held-out timing
or checkpoint-specific dispatch selects the candidate. All current post-hoc
and checkpoint retention contracts remain. A useful result would improve
high-R latency without numerical failures; extra scanning and fallback
overhead could instead refute this implementation's usefulness. The expected
existing-Pod duration is roughly four minutes /USD0.046, bounded by20min /
USD0.23 inside the22:57 guard andUSD20 budget. GPU execution is not yet done.

Prelaunch CPU suite:264 passed in7.27s, including the hybrid counter reference.
One initial test expected16 times too many MMA atoms for32 rows; that test
constant was corrected before launch (the accounting implementation was
unchanged). Code037 SHA256
`9d22b09dc4f37aa35071dda4f32fc1d79998b1ffe79d3d8cceb825359387d821`
and211 overlay files /2212146 bytes are verified remotely. The51 original
frozen sources still verify. Pod priceUSD0.69/h and guardPID32840 are checked;
GPU was idle before launch. K043 launches at21:28 UTC under timeoutPID54098.
