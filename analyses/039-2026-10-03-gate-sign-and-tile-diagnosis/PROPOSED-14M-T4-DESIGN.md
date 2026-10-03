# Proposed 14M T4 gate sign experiment

**State:** concrete proposal awaiting design confirmation. The user requested
a compact 14M RunPod experiment on 3 October 2026 and explicitly deferred tile
pressure. No real numbered run folder or experiment code has been created.

The latest request supersedes the earlier T2-only idea: use T4 gates at
a,m,h,z, compare unilateral versus bilateral gates after the two branch
LayerNorms, and use T4/Ph plus a second pressure recipe. P-o is provisionally
interpreted as Pall below; the user must clarify that label. If it means
pressure only at a/m or no pressure, this design will be revised before
implementation.

## Question and comparison

Does preserving large negative activations at the post-LayerNorm a/m sites
improve the quality versus sparsity trade-off of the studied T4 intervention,
with either h-only or four-site orthogonal L1 pressure?

Proposed thresholds: **kappa 0 and 0.1**, shared across active sites.
Zero gives the clean operator distinction: bilateral a/m is identity, while
unilateral a/m removes negative values. Kappa 0.1 tests a positive central-band
cutoff and matches an existing moderate T2/Ph endpoint. A third threshold is
not included in this proposal; adding one requires a revised agreed grid.

| Variant at each kappa | Gates at a and m | Gates at h and z | Pressure sites |
|---|---|---|---|
| T4/Ph unilateral | Unilateral | Unilateral | h |
| T4/Ph bilateral a/m | Bilateral | Unilateral | h |
| T4/Pall unilateral | Unilateral | Unilateral | a,m,h,z |
| T4/Pall bilateral a/m | Bilateral | Unilateral | a,m,h,z |

This is **eight fresh training conditions**, not a five-threshold sweep.
Fresh unilateral controls make the operator comparison part of this same run;
historical endpoints provide context rather than replacing those controls.
The design varies a and m jointly, so it does not identify their separate
contributions. Mixed unilateral/bilateral assignments between a and m are
outside this compact first experiment.

## Exact activation and pressure contracts

Use operational topology A4-Z in all six layers. a is the attention-branch
LayerNorm output immediately before QKV; m is the MLP-branch LayerNorm output
immediately before W1. These are distinct LayerNorm tensors after their learned
affine scale and shift. h is the FFN hidden gate output before W2; z is the
concatenated attention context after PV, before the attention output projection.
No Q/K/V gate is added. The h gate replaces GELU as in the historical recipe.

Unilateral gates retain x when x >= kappa; bilateral gates retain x when
abs(x) >= kappa. Equality survives. Detached comparisons give identity input
gradients to survivors and zero input gradients to rejected values.

Pressure remains orthogonal_l1 with lambda 1 and relative trust budget 1.
Ph captures exactly six post-gate h tensors. Pall captures exactly 24
post-gate a/m/h/z tensors. Average mean(abs(tensor)) uniformly over those tensors,
and validate the realized names at every optimizer boundary. Adding sites also
changes this objective's averaging; comparisons across Ph and Pall test complete
recipes. Within each pressure scope, the gate comparison holds that averaging
and target set fixed. No tile gate or group pressure is introduced.

## Matched model data and training

Use the executed Run041/044 model, data and optimization recipe. Pythia-14M
has six layers, hidden width 128, FFN width 512 and four attention heads.
Architecture and tokenizer are EleutherAI/pythia-14m-deduped at revision
7386d9a4ae45aef494a6e704910394def3037fc5. Construct the architecture from config;
initialize from the byte-verified historical random step-zero snapshot generated
with small_init/wang_init, not released or trained weights. Preserve explicit
random-initialization replay provenance and verify initial parameter hash
ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57.

Model and data-order seeds are both 1234. MiniPile revision is
18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0. Use all 1,000,000 training documents,
append EOS, no encode-time special tokens, int32 storage and length 2,048.
Training cache has 1,491,711,416 tokens, 728,374 complete blocks and a 1,464-token
tail, with SHA256 da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c.

Each condition receives 712 optimizer updates, global batch 1,024 sequences
or 2,097,152 input tokens per update, initially MB32 times accumulation 32.
The historical ordered schedule consumes 729,088 blocks, wrapping 714 blocks,
and its SHA256 is f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa.
Each condition sees 1,493,172,224 input tokens; eight conditions total
5,696 updates and 11,945,377,792 input tokens. The shorter experiment comes
from fewer conditions, preserving the per-condition horizon.

Fused AdamW uses betas 0.9/0.95, epsilon 1e-8, weight decay 0.1 excluding
biases and LayerNorm, task-gradient clipping at norm 1, peak/minimum LR
0.001/0.0001, 1 percent warmup and the original GPT-NeoX-v1 pre-step cosine.
FP32 parameters and moments, dynamic FP16 autocast, Flash SDPA, zero dropout,
and no activation checkpointing preserve the actual historical recipe.
The task-only gradient drives AdamW moments and the OL1 preconditioner;
the pressure correction follows the existing conflict projection and trust cap.
Valid completion requires zero skipped optimizer boundaries.

## Validation checkpoints and diagnostic retention

Ordinary validation follows update 1 and the reloaded final checkpoint.
Use all 500 validation documents, all 338 complete blocks, 692,224 input tokens,
691,886 prediction tokens, and report the 1,444-token excluded tail.
Cache SHA256 is 51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451.
Do not split this validation set into tuning and confirmation subsets.

Retain model checkpoints at 0,1,2,4,8,16,32,64,128,256,512,712 and full
optimizer/scaler/RNG recovery states at 256,512,712, including every final model.
Preserve cache identity, raw events, environment, source and transfer hashes.

Proposed measurements for later use, to be confirmed with the design:

- Full-validation pre-gate and post-gate a/m signed counts, exact/near-zero
  counts at epsilon 0,0.001,0.01 and the declared gate threshold; pooled sums,
  squared sums, absolute sums, RMS and L2 by layer. Record rejected count and
  squared magnitude for each actual gate, including discarded large negatives.
- Existing post-gate statistics for a,m,h,q_post,k_post,v,z and attention output.
  Save learned LayerNorm scale/shift values and all named parameter norms,
  including bias and normalization.
- Final full-validation integer row occupancy histograms and empty 8 by 16
  and 16 by 16 tile counters for a,m,h,z. Tiles stay within a sequence and
  group token rows and feature columns on a fixed aligned grid.
- Every-boundary task/pressure loss, gradient norm/dot/cosine/conflict,
  clipping site/threshold, realized pressure identities and full OL1 projection
  and trust-budget diagnostics. These cannot be recovered from final weights.
- Full-validation logical-product integers for all six operation families,
  R_block, R_model and the analytic A4-Z reach ceiling, with explicit units.

No extra post-hoc clipping sweep or specialized sparse-kernel benchmark is
proposed here. Historical K050 assumes the studied gates; changing its
threshold arithmetic would be a separate implementation and qualification task.
This experiment measures quality and structure, not hardware acceleration.

## Support refutation and limits

Compare bilateral minus unilateral validation loss within each matched pressure
scope and kappa, together with sitewise zeros, discarded magnitude, tile occupancy
and R_model. Bilateral points that preserve or improve quality at comparable
logical opportunity support the proposed trade-off. Consistent domination by
unilateral controls weakens that hypothesis. Lower loss with much less sparsity
supports an information-preservation effect but does not establish a better
quality versus compute frontier. The two-threshold grid need not produce exactly
matched zero fractions; do not interpolate unmeasured conditions as evidence.

Results remain a joint a/m, one-seed, fixed-budget 14M comparison. They cannot
establish a LayerNorm-specific causal mechanism, separate a from m, guarantee
tile acceleration, or generalize to autoregressive single-token decoding.
The submitted manuscript's targeted versus broad thresholding and co-design
discussion motivates the test. The new mixed gate mapping is stated explicitly;
no manuscript or promoted finding will change without a later user request.

## RunPod preparation and approval boundary

RunPod is the requested location. The unchanged MB32 workload historically
reserved approximately 57 to 62 GiB, so an 80 GB device is the starting point;
the new gate/pressure paths still require target-device preflight and headroom.
Use the existing pinned image digest
runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35
and the tested Python 3.12, Torch 2.11.0, Transformers 5.12.1 environment.

Read-only catalog checks on 3 October 2026 near 15:48 UTC returned Secure
A100 SXM 80 GB at USD1.59 per GPU-hour. One-GPU stock was MEDIUM, with LOW
stock in EUR-IS-1 among the returned locations. Four-GPU stock was LOW, with
US-MD-1 returned; the four-GPU read showed CUDA 12.8 unavailable and 13.0/13.2
available. Secure H100 SXM 80 GB was USD3.49 per GPU-hour with HIGH one-GPU
stock. These are current discovery facts, not a selected fleet or launch envelope.

The account read returned an existing EXITED Pod qa00sm05lmrs7w named
run052-sparse-attribution-001 and the existing 100 GB STANDARD network volume
9luykg5yc3 in EUR-IS-1. No resource was changed. The Pod's presence differs
from historical closeout text claiming its deletion; its ownership is outside
this conversation and it will not be reused or removed without instruction.

After design confirmation, create the next numbered run, implement only this
grid, verify hook/gate gradients and realized pressure captures, run focused
checks and the full bootstrap suite, and present the launch definition.
That definition must include the freshly selected GPU fleet, live billable
price, full end-to-end ETC, maximum duration/cost, storage, transfer inventory
and teardown guards. Before launch explicitly confirm any additional post-hoc
measurements. Detached execution must persist logs, monitor at the agreed
interval, copy and locally hash-verify all artifacts, then terminate only newly
created Pods and check for unintended owned billable resources.

## Remaining design confirmation

Clarify P-o, confirm the eight-condition matrix and kappa 0/0.1, and confirm
or extend the proposed retained measurements. These are scientific choices;
full filesystem access does not resolve their meaning. The repository's
AGENTS.md explicitly requires design confirmation before experiment code
or a real run folder, followed by a separate launch confirmation.
