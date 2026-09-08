# Kernel auto-research: implementation and evidence companion

This accompanies [the short systems subsection](kernel-autoresearch.tex).
It documents the frozen implementation and the 8 September 2026 audit, not
a new kernel or a promise that all historical candidates are correct.

## Primary resources

- [Run029 and reproduction instructions](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/README.md).
- [Main figure](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/figures/04-kernel-autoresearch-and-rmodel.pdf),
  whose plotted data, qualification and regression are unchanged by this audit.
- [Implementation/claim audit](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/06-implementation-and-claim-audit.md)
  and [independent machine-readable reconstruction](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/implementation-audit-001.json).
- [Sparse-path ablations and executed-work counters](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/03-matched-rmodel-speedup.md).

## What the final "kernel" contains

K050 is a specialized full-model execution policy, not one monolithic CUDA
function. The following paths are relative to the frozen
`runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/`:

| Component | Implemented behavior | Frozen source |
| --- | --- | --- |
| Two branch normalizations | Compute shared-input Welford statistics once; preserve separate affine weights/biases and existing a/m gates | `runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k050/{candidate.py,norm.cu}` |
| QKV and FFN up projections | CUTLASS BF16 tensor-core multiplication with warp-uniform exact-zero A-fragment bypass | `runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/{candidate.py,projection.cu}` |
| Q/K/V preparation | Fuse partial RoPE, existing symmetric gates and head-major layout, retaining intermediate BF16 roundings | `runs/026-2026-09-06-pythia14m-fused-sparse-kernel/autoresearch/candidates/k019/{candidate.py,kernel.cu}` |
| Causal attention | Two-split FlashAttention-derived QK/softmax/PV; bypass MMA atoms with zero A or B fragments while preserving causal masking and softmax normalization | `runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/{candidate.py,kernel.cu,sparse_gemm.h,flash_fwd_kernel.h}` |
| FFN down/output projection and residual | At most two nonzeros per eligible row use SIMT; remaining rows use tensor-core MMA with empty K16 tiles bypassed; retain BF16 linear/residual roundings | `runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/{candidate.py,joint.cu}` |
| Full-model scaffold | Refresh embeddings, execute every block, final normalization and dense 50,304-way LM head | `runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/45_graph_forward.py` |

Run029's `replay.py` installs this composition from frozen sources. The
actual K050 policy sets `shortcut=False`: the experimental zero-query
prefix shortcuts are not active. The no-skip control uses the same fusion
with `skip=False, projection_skip=False`; the attention-dense control changes
only to `skip=False, projection_skip=True`.

### Actual sparse primitive used by the final policy

Verbatim operator excerpt from the frozen K042 `projection.cu`, inherited by
K050. This is a snippet, not a standalone compilable or complete kernel:

```cpp
CUTLASS_DEVICE void operator()(FragmentC &d,FragmentA const &a,FragmentB const &b,FragmentC const &c) const {
    const unsigned* words=reinterpret_cast<const unsigned*>(&a);
    bool zero=((words[0]|words[1]|words[2]|words[3])&0x7fff7fffu)==0;
    if(__all_sync(0xffffffffu,zero))d=c;
    else Base::operator()(d,a,b,c);
}
```

The bit mask ignores the sign of BF16 zero. The entire warp agrees whether
the activation fragment is zero; if so, the accumulator is retained without
issuing that MMA operation. A partially sparse fragment still follows the
tensor-core path. This explains why scalar logical sparsity is an opportunity,
not the fraction of executed instructions eliminated. The h/z path additionally
uses sparse-row SIMT, so its MMA bypass counts include substituted computation.
Finite operands and the declared layout are part of this interpretation.

The implementation uses existing CUDA, CUTLASS, FlashAttention and PyTorch
building blocks; it was not written independently of upstream software.
The frozen tree retains the corresponding licenses, including
`candidates/k050/LICENSE-PYTORCH` and the FlashAttention/CUTLASS vendor licenses.
The early Sakana-derived P0 adapter is not an unchanged upstream benchmark.

## Measurement and numerical contract

- Model: the 35 retained Pythia-14M checkpoints, one training seed; no
  retraining or post-hoc alteration of their gates in this evaluation.
- Runtime: one physical RTX5090, BF16, batch 1, 2,048 tokens, no KV cache,
  no padding/arbitrary attention mask, full 50,304-vocabulary logits.
- Denominator: same-checkpoint native PyTorch/SDPA using the shared CUDA-graph
  scaffold. Unmodified eager native output anchors numerical checks, not timing.
- Each timing point pools 64 identical validation inputs, seven randomized
  paired passes and three fresh processes: 1,344 native/candidate ratios.
  The statistic is their geometric mean, not a ratio of median latencies.
- Compilation, static weight layout preparation, graph capture and equal
  input staging are excluded. Activation-dependent preprocessing and full
  model execution are included. No activation or logit cache is reused.
- Qualification covers all 338 complete blocks of 500 validation documents;
  1,444 trailing tokens are excluded. All outputs must be finite, every logit
  must satisfy `abs(candidate-native) <= 0.25 + 0.02*abs(native)`, per-block
  relative L2 must be at most 0.02, and pooled loss difference at most 0.001.
  All three processes must qualify. This is bounded numerical agreement,
  not bitwise equivalence or proof for arbitrary inputs.
- Canonical `R_model` is the retained FP16 validation measurement, expressed
  as a fraction in the OLS equation and percentages on the plot. It measures
  training-induced sparsity post hoc, not runtime instruction savings.
  Projection credit counts zero activation operands, not zero weights;
  QK/PV count the union of their zero activation operands on valid causal
  pairs. The dense LM head is in the denominator without zero credit.
  Runtime BF16 counters are separate. In the manuscript this metric is
  denoted by calligraphic `S_model`.

## Results and attribution

The final historical kernel qualifies on 35/35 checkpoints: speedup range
1.005938-1.783175x, equal-checkpoint geometric mean 1.250205x. The lowest
gain is close to the noise floor. The matched historical incumbent ends at
1.783029x; its tiny difference from the final sweep is a separate measurement.

The independently reconstructed fit is
`speedup = 0.9531843003 + 3.8586397669 * R_model`, with `R^2 = 0.7812571630`.
There are 41 non-monotonic checkpoint pairs. Every leave-one-checkpoint-out
fit retains a positive slope (3.630-5.068), but the checkpoints are related
training conditions, not independent seeds; this is not an inferential
replication count or a causal identification design.

| Control/result | Same-cohort result | Meaning |
| --- | --- | --- |
| K050 vs native | 1.250205x geometric mean | Total implementation gain includes fusion |
| No-skip fused control vs native | 1.183457x geometric mean | Acceleration also exists without sparse skipping |
| K050 vs no-skip control | 1.056401x geometric mean; helps 19/35 | Positive but nonuniform sparse-path contribution |
| Same comparison at search checkpoint c30 | 1.311153x | 31.1% additional acceleration over fused no-skip; not 31.1 percentage points of total native speedup |
| Attention-skipping-disabled control vs native | 1.267131x geometric mean | Faster than K050 on all 35; skipping attention adds overhead here |

Even the no-skip control has a positive speedup--R_model association
(`R^2 = 0.496`). Therefore the main figure alone cannot attribute the complete
trend to sparse multiplication. Native-normalized ablations are measured in
separate randomized processes, not direct within-process toggles. Enabling
the hybrid path changes detection/selection and SIMT versus MMA execution
together, not only the count of zero multiplications.

"Best" means the best eligible historical proposal on the fixed c30 score.
The later attention-dense diagnostic ablation is not relabeled as a searched
candidate. P0 qualifies on only 5/35 final checkpoints; no qualified
all-checkpoint P0 average exists.

### Attention skipping and the longer-sequence hypothesis

K050 does execute attention skips: at c30, pooled counters show 57.9952% of
QK and 67.2084% of PV tensor-core MMA atoms bypassed. These are instruction
fragments, not fractions of all individual scalar zero-products. The
attention-dense ablation is faster on every checkpoint; skipping work and
reducing latency are different outcomes. The positive net sparse-path
contribution in this workload comes from the projection-side paths.

For a fixed architecture, uncached causal QK+PV logical work is `d*T*(T+1)`
per block, while projection and dense LM-head work scale linearly with `T`
(the manuscript's denominator equation). Longer sequences could therefore
make profitable attention specialization more consequential for full-model
speed. They do not guarantee that this skip mechanism becomes profitable:
`k035/sparse_gemm.h` repeats zero checks and warp votes within the attention
tile computation, while occupancy and zero-fragment patterns may also change.
No crossover length or longer-sequence speedup has been measured.

The frozen K035 attention wrapper explicitly requires `(B,H,T,D)=(1,4,2048,32)`
and allocates fixed-length buffers. This is not a switch that can safely be
tested simply by increasing `T`. A future test would need adapted kernels,
an explicitly declared longer-context model/data protocol, fresh numerical
qualification, recomputed logical counters, and matched skip-on/off timings.
This discussion is a hypothesis, not a new experiment or a context-extension
quality claim. It does not change the plotted K050 policy or its measurements.

## Known defects and scope limits

1. **Non-cohort threshold bug:** inherited h/z and Q/K/V fused gates compare
   BF16 values promoted to FP32 against an FP32 threshold. Native wrapped
   scalars first round to BF16. At `kappa=0.7`, BF16 `0.69921875` survives
   native gating but is zeroed by these fused comparisons. K050's a/m
   normalization pair already rounds the threshold correctly. The new audit
   exhaustively checks all 65,280 finite BF16 values at all actual cohort
   thresholds (`0, 0.01, 0.05, 0.1, 0.5`), for signed and symmetric gates;
   there is no such discrepancy in the figure's cohort. Frozen kernels were
   not patched; other thresholds require a correction and fresh qualification.
2. **Historical composition mismatch:** K044 instantiated the K033 base,
   not its intended K036 base. The retrospective deliberately benchmarks
   the executed code and records that distinction. K050 does not inherit K044.
3. **Not all candidates are valid:** 74 of 214 plotted history comparisons
   fail numerical qualification. They remain gray points but cannot update
   the qualified progress line. Two other history combinations are unsupported.
4. **Non-general interface:** cached transposed weights assume immutable
   checkpoints; persistent work buffers assume the fixed device/layout and
   non-overlapping execution. This is not a training/backward implementation,
   a concurrent serving engine, or a verified arbitrary-shape/nonfinite-input API.
5. **Scientific boundary:** different checkpoints have different losses,
   gate topologies and weights. Neither this figure nor its OLS proves a
   universal monotonic law, preserved task quality, agent superiority,
   strongest-compiled-dense superiority, or transfer to other model sizes,
   hardware, sequence lengths, batch sizes or cached token generation.

## Agent provenance and search accounting

The retained [controller record](../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/prelaunch/controller.json)
identifies an existing interactive Codex session, default model `gpt-6-astra`,
and default reasoning `high`. Per-request model/version, token counts and
served-model metadata are unavailable; configuration is not runtime attestation.
The human selected the scientific objectives, workload and approval envelope.
This supports an agent-assisted development case, not autonomous or
model-specific capability benchmarking against human/non-agent controls.

The retrospective numbers 42 eligible 14M proposals from the historical
K001-K050 lineage. Eight proposals targeting other sizes or standalone
primitives are archived but excluded. K011's 12 masks share one ordinal;
53 eligible configurations plus P0 are benchmarked on four checkpoints.
These are not 42 independent trials or the number of agent messages.
K003/K004/K005 first receive common-protocol full-model measurements in the
retrospective; the curve is not their original online feedback trace.

## Reproduction and audit boundary

Start from Run029's frozen `archive/root`, input identities, candidate catalog,
and `replay.py`; do not install the K050 wrapper alone without its dependencies.
The recorded runtime is PyTorch 2.11.0, Transformers 5.12.1, CUDA 12.8 and
NumPy 2.5.0. The Run029 README owns the exact installation, benchmark and
artifact verification workflow. A new GPU run requires its own execution record.

The current audit verified 5,175 file identities, all 1,173 process outcomes,
all 522,816 paired timings and all 391 grouped comparisons. It parsed 79
candidate Python source files, checked all eligible installation interfaces,
reviewed the final kernel's executed CUDA paths and FlashAttention changes,
and reran CPU regression/component tests. It did not rerun GPU benchmarks,
run CUDA sanitizers, or formally prove every historical kernel correct.
Raw full logits were not retained; numerical re-audit uses the complete
stored per-block diagnostics plus source review. The distinction is explicit.
