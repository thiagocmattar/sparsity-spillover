# 70M sparse-kernel families: source review and proposed experiment

Status: exploratory review complete. The user approved the design, launch and
GPU extension; implementation and execution are recorded in
[Run050](../../runs/050-2026-09-22-pythia70m-kernel-families/README.md).
This document retains the approved proposal. It introduced no training or
manuscript edit.

The next search should compare different execution algorithms, not extend the
Run049 row-capacity sweep. There is a measurable opportunity, but no evidence
yet that an alternative sparse algorithm will beat the best dense implementation.

## Evidence that sets the target

Run049's qualified efficient implementation uses dense h/z GEMMs and takes
1.323351 ms at T2/Ph kappa=.05 and 1.324743 ms at .1. It already beats native
Base (1.642892 ms), but remains slower than efficient Base (1.254255 ms).
Thus, beating native Base alone no longer establishes a sparse-kernel benefit.

The retained graph profiles attribute approximately 203 microseconds per
forward to the six h and six z GEMMs. Gate comparison, fill and associated
copies add approximately 82 microseconds. Matching efficient Base requires
69.1 / 70.5 microseconds at .05 / .1. This is around one quarter of the profiled
GEMM-plus-gate budget. These are four-input instrumented profiles, not an
additive prediction of achievable speedup. A 10% reduction from the current
gated implementation would require about 132 microseconds, nearly half that
budget. The LM head alone contributes about 431 microseconds in these profiles
and is outside this proposed sparse search.

Reproduce the profile reduction:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/036-2026-09-22-70m-kernel-family-exploration/01_profile_audit.py
```

See [the source review](observations/001-source-review.md),
[profile evidence](data/profile-audit.json), and
[source identities](data/source-inventory.json).

## Proposed design for Run050, if approved

**Question.** Can a lossless implementation of the existing h/z gates and
projections at moderate kappa exploit the observed zeros more efficiently than
the current dense h/z replacement, and beat an equally optimized ungated Base?
The specific hypothesis is that the current failure reflects packing,
register pressure, reduction order and weight-reuse costs, not insufficient
zero counts alone. Several algorithm families must get a bounded first trial.

**Matched scientific inputs.** Reuse Run049's hash-pinned c00 Base, c24 T2/Ph
.05 and c25 T2/Ph .1 final checkpoints at step 712. They descend from the same
seed 1234 random-initialization pretraining recipe, 1,493,172,224 MiniPile input
tokens, and original AdamW/OL1 history. No optimizer, backward pass, new
initialization, parameter update or retraining occurs. Preserve Pythia70M's
six layers, D512, M2048, eight heads and vocabulary 50304. T2 keeps one-sided
`x >= kappa` gates at h and z; h-only orthogonal L1 is a training intervention
already embodied in the weights. Equality, bias, BF16 branch rounding and the
two residual additions retain their existing operational meaning. No additional
pruning, magnitude thresholding, quantization or dropped overflow is allowed.

**Matched workload.** One RTX5090, the Run049 pinned software/runtime and
precision flags, BF16, batch 1, 2048-token uncached full forward and all 50304
logits, CUDA-graph timing. The exact checkpoint/cache/source identities carry
forward. Static weight layouts, compilation, capture and equal input staging
remain outside timing; every input-dependent gate, scan, pack, permutation,
metadata build, buffer reset, overflow route, reduction and scatter is inside.
No speedup from TEAL decoding or Sakana's default batch 64 is used as a forecast.

### Stage 1: characterize and establish stronger controls

Use training blocks 0--63 for development, with blocks 64--127 as a fixed
training-only confirmation set. Preserve the exact token-cache ordering and
hashes. No candidate or dispatch selection uses validation latency or loss.
Existing validation observations motivate this design and are not a fresh
statistical holdout.

For each site/layer and both kappas, measure:

- NNZ distributions per row and per 256-feature segment, including maxima and
  exact overflow counts for 31/63/127 payload slots;
- active-feature union and overlap across 1/4/8/16/32 consecutive rows, and
  empty tiles for K16/32/64/128/256;
- 0/1/2/3/4 nonzeros per contiguous four-feature group, plus the fraction of
  complete MMA tiles that satisfy 2:4 without deleting a value;
- gate, metadata, pack, sparse compute, overflow and output-combination costs,
  with actual operand bytes, issued work, register use and spills distinguished
  from estimated traffic. Measure hardware traffic/cache counters when the
  provider permits them; otherwise explicitly mark them unavailable.

Keep native PyTorch Base and same-checkpoint native references, Run049
native_hz, and opt073 as controls. Add a **fused-gate dense h/z control** that
retains dense tensor-core multiplication while removing avoidable gate/copy
and output-combination launches. This is necessary: the current 82-us gate cost
is comparable to the entire gap to efficient Base. Base uses the efficient
dense route with no gate/packing scan. Separate h-only, z-only and h+z
replacements expose which site actually benefits.

### Stage 2: a bounded search across four families

| Family | Concrete first implementation | Difference from prior work / failure risk |
| --- | --- | --- |
| A. Warp-streamed packed SIMT | Pack indices and BF16 values in small warp-distributed chunks; vectorized weight loads, FP32 products/accumulation, lossless overflow chunks or a dense row route. Separate h and z, and bound accumulator width. | Reuses TwELL's useful representation idea, but must improve on P0's 288 words per 256 inputs and K001's direct scan. A historical P0/K001 component is a control, not a new candidate. Packing and irregular weight reads may still lose. |
| B. Direct masked-load split-K | TEAL-inspired masked weight loads without compulsory global compaction; distribute K across workers and compare 1/2/4 splits, bounded output tiles and a deterministic FP32 partial-sum reduction. Adapt for 2048 rows and the original one-sided gates. | Different parallel reduction from K001's serial ascending-NNZ loop. Input rescans, partial buffers and loss of cross-token reuse can exceed savings; all are timed. |
| C. Gathered/tiled tensor-core SpMM | Build a shared active-K list for a small group of rows, gather the corresponding weight rows once per output tile, and multiply a padded compact activation tile using BF16 tensor cores. Skip empty tiles; use dense tiles where unions are too large. | K004/K007 already tried empty-tile bypass. The new axis is active-K compaction and weight reuse inside partially occupied tiles. Start with row groups 4/8/16 and K tiles 32/64; pad sub-warp row groups to the MMA shape and count that cost. Overlap diagnostics determine whether gathering is worthwhile. |
| D. Exact 2:4 sparse-MMA routing | Encode already-valid 2:4 activation tiles for `mma.sp::ordered_metadata`; route violating tiles through ordinary dense MMA into the same output computation. No magnitude-based selection and no discarded third/fourth nonzero. | A hardware sparse-tensor-core family, distinct from scalar skipping and gathered dense MMA. High global sparsity does not prove tile eligibility. Metadata, mixed-tile dispatch and SM120 code generation must be measured. |

Allocate at most six initial configurations to each family (24 total), at most
two library CSR-SpMM reference probes if the pinned BF16/graph contract is
supported, and at most eight refinements/integrations across the two best
qualified families. Keep all failures. This is a small run-local search, not
a general optimizer or a sweep over thousands of archived kernels.

Screen on real h/z operands from the training inputs, including all six
layers. Each trial checks packing/overflow and operator numerics before timing
pack-plus-compute. Use five paired development timing passes. Do not promote a
family solely on kernel-only timings. A family gets no refinement unless its
complete component path beats the corresponding dense control by at least 5%
on both development and training-confirmation aggregates for at least one
site/layer; a one-site winner is eligible even if the other site stays dense.
Freeze per-site/layer choices using the minimax latency ratio over both kappas,
with the dense control always available. Break timing ties in favor of dense.
This produces one shape/site/layer policy shared by .05/.1, not validation-
tuned or checkpoint-ID-based dispatch. Runtime decisions depend only on the
current input's measured tile/row occupancy, with their costs included.

Treat producer fusion as an integration option for successful families, not a
fifth sparse algorithm: gate/pack at the W1 or attention-context output can
avoid materialization and rereading. Do not port Hopper WGMMA/TMA blindly to
RTX5090. Preserve the original BF16 value before testing the gate, bias and
branch rounding. Fusion should not overwrite the non-gated up projection
contract with Sakana's two-up-projection gated-MLP architecture.

### Stage 3: freeze, qualify and attribute full-model performance

Freeze at most two sparse policies, the fused-dense control and the existing
references before final validation. Check **every** implementation against the
same-checkpoint unmodified eager model over all 500 validation documents:
338 complete 2048-token blocks, 692224 input tokens, 691886 prediction tokens,
with the 1444-token tail explicitly excluded. Keep Run049's original logit
bound `abs(error) <= .25 + .02*abs(reference)`, relativeL2<=.02 and pooled
loss difference<=.001. No threshold relaxation, rejected-input exclusion or
post-validation repair within the frozen scientific attempt.

Use 64 fixed validation timing inputs, seven paired passes and three fresh
processes per checkpoint. Rotate backend order; retain process-level effects,
paired samples and a process-stratified, input-block bootstrap interval.
Failed candidates remain in the report but cannot support a performance claim.

**Support:** at both kappas, a qualified sparse policy is at least 5% faster
than the best qualified dense-h/z control on the *same checkpoint*, with the
paired 95% interval entirely favoring sparse, and beats efficient Base in each
fresh process. Its Base route must stay within 2% of efficient Base. Report the
native-Base ratio too. An engineering target is about 1.19 ms using Run049's
session as an anchor (roughly 10% below current T2/Ph and 5% below efficient
Base); this is a target, not a prediction or a cross-session acceptance cutoff.

Use a same-layout skip-disabled ablation where meaningful, plus matched
fused-dense replacement, h-only and z-only comparisons. Report fusion and
sparse effects separately; speedup over a deliberately slow skip-disabled
kernel is insufficient. If only the fused-dense control wins, that is a useful
latency result but does not demonstrate a sparse-execution contribution.
No qualifying sparse policy, excessive representation cost, or numerical
failure refutes this bounded design's usefulness; it does not establish that
all possible sparse kernels are futile.

## Diagnostics, scope and execution budget

Retain the user's existing inventory: checkpoints/cache identities;
full-validation exact/near-zero counts, RMS/L2 and weight norms; row/group/tile
occupancy; h/z scalar, MMA and skipped-load opportunity counters; profiles;
raw host/CUDA timings; compiler logs, register/spill reports; every block's
numerical checks, pooled losses, failures and source/environment hashes.
Add the 256-feature, cross-row-support and 2:4 eligibility/overflow inventories
above. Collect full-validation structure after freezing, without retuning.
Original training gradient/OL1 diagnostics remain linked; inference cannot
reconstruct those measurements. This continues the already-confirmed inventory.

This tests the 70M runtime-realization claim in
`manuscript/draft/training-results.tex` and the h/z attribution in
`kernel-appendix.tex`. It preserves operational T2/Ph definitions and does not
edit the manuscript. The study covers these two existing checkpoints, one
training seed, one GPU SKU and full-sequence inference. T7/Pall generalization,
decode, additional kappas and retraining require a separate decision.

Planning allowance: up to four GPU-hours including compilation, diagnostics,
final passes and retrieval, with a checkpoint after the first GPU-hour before
spending the remainder on refinements. This is an uncalibrated ceiling for
design review; a representative smoke will determine ETC. Local implementation
and CPU checks precede a concrete launch request. The planned remote device
matches the prior RTX5090 session; the local laptop cannot establish that
same-device comparison. Refresh price, fit, storage, transfer, total cost and
deadline immediately before launch approval. Run049's rate was$0.99/hour,
which is historical context, not a new quote or authorization.

Reuse the retained Pod if still available under an explicitly approved lease;
do not silently extend its existing 20:30:06UTC compute-stop guard. The user's
instruction remains to retain the Pod and notify them after results; do not
delete it at experiment completion. Persist logs, run detached, monitor every
60seconds with progress/loss/throughput/ETC, and transfer/hash-verify outputs
before any agreed infrastructure change. New run implementation and launch
follow the two explicit confirmations in `AGENTS.md`.

## Exploratory verification

The profile reducer completed its projection-shape, launch-sequence and call-
count assertions for all three checkpoints. A second run reproduced its JSON
byte-for-byte. All referenced retained-source SHA256 values and Markdown local
links were checked. Only the analysis and index belong to this change; existing
manuscript edits remain separate. No CUDA correctness or speed claim is made
for any proposed family, and no experiment test suite was run for this review.
