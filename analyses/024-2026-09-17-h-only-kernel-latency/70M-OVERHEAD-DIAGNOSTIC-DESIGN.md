# Proposed 70M implementation-overhead diagnostic

Status: both-checkpoint design approved by the user on 20 September 2026.
The user additionally authorized kernel optimization to seek speedups over
native PyTorch. Implementation and the launch packet follow; cloud launch
has not yet been approved for this new run.

## Question and hypothesis

Which implementation components explain the 70M specialized base model's
3.313 ms latency versus 1.664 ms for the native base model? Does that cost
remain on the fastest retained sparse checkpoint? The leading hypothesis is
that the inherited h/z dense fallback and input-projection schedule scale
poorly with width; additional sparse checks may also contribute. This is
latency attribution. The approved extension then uses measured bottlenecks to
guide kernel optimization, without changing training or model semantics.

## Matched models and protocol

- Two retained final step712 Pythia-70M checkpoints: T0/P0 and T7/Ph at
  kappa=0.5 (the fastest retained specialized endpoint). Resolve exact weight,
  configuration and validation-cache hashes from Run035. Checkpoint keys:
  the analysis reducer retains all identities in `data/paper-checkpoints.json`.
- Architecture: six layers, width512, intermediate2048, eight D64 heads.
  Training initialization/data order/seed1234, optimizer and token budget
  remain those of the existing runs. No optimizer updates or new training.
- T0/P0 has no gates or pressure. T7/Ph uses G+ at a,m,h,z and symmetric G+/-
  at post-RoPE q,k and v, kappa0.5; training used OL1 at h only, lambda=b=1.
  Preserve gates, their rounding/order, and weights in every execution mode.
- Same RTX5090 GPU class, pinned Run035 software/compiler/vendor sources;
  BF16, batch1, sequence2048, uncached causal inference, full50304 logits.
  Runtime seed2801; timing input/order seed2504 and deterministic replicate
  offsets. One active GPU job, with randomized recorded mode order.
- Every mode/process qualifies on all500 validation documents:338 complete
  blocks,692224 input tokens and1444 excluded tail tokens. Preserve bounds:
  logit atol0.25 + rtol0.02, relativeL2<=0.02, pooled loss difference<=0.001.
  Record actual errors and bitwise equality. A failed mode supplies no
  qualified speedup; shared implementation errors stop dependent modes.
- Existing64 validation inputs x7 paired passes x3 fresh processes. Pair
  each modified mode with the unchanged native implementation of the same
  checkpoint. Include an untouched full implementation in each process as
  an additional reference where memory fits; otherwise pair it in recorded
  adjacent processes. Decide and freeze this resource-dependent detail
  during implementation, before launch/timing.

## Nine execution modes per checkpoint

| Mode | Change from frozen full implementation | Identifies |
|---|---|---|
| native | Original PyTorch implementation | Same-session reference |
| full | None | Reproduce the observed deficit/gain |
| all-skips-off | Disable all sparse bypass/SIMT paths, retain custom layouts | Net cost/benefit of sparse machinery |
| hz-skips-off | Disable h/z inspection, skipping and scalar substitution only | h/z sparse machinery versus its padded fallback |
| native-hz | Replace joint h/z with native linear, bias and residual operations | Total conditional cost of the h/z implementation |
| native-am | Replace QKV and FFN-up with native linear operations | Input-projection implementation cost |
| native-attention | Replace QK/softmax/PV with native SDPA | Attention implementation cost |
| native-norm | Replace paired LayerNorm with native LayerNorm and identical gates | Normalization/fusion cost |
| native-rope | Replace fused RoPE/QKV postprocessing with native operations and identical gates | RoPE/layout/gating cost |

The five substitutions leave other custom components unchanged. Their effects
are conditional and need not add up: fusion, memory reuse and scheduling can
interact. Disabled skipping is not T0/P0; the checkpoint and gates still apply.
Native h/z substitution removes a fusion as well as changing matrix execution;
its measured difference must not be attributed solely to padding or reads.

## Profiling, retained diagnostics and interpretation

Record uninstrumented paired host and CUDA-event timings, process summaries
and ranges, complete correctness/loss records, gate settings and all identities.
Separately profile native and full modes on the first four predeclared timing
inputs after warmup, in each replicate. Retain GPU timelines, kernel names,
shapes and durations grouped into norm, a/m, RoPE/layout, attention, h/z,
final normalization/head, and remaining work. Profiler timings explain execution
structure; they are not substituted for the uninstrumented latency result.

Retain full-validation integer issued/bypassed MMA and scalar counts, row/tile
occupancy, exact/near-zero counts and RMS/L2 by site/layer, and weight norms.
Reuse matching immutable existing diagnostics where identical; validate the
new mode counters against actual operands. Source load-request counts are
not DRAM bytes. Do not claim memory-bandwidth attribution without hardware
counters. No post-hoc clipping or gradient measurements are added; training
gradient interactions cannot be recovered in inference. Preserve both final
checkpoints and validation-cache identities for any later diagnostics.

The h/z explanation is supported if native-hz removes a large reproducible
part of the baseline deficit and the timeline locates that time in h/z.
If hz-skips-off remains slow, the fallback/layout matters in addition to
inspection. Large native-am or native-attention improvements instead redirect
the explanation. If no component dominates, report distributed overhead and
interactions. A change that helps T0/P0 may hurt sparse T7/Ph; report both.

The paper claim affected is practical speedup relative to a common native
base per size. Report that separately from gains over a custom dense path
and over each checkpoint's native implementation. This experiment cannot
establish the best achievable 70M kernel or generalize to other shapes/GPUs.

## Approved optimization extension

Following the nine-mode diagnostic, develop changes on a fixed, recorded set
of training blocks, preserving weights, thresholds, sites and numerical bounds.
Start with native replacements where sparse paths lose time, then improve the
identified fallback, inspection or matrix scheduling bottlenecks. Preserve
every candidate source, development result and failure. Use a single frozen
implementation policy for both checkpoints; any dispatch must depend on shape
or actual operands, not checkpoint identity, validation loss or input identity.
Final full-validation results are qualification, not a tuning objective.

Freeze a candidate selected by development correctness and latency before the
full338-block evaluation and64-input final timing. Compare its measured gain
against the same-session native base, same-checkpoint native implementation,
and unchanged specialized reference. Do not multiply historical ratios or
predict gain from a faster dense reference. A numerically valid failure to
beat native PyTorch is an admissible outcome. Search duration/candidate budget
will be bounded in the launch packet; further scientific input changes require
a new record under the repository's append-only convention.

Next: implement in a new numbered run, run focused and
full bootstrap checks, then present GPU smoke scope, current price, ETC/cost
cap, exact resource plan, artifact transfer and teardown for launch approval.
Confirm additional later-use measurements before launch. No main.pdf rebuild.
