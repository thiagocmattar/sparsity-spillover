# How much does h/z zero exploitation save at the 14M T2/Ph headline point?

On the fixed Pythia-14M T2/Ph, kappa=0.1 checkpoint, enabling both sparse paths
reduces latency from **0.641538 to 0.558789 ms: 82.749 microseconds, or 12.899%**.
This is a **1.1481x incremental speedup** within the same specialized stack.
Thresholds, h/z values, zero masks, weights and mathematical outputs are unchanged.

## Method and coverage

Use Run041's final c03 checkpoint, content SHA256
`5b17b232dff765910ea9d622218f23e437668ecfa0411bb5081ff611d6b2a548`.
Retain one-sided h/z thresholding at 0.1 in every forward. Change only the
Run037 joint kernel's independent h/z execution switches. A disabled site uses
the dense matrix path within the same fused kernel; it still receives the
thresholded activations. Both empty-tile skipping and short-row scalar execution
belong to the enabled sparse path. All other implementation settings stay fixed.

One RTX 5090; BF16, batch one, 2,048-token causal sequences and all 50,304 logits.
Three fresh processes each time A–D, untouched K050, and same-checkpoint native
PyTorch/SDPA in randomized paired order over 64 fixed validation sequences and
seven passes. Each backend has 1,344 host/CUDA timing pairs. Report geometric
mean full-forward host latency; exclude compilation, graph capture, static
weight preparation and equal input staging. All input-dependent gates and
sparsity checks remain timed.

Every process qualifies on all 338 validation blocks from 500 MiniPile documents:
692,224 input tokens and 691,886 predicted tokens, with the 1,444-token tail
excluded. No training, new checkpoint, threshold tuning or post-hoc clipping.

## Four-way control

Latency and conditional differences on the same checkpoint. Ranges are the
three process geometric means, not population confidence intervals.

| Configuration | h skipping | z skipping | Latency (ms) | Process range (ms) | Increase over A (microseconds) |
| --- | --- | --- | ---: | ---: | ---: |
| A: full execution | on | on | 0.558789 | 0.554875–0.561594 | 0 |
| B: disable h | off | on | 0.648529 | 0.644908–0.651165 | 89.740 |
| C: disable z | on | off | 0.562998 | 0.558981–0.567270 | 4.208 |
| D: disable both | off | off | 0.641538 | 0.636399–0.645749 | 82.749 |

The h path dominates: B−A is 89.740 microseconds with z enabled. The z path
saves 4.208 microseconds with h enabled; its process estimates are
2.852–5.676 microseconds. Joint savings are 81.524–84.155 microseconds across
the three processes, corresponding to 12.810–13.032% relative to D.

These effects are **conditional, not additive**. With h disabled, enabling z
increases latency by 6.991 microseconds (B−D). Thus adding B−A and C−A would
overstate the joint saving. D−B−C+A is −11.199 microseconds.

## Sparse contribution versus total implementation gain

Same-checkpoint native PyTorch takes **0.701788 ms**, giving a total native-to-A
reduction of **142.998 microseconds (20.376%, 1.2559x)**. The matched control
separates this gap as follows:

| Comparison | Saving (microseconds) | Share of native-to-A gap |
| --- | ---: | ---: |
| h/z zero exploitation: D−A | 82.749 | 57.867% |
| Net other implementation differences: native−D | 60.249 | 42.133% |
| Total: native−A | 142.998 | 100% |

The second row includes all differences retained in D, including fusion and
other execution paths; it is not a separately identified dense-optimization
effect. This decomposition uses the **same thresholded checkpoint and session**.
It must not be applied as a percentage of the paper's historical native Base
gap, which uses a different checkpoint/reference.

Untouched K050 takes 0.561414 ms, 2.624 microseconds (0.470%) above the all-on
control A. Their outputs agree exactly; this small implementation/control
offset is material when interpreting the much smaller z effect. Using untouched
K050 as the all-on latency reference gives an 80.125-microsecond joint saving,
or 12.489% relative to D. Preserve both references rather than pooling sessions
or silently treating their timings as identical.

## Verification and interpretation limits

- All logits have maximum absolute difference zero against the common eager
  reference, on every validation block in all three processes. Every backend's
  pooled BF16 loss is 5.15232672967722. The canonical FP16 training evaluation
  remains separate.
- The complete diagnostic pass verifies identical post-threshold BF16 values
  and zero-mask SHA256s at h and z in all six layers across A–D. Activation
  statistics also agree exactly. Disabled sites have zero bypass and scalar
  work counts; every instrumented h/z count matches its independent operand oracle.
- Forty direct CUDA cases passed, including active thresholding, equality at
  the threshold, and independent switch combinations. Eleven local focused
  tests, 242 bootstrap tests and real-checkpoint CPU installation checks passed.
- This is a conditional benefit of the implemented sparse paths, including their
  inspection costs and scalar substitutions. It does not measure an abstract
  cost per zero or establish performance on other checkpoints, scales or workloads.

## Evidence and closeout

Sources: [full reduction](../results/hz-latency.json),
[native-gap decomposition](../results/engineering-comparison.json),
[zero-pattern audit](../artifacts/attempts/final-r1-001/zero-pattern-check.json),
and the timing, quality and diagnostics files under `artifacts/attempts/final-*`.
Generating scripts: [07_reduce.py](../07_reduce.py) and
[11_engineering_comparison.py](../11_engineering_comparison.py).

All 85 scientific/source files and 22 terminal infrastructure files were
retrieved and hash-verified locally. Independent local reduction exactly matches
the remote JSON. Pod `nol9c179grw1vo` was deleted at approximately 12:08 UTC on
21 September 2026; the final Pod list is empty and the pre-existing shared volume
is unchanged. The GPU-time estimate is USD0.89; billing had not posted at closeout.
See [closeout record](../results/closeout.json). No manuscript changes were made.
