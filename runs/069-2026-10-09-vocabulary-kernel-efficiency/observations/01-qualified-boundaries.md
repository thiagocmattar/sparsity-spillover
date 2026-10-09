# Qualified head implementation boundaries

Question: can the implementation penalty be reduced while preserving trained
weights, gates and output, and does that improvement survive full forward?

Method: retained Run068 14M checkpoints, head-input corner 16x16 kappa 0.1/0.5,
h/z corner 8x16 kappa 0, Ph+head training history, zero new optimizer updates.
Native dense GEMM, current N256, and at most two screen-selected implementations
run in three fresh processes per checkpoint on one RTX 5090. All use gated H_L.
Custom no-skip disables MMA bypass; it does not disable the trained threshold.

Coverage: all 500 validation documents through 338 complete 2,048-token blocks,
692,224 input tokens, 691,886 predictions and a 1,444-token excluded tail. All
14,196 mode/block comparisons are bitwise exact against the native anchor.
Timing has 64 real inputs x 7 randomized paired passes = 448 samples/mode/process.
The source checkpoint, cache, runtime and physical GPU identities reconcile.

Figure: [01-qualified-boundaries.pdf](../figures/01-qualified-boundaries.pdf).
Blue denotes native dense GEMM; orange custom MMA bypass disabled; green bypass
enabled. These colors are identical across thresholds and boundaries. Bars are
host-wall geometric means of equal-size process means. Whiskers show process
min/max, not confidence intervals. Dotted lines mark the corresponding native mean.
Reuse2 executes M32 blocks with two independent trained M16 masks; N is vocabulary
tile width. It does not change the trained tile-zeroing rule.

Result: best no-skip projection is 246.8/245.5 us, versus native 192.6/193.0 us
and current no-skip 318.1/318.2 us. The best no-skip gap remains 27-28%.
Best sparse full-forward times are 0.980100/0.968456 ms versus matched native
0.998643/0.998882 ms, reductions of 1.86%/3.05%. The best modes are reuse2 N256
at kappa 0.1 and prefetch N128 at kappa 0.5. Best no-skip and sparse modes differ.

Caveats: independently measured boundaries overlap and cannot be summed. Static
weight packing/casts, staging, loss, softmax and token selection are excluded.
Head-model quality penalties persist; the strongest existing body/frontier was not
integrated or remeasured. Winner identification is within the selected shortlist.
One GPU, one training seed and one workload limit generalization. Hardware counters
were denied, so no GPU-cycle decomposition is available.

Source: [comparison.json](../results/comparison.json), its six raw endpoint records,
[verification](../results/verification.json), and [figure provenance](../figures/provenance.json).
Generator: [14_figures.py](../14_figures.py). Interpretation: [13_diagnose.py](../13_diagnose.py).
