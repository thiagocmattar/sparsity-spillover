# Wider feature unions and tile support

Question: can wider row groups amortize sparse indexing enough to improve the
70M h/z projections, starting from a high-kappa positive control?

Method: unchanged T2/Ph checkpoints at kappa .5/.05/.1, RTX5070Ti Laptop,
BF16, all12 h/z sites, training blocks0:16 and16:32, five randomized paired
passes. Each timed CUDA graph contains20 complete operator invocations,
including gating and input-dependent metadata. Compare feature-union groups
of32/64 rows and support tiles of16/32 rows against three dense paths, the
previous selected sparse policy, and matched no-skip variants. This is a
component screen, not end-to-end model latency. No figure generated.

Coverage: all96 synthetic operator cases and32 changed-input graph checks
passed. All13,824 real-input numerical comparisons passed;69,120 raw timings
were retained. Independent integer reduction-work checks and count-on/off
output equality passed. No compiler-reported spills occurred. The bounded
stage completed in330.6seconds. All1,310 returned files /46,426,614bytes passed
SHA256 and size verification. This includes a complete Triton cache snapshot
with previous controls and environment checks, not only new kernels.

Result: no candidate meets the stated promotion criterion: at least5% faster
than the best qualified dense/prior control at both moderate kappas in both
training prefixes. The closest site is z.1 with union_m32: worst moderate
ratio0.9750, below the required improvement. No full-model candidate is selected.

The high-kappa positive control does work at component level. union_m32 is
at least5% faster than the best control in both prefixes at11 of12 sites.
For all six z projections, confirmation-prefix host ratios are0.471--0.522
(about48--53% less time); the matched no-skip paths are slower. These z paths
execute approximately5.1--6.25% of potential reduction tiles. This supports
the proposed high-kappa development strategy on this device.

At kappa .1, union_m32 clears the5% rule at z.1 and z.5 separately; confirmation
ratios are0.837 and0.729. At .05, no site clears the rule in both prefixes.
Even the .1 union path executes only12.4%/9.3% of potential z.1/z.5 reduction
tiles. Thus large logical work savings do not translate proportionally to
latency. Gather/layout, indexing, scheduling and residual dense work remain
costs. These measurements do not isolate their individual contributions or
measure DRAM bytes; counters are executed software reduction tiles, not
hardware instruction counts. The h paths generally lose to the previous
selected policy at moderate thresholds.

Next bounded variation: preserve this algorithm but increase output tile
width from64 to128/256 with row groups32/64. This reduces consumer CTAs and
repeated activation/index reads. It may sacrifice parallelism or increase
register pressure; those tradeoffs must be measured. Keep the same numerical
bounds, timing coverage, controls and promotion rule. Reusing these training
prefixes for tuning does not create an independent holdout.

Caveats: a single process per checkpoint, local laptop/WSL timing, training
prefixes rather than full validation, repeated fixed-input graph throughput,
and adaptive tuning. No significance, manuscript, cross-size delta or final
baseline claim follows. Moderate sparsity is not proved unexploitable.

Sources: `06_component_operators.py`, `07_component_screen.py`,
`09_reduce_components.py`, `results/components-001-summary.json`,
`results/components-001-table.md`, `results/components-001-retrieval.json`.
Source hashes, checkpoints/cache identities, activation counts and moments,
weight norms, occupancy, work counts, compiler resources and raw timings are
retained. Full-validation diagnostics and model-level profiles are still
required for any future frozen candidate.
