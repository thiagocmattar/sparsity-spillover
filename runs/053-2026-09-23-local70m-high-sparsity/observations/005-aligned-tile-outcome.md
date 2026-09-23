# Aligned tiles preserve tested outputs but miss the moderate speed margin

Question: can wider output tiles make position-preserving K16 skipping fast
enough while avoiding the feature-compaction path's numerical discrepancy?

Method: four configurations, row groups16/32 and output widths128/256, at the
same .5/.05/.1 checkpoints and32 training blocks split16/16. Same five paired
passes,20 complete operator calls per graph, three dense/prior controls,
matched skip-disabled variants, numerical bounds and joint5% shortlist rule.
These reused prefixes are adaptive development data. No figure generated.

Result: all96 operator cases,32 changed-input graph checks and13,824 real-input
numerical comparisons pass. All9,216 real-input comparisons of the four new
paths and their skip-disabled variants are bitwise equal to the reference.
This is component evidence on the training prefix, not full-model validation
or a proof of equivalence for arbitrary inputs.

No configuration passes the joint moderate-endpoint shortlist. The closest
case is tiles_m32_n128 at z.1, with worst ratio0.9629 (3.71% less time, below
the required5%). It clears5% separately at .1 for z.1 and z.5, but nowhere at
.05 across both prefixes. All four variants clear5% for every z site at .5.
No aligned-tile full-model candidate is promoted.

The measured work counters explain a concrete limitation of this layout.
At z.1, the32-row aligned path executes74.77%/62.67% of potential reduction
work at .05/.1. The32-row feature-union path in the first screen executes
14.10%/12.39%. Many individual zeros share tiles with a surviving activation;
the aligned path must still process those occupied tiles. These normalized
software reduction counts are not measurements of hardware FLOPs or DRAM
traffic. The32-row/256-output aligned consumer also reports two spills at
each site; the best128-output variant has none. Spilling cannot explain the
whole negative result, since the spill-free variant also misses the rule.

The stage completes in287.1seconds. All648 returned files /58,256,070bytes pass
local SHA256 and size checks. The archive includes617 new/changed Triton-cache
or C++-extension files /40,752,691bytes. The earlier verified2,303-file Triton
snapshot supplies the unchanged base. All worker states are terminal; local
GPU benchmark work has ended and the WSL environment remains available.

This bounded local iteration tested12 configurations across three component
screens and one full-model candidate. It demonstrates useful local resource
fit and diagnoses two obstacles: feature-compaction rounding can propagate
through hard gates, while coarse aligned support leaves much more work at
moderate thresholds. It does not establish the requested qualified kernel,
statistical sparse contribution, or larger70M-versus14M latency drop.

Next priorities are a precision-preserving fine-grained path (or a measured
correction/fallback for ambiguous rounding), a matched dense consumer without
metadata overhead, and an ablation disabling only the newly introduced sparse
sites. A qualified candidate then needs interleaved checkpoint timing within
processes, independent process replication and a matched local14M comparator.
The current dense-control timings vary substantially between checkpoints;
their raw sequential-process kappa difference is not evidence of sparse
scaling. No more GPU work is left running by this stage.

Sources: scripts21--25, `aligned-tile-config.json`,
`results/aligned-001-summary.json`, `results/aligned-001-table.md`,
`results/aligned-001-retrieval.json`, plus observations002--004 and their
source-hashed artifacts. No manuscript text or consolidated finding changed.
