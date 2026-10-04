# Do T7/Ph corner tiles improve the orange quality-latency frontier?

Question: place the two new14M T7/Ph corner checkpoints at kappa0/.1 in the
previous manuscript quality-on-X, latency-on-Y comparison, and compare their
runtime with matched scalar T7/Ph controls.

Method: X retains canonical native FP16 final-checkpoint loss, including all
historical coordinates. Y uses verified Run062 BF16 sparse synchronized-host
geometric means on one physical RTX5090:64 fixed inputs × seven interleaved
passes × three fresh processes per threshold,1344 paired samples. The two
retained scalar checkpoints are measured in the same processes, using the same
consumer kernels and common gate passes with scalar/corner modes. H/Z consumers
remain frozen from Run061; K050/K042/K035/K019 remain pinned. The new a/m and
q/k/v gate passes are timed. Historical orange gate fusion and sessions differ.

Coverage and verification: B1/T2048, full50304 logits, uncached. Each graph
qualifies against its own unchanged eager BF16 model over all500 validation
documents,338 complete blocks,692224 inputs and691886 predictions;1444 tail
tokens are excluded. All six scientific processes pass. All12168 model/graph/
block comparisons have zero maximum absolute error and zero relative L2 error;
pooled losses also match. The archive and80 returned files have verified byte/
SHA256 identities. Original FP16 quality is separate from BF16 qualification.

Caption: (a) logarithmic cross-scale overview retains the36 previous execution
points and appends four tile endpoints: two prior T2 diamonds in teal and two
new T7 plus markers in purple. Circles/triangles/squares identify14M/31M/70M;
gray Base markers are native filled/kernel hollow. Orange/blue curves are
historical scalar T7/Ph/T2/Ph. (b) linear14M detail also displays the two matched
T7 scalar references as hollow purple markers, excluded from frontier membership.
The gray dotted staircase is the observed nondominated set of40 endpoints.
Whiskers span the three process geometric means, not confidence intervals.

| kappa | Scalar canonical loss | Tile canonical loss | Matched scalar sparse (ms) | Tile sparse (ms) | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.219827 | 5.721071 | .635143 | .564432 | 11.133% |
| .1 | 5.237395 | 5.835102 | .556401 | .516416 | 7.187% |

Three-process tile ranges are.564240–.564585ms and.514983–.518140ms.
Paired scalar/tile ratios are1.125277x/1.077430x, consistent across all three
processes per threshold. Quality penalties remain.501244/.597707 NLL and
65.077%/81.795% perplexity. Empty-tile structure and surviving activity are
documented in observations005/006.

Neither new T7 endpoint expands the observed frontier; its15 members remain
unchanged. Both are dominated even within the historical orange14M series:

| New endpoint | Previous orange dominator | Previous loss | Previous latency (ms) |
|---|---|---:|---:|
| T7 tile kappa0 | Scalar T7/Ph kappa.1 | 5.237395 | .541844 |
| T7 tile kappa.1 | Scalar T7/Ph kappa.5 | 5.732049 | .473888 |

Historical scalar T2/Ph kappa.5 (5.536255,.506323ms) also dominates both.
Thus tiles improve runtime relative to their same-threshold scalar controls,
while this complete intervention gives a worse observed quality/latency trade-off
than existing checkpoints at the tested budget.

Caveats: historical dominance is arithmetic on separate sessions/kernel variants,
not a controlled comparison against scalar kappa.5. Coordinates are not normalized.
One training seed and one inference GPU do not establish equal-quality gains,
representational redistribution, portability, measured DRAM savings or decoding
speed. The no-skip control changes only h/z bypass and short-row substitution.
No manuscript edit or finding promotion is performed.

Artifacts: [PDF](../figures/07-t7-quality-latency-frontier.pdf),
[exact coordinates, Pareto set and source hashes](../data/t7-quality-latency-frontier.json),
[coordinate table](../results/t7-quality-latency-frontier.md),
[complete runtime comparison](../../../runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/latency/results/tables.md).
Source script: [04_t7_frontier.py](../04_t7_frontier.py); the exact numerical
summary is reproduced by [21_numerical_summary.py](../../../runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/latency/21_numerical_summary.py).
PDF rendered and visually
inspected after correcting a clipped annotation. All previous six PDFs remain
byte-identical; the manuscript figure is unchanged.
