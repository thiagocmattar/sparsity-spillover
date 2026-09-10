# O001 - Introduction-facing 14M quality-sparsity trade-offs

## Question

How can the existing overview expose the effects of thresholding, pressure and
placement without recipe codes, while making the clipping comparison's
computational reach explicit?

## Method and coverage

Replot existing Analysis 018 trained endpoints and Run 030 post-hoc evaluations.
No new measurements. Of 30 Pythia-14M checkpoints, retain 26 and omit the four
naive-L1 checkpoints requested by the author. Retain all ten clipping settings
for each of baseline and ReLU (p=0, 0.1, ..., 0.9). The focused loss range is
5.03-6.23; six of ten clipping points per checkpoint are visible, with paths
continuing above the panel. All trained points are visible.

The compact layout, refined on 10 September, is 5.5 by 3.35 inches rather than
5.5 by 5.1. The legend occupies one row below the axes; the data-region height
is preserved. The plot retains three condition labels, two single-line ceiling
labels, the matched pressure statement and the 27.48% endpoint label. The
x-axis uses the manuscript's calligraphic S_model notation. One post-hoc
clipping label sits at the upper left near the gray path; the caption identifies
both dotted paths and reports the 91.8% utilization. The comparison
arrow and all leaders except the pressure-label leader are removed. Markers
remain 5.3 points, with thinner and lighter sweep lines. The manuscript retains
the first layout from commit `56d4ec6`; this revision is confined to Analysis 021.

Validation uses all 500 MiniPile documents, 338 complete 2048-token sequences,
692224 input tokens and 691886 prediction tokens, excluding the 1444-token
tail. One matched pretraining seed. Coordinates use full-precision losses and
ratios computed from pooled integer zero-product counts, never averages of
site percentages. The measured clipping p=0 losses are preserved independently
of the trained-endpoint losses.

## Labels and legend

| Source family | Figure label | Sites |
| --- | --- | --- |
| A0 | Baseline | Stock GELU |
| A1-H | ReLU | h |
| A1-H-OL1 | ReLU + pressure | h, with ReLU |
| A4 | 4-site | a, m, h, z |
| A4-OL1 | 4-site + pressure | a, m, h, z |
| A7 | 7-site (+Q/K/V) | a, m, h, q_post, k_post, v, z |
| A7-OL1 | 7-site + pressure | a, m, h, q_post, k_post, v, z |

Pressure is orthogonal L1, not naive L1. Projection sites include the FFN and
attention projections, so the A4 recipe and its ceiling are not labeled
FFN-only. All clipping uses a,m,h,z. Blue denotes projection-site thresholds;
orange adds attention operands. Open diamonds/triangles and solid lines denote
thresholds without pressure; filled markers and dashed lines add pressure.
Gray/green open circles on dotted paths denote baseline/ReLU post-hoc clipping.
The orange statement compares the matched threshold-0.5 endpoints. The curves
connect settings within each recipe; they do not depict this paired difference.

## Results

- ReLU plus pressure at weight 0.5 reaches 3.768% model-wide sparsity and
  validation loss 5.1102, versus baseline loss 5.2086.
- At the largest trained threshold (0.5), adding pressure to the broader
  recipe moves from 15.3868% to 27.4827% sparsity and loss 5.7029 to 5.8294:
  +12.0959 percentage points and +0.1265 loss.
- Projection-site reach is 2415919104 / 18825609216 = 12.8332%; adding the
  attention operands gives 5638717440 / 18825609216 = 29.9524%. These are
  scalar-product counts per sequence, including the dense final output
  projection in the denominator and excluding future-masked attention pairs.
  The 27.4827% endpoint is 91.8% of the 29.9524% seven-site reach.
- At clipping target 0.5, the ReLU checkpoint reaches 6.9945% sparsity at loss
  5.9763, compared with baseline clipping at 6.4419% and loss 6.0777. Both
  clipping references remain visible and are identified in the caption.

## Caption

**Quality-sparsity trade-offs for Pythia-14M.** Markers show 26 trained
checkpoints and post-hoc clipping of baseline and ReLU. Four-site thresholds
act at a,m,h,z, which feed FFN and attention projections; seven-site thresholds
also act at Q/K/V, with Q/K after RoPE. Pressure denotes orthogonal L1.
Solid and dashed paths connect trained settings without and with pressure;
gray and green dotted paths show post-hoc clipping of baseline and ReLU,
respectively. Vertical guides mark model-wide sparsity ceilings for the selected sites;
both clipping paths share the four-site reach. The orange annotation reports
the matched seven-site pressure addition at threshold 0.5:
+12.1 percentage points of sparsity for +0.13 loss. The resulting 27.48%
sparsity is 91.8% of seven-site reach. Lines connect settings within each recipe, not
fitted frontiers. Six of ten points on each clipping path lie in the displayed
loss range; the complete trajectories remain in the clipping appendix.

## Caveats

The wider attention recipe changes both trained representations and reachable
operations. Projection-site thresholding versus clipping shares the site set
but differs in training and threshold rules. Curves connect measured doses;
they are not fitted Pareto envelopes or a continuous training path. No jitter
or interpolated outcomes are introduced. The ceiling is analytic selected-site
reach; natural zeros elsewhere can exceed it. Logical sparsity is not measured
runtime. The highlighted pressure gain is a matched endpoint comparison, not
a claim of uniform improvement at every setting or across seeds.

The full 30-checkpoint recipe figure remains in Analysis 018 and the draft's
`figures/01-v2-14m-overview.pdf`; all 300 14M clipping measurements remain in
Run 030. No new finding is promoted.

## Source and verification

- Generator: [01_quality_sparsity.py](../01_quality_sparsity.py).
- Figure: [01-14m-quality-sparsity.pdf](../figures/01-14m-quality-sparsity.pdf).
- Full-precision selected evidence and source hashes:
  [14m-quality-sparsity.json](../data/14m-quality-sparsity.json).
- Sources: [Analysis 018](../../018-2026-09-08-results-materials/README.md),
  [Run 030](../../../runs/030-2026-09-08-all-models-posthoc-clipping/results/README.md).
- Focused verification: `test_overview.py`, all three tests passed. All 46
  measurements match the preceding version. The final compact PDF was
  inspected with Poppler. All figure fonts are embedded, text stays within
  the page, and the page dimensions are 5.5 by 3.35 inches. The legend sits
  outside the axes. The annotation values and caption's 91.8% utilization
  were checked against the source endpoints.
  No manuscript file changed during this revision.
