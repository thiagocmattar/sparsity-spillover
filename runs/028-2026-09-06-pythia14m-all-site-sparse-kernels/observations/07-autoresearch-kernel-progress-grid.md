# Auto-research progress with a light grid

## Question and method

On 2026-09-07 the user requested a grid on the existing single-metric
progress figure. Figure07 adds light horizontal and vertical major grid
lines to Figure06. It preserves all 136 graph-speedup points, exact integer
iteration positions, numerical statuses, fixed-c30 incumbent, labels, and
expanded linear y-axis. No new measurement, filtering, bar, quantile,
variant legend, or kernel description is added.

The previous PDF and generating script remain unchanged as append-only
evidence. Script `../127_plot_search_progress_grid.py` reuses the Figure06
builder after verifying its recorded source/script/figure hashes, then adds
the grid. The new provenance record is `../results/search-progress-grid-001.json`.

## Coverage, caption, and result

**Figure07. Auto-research progress with matched full-model graph speedup.**
Each point shows one model/checkpoint's native-graph / candidate-graph
paired geometric-mean latency ratio at its kernel iteration, K031-K050.
Blue circles denote passing numerical checks; the orange cross retains the
one failure. The orange step line is the cumulative best fully qualified
c30 speedup, ending at 1.739x. Light major grid lines aid coordinate reading.
There are no bars, boxplots, quantiles, or legend. The expanded linear y-axis
spans 0.70-1.86x without removing observations.

The unchanged source is `../results/search-progress-001.json`: 135 passing
points and one failure on Pythia-14M, RTX5090, BF16, B1/T2048, full50304
logits. The 66 development points use 32 inputs x7 passes in one process;
70 final points use 64 inputs x7 passes x3 processes. All cover 338 numerical
validation blocks from 500 documents, with the 1444-token tail excluded.
Checkpoints use training seed1234, step712. See [observation06](06-autoresearch-kernel-progress.md)
for the complete aggregation, selection, caption, and interpretation limits.

The grid changes presentation only: the final fixed-checkpoint ratio remains
1.738989802462756. It does not change the adaptive-history caveats, differing
cohort sizes/timing samples, or the distinction between sparse-path and total
implementation gains. No manuscript or finding is changed.

## Verification and related figure

`../test_search_progress_grid.py` checks all 136 exact point coordinates and
the incumbent sequence, a single axes, visible x/y grids, and absence of
bars or a legend. The PDF skill's render-and-inspect workflow confirms the
150-dpi complete render is legible and unclipped; vector fonts are embedded.
Output: `../figures/07-autoresearch-kernel-progress-grid.pdf`.

The separately requested cross-run history beginning at K001 belongs to
[Analysis017](../../../analyses/017-2026-09-07-kernel-progress-all-iterations/README.md),
not this run. It explicitly marks hardware/execution changes and missing
14M timings; it is not a homogeneous extension of this graph-only metric.
