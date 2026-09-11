# Analysis 021 - Training-results figures

Rework the manuscript's figures from retained evidence, starting with an
introduction-facing Pythia-14M quality-sparsity overview. The author approved
this new analysis and the first figure on 10 September 2026. No new training,
checkpoint evaluation, or measurement is required.

## Figure 01

[Quality-sparsity trade-offs](figures/01-14m-quality-sparsity.pdf) retains
26 trained checkpoints, omits naive L1, and shows both baseline and ReLU
post-hoc clipping. Direct annotations highlight the local-pressure regime and
the matched high-threshold pressure addition. Vertical guides expose the
12.83% and 29.95% model-wide sparsity ceilings.

The compact revision is 5.5 by 3.35 inches (1.64:1), 34% shorter than the first
version. The one-row legend sits below the axes, preserving the preceding
version's data-region height. Inside the plot, labels identify baseline, ReLU,
ReLU plus pressure, the two ceiling guides and the 27.48% endpoint. A single
post-hoc clipping label sits near the upper gray path, and the orange statement
reports the matched pressure effect. The x-axis uses the manuscript's
calligraphic S_model notation. There is no internal title, comparison arrow
or utilization callout. Only the local pressure label retains a leader line.
Four-site thresholds act at a,m,h,z; seven-site thresholds add Q/K/V.
The caption defines these sites. Solid, dashed and dotted paths respectively
denote training without pressure, training with pressure and post-hoc clipping.
"Pressure" means orthogonal L1 throughout this selected view. Stronger markers
emphasize measured checkpoints over their connecting lines.

The orange annotation reports the matched seven-site pressure addition at
threshold 0.5: +12.1 percentage points of sparsity for +0.13 loss. The caption
identifies the gray/green post-hoc sweeps and reports that the 27.48% endpoint
uses 91.8% of seven-site reach.

Sources are the tracked Analysis 018 `figure_data.json` and Run 030
`results/clipping-points.json`. The compact
[figure data](data/14m-quality-sparsity.json) retain all 46 selected records,
pooled integer counters, coverage, recipe identifiers, and source hashes.
Eight clipping evaluations lie above the displayed loss range; their measured
coordinates remain in the figure paths and data. Earlier analysis artwork and
the manuscript's full recipe-level PDF are preserved.

After approving the figure polish, the author requested its adoption in the
introduction. The draft now embeds the approved PDF byte-for-byte as Figure 1
on page 2. The Pythia paragraph cites it, and the experimental section refers
back to the same figure. Its caption identifies both clipping paths and the
91.8% utilization. The rebuilt `main.pdf` has 26 pages. The earlier figure
adoption remains in commit `56d4ec6`.

## Reproduction and checks

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/01_quality_sparsity.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_overview.py -q
```

The focused checks cover exact source selection and coordinates, complete
validation, integer pooling, both unclipped p=0 measurements, and independent
ceiling accounting. All three checks passed. All 46 measurements are unchanged
from the preceding version. The compact PDF was rendered and visually checked;
all fonts are embedded. The annotation values, page dimensions, text bounds
and placement of the legend outside the axes were also verified.
See [O001](observations/O001-14m-quality-sparsity.md) for the caption,
results and interpretation limits.

## Figure 02

**Frozen by the author on 11 September 2026.** Final edits clarify the percent
axis, shorten the negative-dot annotation, and define both panels' summaries
in the caption. Layout and data remain unchanged from `444ad84`.

[OL1 conflict geometry and target-set-dependent saturation](figures/02-14m-ol1-geometry.pdf)
is a compact two-panel mechanism diagnostic, requested on 10 September 2026.
It uses every optimizer-boundary record from the five corrected four-site
Run 015 conditions and five seven-site Run 014 conditions: 7,120 observations,
with lambda = budget = 1. The 11 September revision makes the task-relative
opposing component rho_opp the primary diagnostic. Panel (a) shows its median
and interquartile range over 712 steps at each categorical threshold, multiplied
by 100 and plotted on a log percent axis. Blue circles and orange diamonds
distinguish the directly labeled target sets. Panel (b) shows five very faint
traces and one unsmoothed median per target set in the same colors. Direct
labels report each cap-active fraction, with "cap binds" at r/b=1.
There is no pooled trajectory, pooled cap annotation, legend or super-title.
No checkpoints or retrospective gradients are used.

Conflict occurs on 99.9% of steps in each group. Seven-site median cosines are
less negative at every matched threshold, while the cap is active on 97.0%
of seven-site steps versus 0.5% of four-site steps. The directions are nearly
orthogonal in magnitude but systematically negatively aligned; cosine alone
does not establish weak practical conflict. Saturation's mathematical independence
from lambda remains conditional on the cap binding. Post-projection
orthogonality is retained in O002 as an implementation check.

On 11 September, the author approved manuscript adoption. The frozen PDF
is copied without alteration into `manuscript/draft/figures/` and appears as
Figure 4 on page 7. Its caption defines the 25th-75th percentile whiskers,
per-step medians and implemented cap condition. The methods, Section 4.2,
Appendix A.2 and discussion now distinguish the target-set regimes instead
of treating lambda = 1 as universally saturated. The rebuilt draft has
26 pages; O002 records the adoption and verification.

The rho_opp diagnostic uses the retained pre-projection dot product
and squared task norm to measure the raw component removed along the task
direction, relative to its norm. Pooled medians are 0.009503 at four sites and
0.588006 at seven sites (about 62-fold separation). Seven-site medians are
higher at every threshold, and rho_opp exceeds one on 21.15% of seven-site
steps versus none at four sites. These are pre-cap, pre-learning-rate quantities,
not the magnitude of the applied update or measured loss effects.
[O002](observations/O002-ol1-geometry.md) gives the per-threshold medians and
IQRs. Across thresholds, medians span 0.25-1.21% of norm(u) at four sites and
19.78-84.78% at seven sites. The upper y-limit allows IQRs above 100% to remain
visible. No cosine or pooled fold-ratio annotation is plotted. The former
pooled ECDF figure remains in rollback commit `b2f46d0`; the per-threshold
cosine figure is preserved in `c2a2780`.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/02_ol1_geometry.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_ol1_geometry.py -q
```

All ten source event logs and manifests are tracked. The script verifies their
complete step coverage, the logged stabilized geometry, cap scaling, pressure
sites, and historical optimizer/pressure code hashes before plotting.
[The summary](data/14m-ol1-geometry.json) retains exact counts, per-condition
statistics and source hashes; the original logs retain each plotted scalar.
Five focused checks cover coverage and pooling, the stabilized cap boundary,
rejection of inconsistent geometry, the opposing-component formula against a
vector projection, percent conversion and full per-threshold IQRs, and exact
per-family traces. All eight Analysis 021 tests passed. The 6.4-by-2.8-inch PDF
uses the overview's fonts, font sizes and blue/orange colors. It was rendered and
visually checked; all text is inside the page and all fonts are embedded.
See [O002](observations/O002-ol1-geometry.md) for the caption and limitations.
