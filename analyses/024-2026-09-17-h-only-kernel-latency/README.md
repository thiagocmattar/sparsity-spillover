# H-only pressure: final K050 latency extension

This analysis owns the requested extension of the model-wide-sparsity versus
full-model-speedup and absolute-latency figures. Only the five new Run032
A7+OL1(h) final checkpoints are measured in Run033. All 35 historical Run029
checkpoints, including A4+OL1(h), are reused from their retained raw pairs.
The 70M counterpart is now complete in Run035: all 22 retained checkpoints and
all 66 fresh processes qualify under one frozen K050-derived shape port. See
[Figure04](figures/04-70m-k050-port-sparsity-latency-topology.pdf), its
[caption and limitations](observations/004-70m-final-latency-topology.md), and the
[22-point table](../../runs/035-2026-09-18-pythia70m-k050-port/TABLE.md).
The [compatibility audit](70M-KERNEL-COMPATIBILITY.md) records why a port was
required. This is a new qualified implementation, not unchanged 14M K050 or an
equal optimization-budget comparison. The 14M results below remain unchanged.

**Complete and verified:** all 15 new processes qualified over all 338 validation
blocks. All 180 returned files passed byte-count/SHA-256 verification before
the Pod was deleted. `01_reduce.py` requires all 15 complete new processes;
`02_plot.py` additionally requires every plotted endpoint to qualify.

## Results and figures

- [Full-model speedup](figures/01-14m-k050-sparsity-speedup.pdf), with
  [observation and caption](observations/001-sparsity-speedup.md).
- [Native and K050 latency](figures/02-14m-k050-sparsity-latency.pdf), with
  [observation and caption](observations/002-absolute-latency.md).
- [Final K050 latency by topology](figures/03-14m-k050-sparsity-latency-topology.pdf),
  added on 18 September: 36 checkpoints (single-site naive L1 excluded), grouped as baseline, 1-site,
  4-sites and 7-sites, with [caption](observations/003-final-latency-topology.md).
- [Complete 40-checkpoint table](TABLE.md) and [full-precision data](data/results.json).
- [70M final port latency by topology](figures/04-70m-k050-port-sparsity-latency-topology.pdf):
  all 22 available checkpoints, four separate dashed pressure-family curves,
  and [caption](observations/004-70m-final-latency-topology.md).
- [Paired seven-minus-four-site distributions](figures/05-14m-paired-topology-effects.pdf):
  one row per pressure recipe, five matched kappa contrasts per row, and three
  panels for loss, logical sparsity and K050 latency. See the
  [observation and caption](observations/005-paired-topology-effects.md).
- [Paired OL1(all)-minus-OL1(h) distributions](figures/06-14m-paired-pressure-effects.pdf):
  four-site and seven-site rows, each with five matched kappa contrasts for loss,
  logical sparsity and K050 latency. See the
  [complete contrasts and caption](observations/006-paired-pressure-effects.md).
- [Pressure versus no pressure, 2-by-3 distributions](figures/07-14m-pressure-vs-none-effects.pdf):
  all-site OL1 on the top row and h-only OL1 on the bottom, each compared with
  no pressure within four- and seven-site topologies. See the
  [observation and caption](observations/007-pressure-vs-none-effects.md).

The five new A7+OL1(h) speedups are **1.2060, 1.3929, 1.4706, 1.5649 and
1.7839x**, in ascending kappa order. At kappa=0.5, h-only pressure has
16.663640% S_model and 0.473888 ms K050 latency. The retained all-site A7
endpoint has 27.482684% S_model, 0.473366 ms and 1.7832x speedup. These two
observed runtime endpoints are close despite different logical sparsity;
this is a descriptive comparison across two GPU/host sessions, not a formal
equivalence test or a causal decomposition of the pressure intervention.

The publication PDFs were rendered and visually checked; fonts are embedded.
Three focused aggregation tests passed. The historical 35-checkpoint raw-pair
reduction reproduces the saved speedups and source hashes. Input preparation
and prelaunch verification passed all 246 bootstrap/contract tests in Run033.
No manuscript TeX or historical figure was overwritten.

Figures01 and02 include all six A4/A7 pressure families at kappa
{0, 0.01, 0.05, 0.1, 0.5}, A0 and nine single-site controls. The clean figures
use no point annotations or fitted trend lines. Small speedup whiskers show
the range of three process means, not confidence intervals.

## Protocol and sources

- GPU: RTX5090; BF16, batch 1, sequence length 2048, full 50,304 logits.
- Full numerical validation: 338 blocks from 500 documents; 1,444-token tail
  excluded. Canonical FP16 sparsity pools integer logical-product counts.
- Speedup: geometric mean of all 1,344 paired native/K050 host-latency ratios.
- Absolute latency: geometric mean of the 1,344 raw host times. This agrees
  with the previous latency investigation's estimand; the original Run029
  summary's `native_ms`/`candidate_ms` instead aggregate process medians.
- Run029 and Run033 use different physical GPUs/hosts. Paired same-model
  native references preserve the speedup definition; cross-session effects
  remain a limitation, particularly for absolute latency.
- New skip-control ablations and a new A0 reference were not requested.
  Historical projection/attention-ablation panels stay in Analysis021.

`01_reduce.py` rechecks old source hashes and reconciles the raw-pair speedups
with Run029's saved reduction. It records every source hash and all exact
counts, qualification results, individual process means, runtime identities,
and timing block indices in `data/results.json`. `TABLE.md` exposes every
endpoint. Run033 owns source/input identity checks and verified retrieval.

Reproduce with `.venv/Scripts/python.exe analyses/024-2026-09-17-h-only-kernel-latency/01_reduce.py`
then `02_plot.py` from the same folder. Publication outputs are PDF only.

The final-kernel-only topology figure is reproduced with `03_plot_final_latency.py`.
It reads the unchanged retained reduction and saves its plotted points and source
hash in `data/final-latency-topology.json`. It uses the four explicitly listed
topology labels in the legend, with pressure recipes labeled directly on the plot.
Markers within each recipe family are connected in increasing kappa or lambda
order. A4, A4+OL1(all), A4+OL1(h), and their A7 counterparts have separate
curves despite sharing topology labels. This figure excludes the four single-site
naive L1 checkpoints at the user's request; its single-site group contains ReLU
and the four OL1 checkpoints. The original 40-checkpoint reduction and Figures01/02
remain complete. Curves are dashed, with small no-pressure/OL1(h)/OL1(all) labels
beside them and no arrows or label boxes. No fitted trend is added.

Reproduce the 70M figure with `04_plot_70m_final_latency.py` after Run035's
verified `16_reduce.py` reduction. It writes `data/70m-final-latency-topology.json`
with source hash, all plotted points, and each family connection. No 70M
pressure-free A4/A7 or single-site OL1 grids are imputed. All 705 Run035 outputs
are verified locally and its Pod is deleted. Only the four kappa .5 endpoints
beat their paired native graph references; speedups span 0.4997–1.1880x.

The paired distribution figure is reproduced with `05_plot_paired_topology.py`.
It joins the 30 multisite endpoints with Analysis023's uniform ordinary final
validation losses, checks matching checkpoint paths and integer sparsity counts,
and subtracts A4 from A7 within each pressure recipe and kappa. The boxes show
the middle 50%, median bars and full-range whiskers; dots retain all five
contrasts. They describe variation across the fixed kappa grid, not seed
uncertainty. Latency differences are in microseconds. The complete pairs,
source hashes and box statistics are in `data/paired-topology-effects.json`.
OL1(all) changes both gate and pressure scope; the h-only latency comparison
spans the Run029 and Run033 GPU sessions. No new timings are measured.

The pressure-scope distribution figure is reproduced with `06_plot_paired_pressure.py`.
It re-pairs Figure05's retained endpoints as OL1(all) minus OL1(h), keeping the
gate topology and kappa fixed, and checks the original reduction hashes.
The complete ten pairs and six box summaries are in `data/paired-pressure-effects.json`.
All-site pressure has higher loss in all ten pairs and higher observed latency
in eight; only kappa=0.5 has a slightly negative latency contrast in each topology.
Four-site timings share Run029; seven-site timings span Run029/Run033 sessions.

The separate two-row figure is reproduced with `07_plot_pressure_vs_none.py`.
It retains Figure06 and changes the reference to no pressure: top row OL1(all)
minus no pressure, bottom row OL1(h) minus no pressure. Each panel contains
four-site and seven-site boxes, each over five matched kappas. Scales are shared
between rows within each metric. The 20 contrasts from 30 unique checkpoints,
source hashes and 12 box summaries are in `data/pressure-vs-none-effects.json`.
Only the seven-site h-only latency contrasts span different GPU/host sessions.

The [combined 14M/70M latency figure](figures/08-14m-70m-final-sparsity-latency.pdf)
is reproduced with `08_plot_combined_latency.py`. It retains all 36/22 points
from Figures03/04, with separate dashed recipe curves, a common topology legend
and independently fitted linear axes in two panels. Source hashes, point and
connection identities, axis limits and the PDF hash are stored in
`data/combined-final-latency.json`. See the
[caption and comparison limits](observations/008-combined-final-latency.md).

The new [matched single-panel figure](figures/09-14m-70m-matched-sparsity-latency.pdf)
is reproduced with `09_plot_matched_combined_latency.py`. It selects the 22
recipe/kappa conditions present at both sizes: baseline, ReLU and the A4/A7
OL1(h)/OL1(all) grids. Open markers indicate 14M and filled markers 70M;
short/long dashed lines distinguish h-only/all-site pressure within each
topology and model size. Shared linear axes show absolute final-kernel latency
without forcing zero. The 44 plotted points, matching keys, source hashes,
integer counts and eight curve definitions are retained in
`data/matched-combined-latency.json`. See the
[caption and comparison limits](observations/009-matched-combined-latency.md).
Figure08 and all prior source data and figures are preserved.

The separate [logarithmic latency version](figures/09-14m-70m-matched-sparsity-latency-log-y.pdf)
is reproduced with `09_plot_matched_combined_latency.py --log-y`. It retains
the same 44 points, eight curves and marker styling, using a logarithmic y axis
with tick labels in milliseconds. Its provenance is saved separately in
`data/matched-combined-latency-log-y.json`; the script checks exact point and
curve equality with the unchanged linear version. Observation009 documents
both versions and the ratio interpretation of distances on the log axis.

The [A0-normalized speedup figure](figures/10-14m-70m-matched-sparsity-a0-speedup.pdf)
is reproduced with `10_plot_a0_normalized_speedup.py`. For each size it divides
the final-kernel A0 latency by each checkpoint's final-kernel latency, placing
both A0 controls at 1x. It retains 42 of Figure09's points and all eight recipe
curves, excluding the two 1-site controls at the user's request. Both axes are
linear, with subtle 14M/70M labels above their groups. This shared-A0 reference
differs from the paired native
reference used in the earlier kernel speedup figures. Exact baseline values,
original latencies, ratios and provenance are in
`data/matched-a0-normalized-speedup.json`; see the
[caption and interpretation limits](observations/010-a0-normalized-speedup.md).

The [quality-sparsity companion panel](figures/11-14m-70m-matched-quality-sparsity.pdf)
is reproduced with `11_plot_matched_quality_sparsity.py`. It restores both
1-site controls to Figure10's recipe cohort, yielding 44 points, and plots
ordinary final-checkpoint validation loss against model-wide sparsity. Colors,
open/filled markers, dashed recipe curves and model-size labels follow Figure10;
both axes are linear. The builder checks all losses against original metrics,
reconciles the uniform loss conventions in Analyses023/025, and verifies pooled
counts and checkpoint identities. Exact values and source hashes are in
`data/matched-quality-sparsity.json`; see the
[caption and provenance](observations/011-matched-quality-sparsity.md).

The separate [quality-sparsity figure with clipping](figures/12-14m-70m-quality-sparsity-clipping.pdf)
is reproduced with `12_plot_quality_sparsity_clipping.py`. It adds four dotted
post-hoc frontiers to Figure11's unchanged trained cohort: baseline and 1-site
at both 14M and 70M, with ten targets each. The full measured loss range is
shown. Each control's clipping acts at a,m,h,z, with its own retained p=0
measurement; no offset is applied to align different loss evaluation passes.
All 84 plotted records, source hashes and checkpoint checks are retained in
`data/quality-sparsity-clipping.json`; see the
[caption and measurement definitions](observations/012-quality-sparsity-clipping.md).
