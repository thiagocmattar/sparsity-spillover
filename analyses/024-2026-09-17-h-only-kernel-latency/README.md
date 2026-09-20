# H-only pressure: final K050 latency extension

The approved [Run040 diagnostic and optimization](../../runs/040-2026-09-20-pythia70m-kernel-overhead/observations/001-overhead-and-optimization.md)
is complete and recovered. Replacing h/z with native operations removes
1.66ms from the original dense 70M path. A training-selected combination of
native a/m and attention with N256 sparse h/z reaches1.328912ms for T7/Ph at
kappa0.5: **1.195408x relative to native T0/P0**, with unchanged checkpoint
loss. The optimized dense path remains slower than PyTorch. This two-checkpoint
result does not replace the earlier grid or establish a scale trend; manuscript
integration remains a separate task. The pod was deleted within the USD6 cap.

The [compact results appendix](observations/043-compact-results-appendix.md)
retains all 74 endpoints in three tables and all 300 existing 14M post-hoc
evaluations in a restyled figure, while removing redundant historical displays.
The 410M stress-test artwork and discussion are preserved. Eight checks pass;
the manuscript is rebuilt at 20 pages with verified layout and references.

The [70M overhead audit](observations/042-70m-overhead-audit.md) rechecks raw
host/device timings and identifies padded h/z arithmetic, repeated inspection
and inherited projection schedules. The 1.649ms host deficit is also present
on the GPU (1.644ms). Its proposed component diagnosis is now completed in
Run040 above. Reproduce the original saved-data audit with
`32_audit_70m_overhead.py`; it predates the new optimization measurements.

The [consolidated kernel appendix](observations/041-kernel-appendix.md)
replaces the historical fit with a [fixed-native-base diagnostic](figures/20-kernel-structure-native-base-speedup.pdf).
All 84 matched trained/clipping settings retain their measured latency and
counts. The audit distinguishes native-base gains (best 1.427x/1.028x at
14M/70M) from specialized-base gains (1.418x/2.047x); the difference materially
limits the 70M scale claim. Seven tests pass; the manuscript is not recompiled.

The [base-model training figure](figures/19-base-model-optimization.pdf)
relabels the existing training trajectories T0/P0, preserving every plotted
value and the original style. It accompanies the compact experimental-setup
appendix and its clarified training table; see [Observation 040](observations/040-experimental-setup.md).

The [OL1 appendix figure](figures/18-14m-70m-ol1-geometry.pdf) now covers all
20 14M Figure 1 OL1 conditions, with a matched eight-endpoint T1/P1 L1--OL1 table.
The code-checked formulation, all 28,480 optimizer records, captions and
interpretation limits are documented in [Observation 039](observations/039-ol1-appendix.md).
Reproduce both artifacts with `29_ol1_appendix.py`.

The [two-panel 70M site-group figure](figures/17-70m-site-groups-sparsity-latency.pdf)
shows h/z contributions on the left (a) and the complement on the right (b).
The 20 T/P checkpoints share a latency scale and recipe legend, with focused
x-axis ranges and unchanged colors and threshold connections. See
[Observation 038](observations/038-70m-site-groups-latency.md) for the caption,
paired coordinates and source checks. Figures 15 and 16 remain unchanged.

New [complementary sparsity-versus-latency figure](figures/16-14m-complement-sparsity-latency.pdf)
uses the same 20 T/P settings as Figure 15, with the a/m/QK/PV contributions
on the x-axis. These sum with the h/z contributions to S_model at every point.
The original figure and latencies are preserved; see
[Observation 037](observations/037-complement-sparsity-latency.md).

New [14M h/z contribution-versus-latency scatter](figures/15-14m-hz-sparsity-latency.pdf)
uses Figure 1's 20 T/P settings and recipe colors, with Base and ReLU controls
omitted and axes focused on the pressure recipes. The x-axis sums only
the h/z contributions to S_model (pp), preserving the full-model denominator;
the y-axis retains the measured K050 full-model latency. Dashed Ph and solid
Pall lines connect increasing thresholds within each recipe. See
[Observation 036](observations/036-hz-sparsity-latency.md) for the caption,
exact source data and reproducible builder. No new timing or manuscript edit.

Completed short follow-up, Run039 after a retained Run038 fixture failure:
[extend the h/z load-avoiding strategy to a/m](AM-LOAD-AVOIDANCE-DESIGN.md).
One retained 14M T7/Pall kappa=0.5 checkpoint, five execution modes and matched
full-validation checks. Both design and launch were explicitly approved;
[Run039](../../runs/039-2026-09-19-pythia14m-am-port-fixture/README.md)
retains the implementation, checks and execution record. The port is correct
but slower: +17.23% latency at a, +27.38% at m, and +44.49% at both.
The [result and mechanism](observations/035-am-load-avoidance-port.md) explain
why the finer grouping and padded fallback do not help these a/m operands.
All evidence is recovered and hash verified; the Pod is deleted.

19 September manuscript update: the compact
[per-operation latency table](observations/034-operation-latency-manuscript.md)
now uses the completed Run037 ablations for 14M T7/Pall at kappa=0.5.
The subsection explains the kernel mechanism first, presents conditional
effects and process variation, then concludes with clear h/z savings and net
attention overhead. The earlier grouped-control table and generator remain
available as historical exports.

Earlier 19 September manuscript revision: a compact
[mechanism evidence table](observations/033-kernel-mechanism-table.md) replaces
the main-text bypass chart. It combines operation counters with three retained
timing modes for 14M T4/Ph and T7/Pall at kappa=0.5, and explains why h/z can
avoid weight reads while attention skipping retains operand loading and softmax.
All earlier figures and evidence remain available. The
[single-checkpoint per-operation attribution design](PER-SITE-LATENCY-DESIGN.md)
was approved at kappa=0.5 on 19 September and is now complete as
[Run037](../../runs/037-2026-09-19-pythia14m-operation-latency/README.md).
Its [direct operation-level evidence](../../runs/037-2026-09-19-pythia14m-operation-latency/observations/001-operation-latency.md)
reports 30 qualified processes, clear h/z benefits and grouped attention
overhead, with small individual a/m/QK effects unresolved against process
variation. All artifacts were recovered and the Pod deleted (estimated USD1.15).
These new measurements are now inserted into the manuscript table through
the adoption record linked above.
Older adoption notes below retain their historical figure placements.

The current [14M/70M quality figure](figures/14-14m-70m-quality-sparsity.pdf)
omits naive-L1, uses independent loss scales, and has a taller/narrower layout
with a three-row legend. Its 36/22 trained points and all clipping data are
documented in [observation 032](observations/032-quality-ol1-layout.md).
The four omitted conditions remain in the export; earlier notes retain
their original display scope and page numbering.

[Figure 14: 14M/70M quality overview](figures/14-14m-70m-quality-sparsity.pdf)
is now manuscript Figure 3, directly after Table 1. The existing Figure 13
speedup plot follows as manuscript Figure 4. The complete
[410M panel](figures/A5-410m-quality-sparsity.pdf) moves to appendix Figure 15,
beside the retained full 410M post-hoc trajectories. The split preserves all
original data records and PDFs; [observation 031](observations/031-quality-main-appendix.md)
documents captions, coverage, associated writing and the builder
`23_plot_quality_main_appendix.py`. Notes below describe previous placements.

[Figure 13: sparsity versus base-model speedup](figures/13-14m-70m-sparsity-base-speedup.pdf)
uses A3's size-specific optimized A0 references and Figure 08's matched cohort
and styling, including all 40 measured control clipping settings. The new
two-panel PDF, exact ratios and reproducible builder are documented in the
[observation and caption](observations/030-base-speedup-clipping.md).
It is now manuscript Figure 3, immediately after the post-hoc calibration
paragraph, with a first-principles account of agentic kernel development and
qualified interpretation of the sparsity/speedup and clipping relationships.
Manuscript figure numbers in older adoption notes below retain their original scope.

[Operation contributions](figures/04-operation-sparsity-changes.pdf) and the
existing bypass summary now form manuscript Figures 5 and 6 in the merged
Section 4.3 on sparsity and speedup. The operation figure is copied unchanged;
the text focuses on additional high-threshold QK/PV products and explains why
their bypass need not lower latency. See the
[integration record](../../manuscript/draft/reviews/2026-09-18-operation-sparsity-speedup/README.md).

The [two-panel operation-bypass summary](figures/12-operation-bypass-summary.pdf)
is now manuscript Figure 6, comparing T4/Pall and T7/Pall at kappa=0.5 across
six operations at 14M/70M. The rewritten kernel subsection explains avoidable
arithmetic, projection savings and the tested attention implementation's
limitation without numerical endpoint narration. See the
[caption and evidence](observations/029-operation-bypass-summary.md).
Reproduce with `21_plot_operation_bypass_summary.py`; the complete Figure 10
and its source measurements remain available.

[Figure 3's paired pressure analysis](figures/03-pressure-scope-threshold.pdf)
is now adopted unchanged as manuscript Figure 4 in Section 4.2. The lean text
compares all-site against h-only pressure across 14M/70M and reports the two
small 14M high-threshold latency reversals. See the
[adoption and verification](../../manuscript/draft/reviews/2026-09-18-paired-pressure/README.md).

New [all-model quality-sparsity overview](figures/11-all-model-quality-sparsity.pdf)
shows all 74 trained paper conditions in three panels (14M/70M/410M), with
uniform final-checkpoint loss, every executed recipe, analytic ceilings and
fixed-control post-hoc paths. It is adopted as manuscript Figure 3 in the
rewritten quality-sparsity subsection. See the [caption and evidence](observations/028-all-model-quality-sparsity.md)
and [data export](data/all-model-quality-sparsity.json). Reproduce locally with
`20_plot_quality_all_sizes.py`; four focused tests verify the reduction.

The new [per-operation bypass figure](figures/10-operation-bypass.pdf) and
[complete table](TABLE_OPERATION_BYPASS.md) cover 62 trained settings plus
40 control-clipping settings from saved full-validation counters. They compare
scalar zeros with matrix-instruction bypass across all six operation families.
Figure 10 now follows Figure 08's recipe colors, line styles and typography,
with twelve separate size/operation panels and one shared legend. The
[explanatory paragraph](observations/027-operation-bypass.md#explanation-for-later-text)
describes the h/z path, attention's retained costs and the attribution limits.
The [observation](observations/027-operation-bypass.md) also checks 35 matched
14M skip ablations: attention can bypass substantial work while remaining
slower. Reproduce locally with `18_reduce_operation_bypass.py` and
`19_plot_operation_bypass.py`; no new GPU pass is required.

**Selected manuscript Figure 1:** [Figure 8, 14M/70M quality and latency](figures/08-14m-70m-quality-sparsity-latency.pdf)
is adopted unchanged in the introduction. The caption, shared T/P legend
definitions, panel references and dependent manuscript scope now reflect both
sizes and measured clipping. See the
[adoption and build record](../../manuscript/draft/reviews/2026-09-18-two-size-main-figure/README.md).

New Figure 9 extends Figure 4 to a
[2-by-3 operation-contribution grid](figures/09-14m-70m-operation-contributions.pdf):
14M/70M rows and κ=0,.05,.5 columns, with one palette and separate row scales
(0-30 pp at 14M, 0-45 pp at 70M). It shows 24 retained checkpoints with only
Ph/Pall in every panel and the concise Y label "Contribution to S_model (pp)".
See the [caption and evidence](observations/026-operation-grid.md).
Reproduce with `17_plot_operation_grid.py`. The original Figure 4 is preserved.

New Figure 8 extends the Figure 6 design to a 2-by-2
[14M/70M quality-sparsity and latency view](figures/08-14m-70m-quality-sparsity-latency.pdf),
with one shared legend, 44 trained checkpoints and all 40 measured control
clipping latencies from Run036. Quality panels retain a focused range; their
high-loss clipping tails and timing-session limits are disclosed in the
[caption and evidence](observations/025-matched-quality-latency.md).
Reproduce with `16_plot_matched_quality_latency.py`. Figure 6 remains preserved.

Run036 supplies the missing final-kernel control clipping latencies: all 40
settings at 14M/70M and p=0,...,0.9 qualify across 120 processes. See the
[latency PDF](figures/07-controls-posthoc-final-latency.pdf),
[complete table](../../runs/036-2026-09-18-controls-clipping-final-kernel/TABLE.md),
and [method, results and caveats](observations/024-controls-posthoc-final-latency.md).
Reproduce with `15_plot_controls_clipping_latency.py`. The separate Figure3
uses six panels for OL1(all) minus OL1(h) at both sizes.
Figure7 now matches Figure3's typography, six-panel layout and circular-marker
style, with concise labels and the title "Post-hoc Clipping Sparsity and Latency".
Its [implementation note](observations/024-controls-posthoc-final-latency.md#implementation-note-for-later-text)
records the tile/short-row skipping rules and timed, unfused clipping operators.

Previous main paper figure: [side-by-side 14M quality-sparsity and latency figure](figures/06-14m-quality-sparsity-latency.pdf)
uses the same 22 trained pressure/control checkpoints in both panels, circular
markers, pressure-specific colors, one shared post-hoc note, and ceiling guides.
The wider layout includes the requested Pythia-14M title, T/P legend labels,
paper-sized text, and larger open/filled markers to distinguish the two controls.
It is a new variant; Figure 1 is preserved. See the
[caption and evidence](observations/023-14m-quality-latency-variant.md).
Reproduce with `14_plot_14m_quality_latency.py` in this analysis folder.

## Current paper figure set: task.md rebuild, 18 September 2026

**Current outputs: five main figures, four appendix figures, and Table 2.**
Start with [CAPTIONS.md](CAPTIONS.md), which links every PDF to its individual
caption, observation, proposed manuscript paragraph, and draft placement.
All 13 previous PDFs are preserved byte-for-byte in
[figures/.archive](figures/.archive/README.md). Historical material below remains
as a record; its links point to the archive. No new training, evaluation, GPU
benchmark, or cloud resource was launched for this rebuild.

The new [checkpoint-indexed evidence table](data/paper-checkpoints.json) joins
ordinary-final loss, canonical integer-pooled logical sparsity, qualified timing,
and clipping through exact checkpoint identities. It retains 62 checkpoint rows;
the declared primary comparison is 32 at 14M and 22 at 70M. The eight historical
local pressure settings are outside that primary cohort. The established
30-checkpoint 14M instruction appendix uses its own explicitly retained membership.
Kernel configuration, timing session, precision, initial-parameter identity,
training schedule, and the ordinary/logical loss passes remain separate metadata.

Figure 1 uses all 54 trained endpoints and the 40 dense/ReLU clipping evaluations,
with 14 high-loss clipping records outside its main Y range. Appendix A1 includes
all 340 retained clipping evaluations attached to the primary cohort. No clipping
sweeps are invented for the 20 h-only multisite checkpoints. Figure 3 now shows
20 OL1(all)-minus-OL1(h) contrasts at 14M and 70M in six panels: model sizes
form the rows, and loss/sparsity/full-model latency differences form the columns.
Latency subtracts retained checkpoint geometric means in µs and uses separate
scales by size. The caption preserves the 14M T7 cross-session caveat. Two curves distinguish
four- and seven-site threshold topologies, with explicit T/P contrasts in the
legend, Figure 6's blue/orange colours, and uniform circular markers.
Its final typography and line/marker sizes also match Figure 6; the figure is
widened to 12.8 inches for three columns. The title is
"Global (Pall) vs. Local (Ph) Pressure Paired Analysis"; panels use paired-difference
nomenclature and concise threshold/sparsity axis labels.
Figure 4 now shows twelve absolute operation decompositions from integer
counters: T4/T7 with P0/Ph/Pall at κ=.05/.5. Each stacked bar sums to the
checkpoint's model-wide sparsity on a shared 0–30 pp scale, using the operation
palette from the manuscript's `pressure-scope/05-site-structure.pdf`. Typography
matches Figure 3. Table 2 fixes pressure on h while comparing threshold placement.
Figure 5 displays absolute full-model latency and measured quality-latency
nondominance, preserving the Run029/Run033 session limitation and the distinct
70M port. No old regression is extended to new checkpoints.

Rebuild locally from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/13_rebuild_paper_figures.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s analyses/024-2026-09-17-h-only-kernel-latency -p test_paper_figures.py -v
```

For the six-panel all-minus-h revision alone, use
`13_rebuild_paper_figures.py --only-pressure-scope`. It updates Figure 3 and its
entry in `data/paper-derived.json` from the retained checkpoint table, leaving
the other PDFs and manuscript unchanged. See its
[revised caption and scientific scope](observations/015-pressure-scope-caption.md).

For Figure 4 alone, use `13_rebuild_paper_figures.py --only-operation-contributions`.
This updates the PDF and its twelve count-based decompositions without
regenerating other figures. See its
[absolute-contribution caption and evidence](observations/016-operation-changes-caption.md).

If MiKTeX requires a separate approved process for its user configuration, compile
`paper-architecture.tex` with `pdflatex` into `tmp/pdfs/analysis024-architecture/`,
then pass `--architecture-pdf tmp/pdfs/analysis024-architecture/paper-architecture.pdf`
to the builder. That argument must refer to a PDF just compiled from the retained
TeX source, not an older diagram. The builder saves exact pairs, selection IDs,
operation contributions, dense references, and source/output hashes in
[data/paper-derived.json](data/paper-derived.json). The nine PDFs were rendered
and visually reviewed. Verification results are recorded in
[PAPER-VERIFICATION.md](PAPER-VERIFICATION.md).

## Historical analysis record

This analysis owns the requested extension of the model-wide-sparsity versus
full-model-speedup and absolute-latency figures. Only the five new Run032
A7+OL1(h) final checkpoints are measured in Run033. All 35 historical Run029
checkpoints, including A4+OL1(h), are reused from their retained raw pairs.
The 70M counterpart is now complete in Run035: all 22 retained checkpoints and
all 66 fresh processes qualify under one frozen K050-derived shape port. See
[Figure04](figures/.archive/04-70m-k050-port-sparsity-latency-topology.pdf), its
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

- [Full-model speedup](figures/.archive/01-14m-k050-sparsity-speedup.pdf), with
  [observation and caption](observations/001-sparsity-speedup.md).
- [Native and K050 latency](figures/.archive/02-14m-k050-sparsity-latency.pdf), with
  [observation and caption](observations/002-absolute-latency.md).
- [Final K050 latency by topology](figures/.archive/03-14m-k050-sparsity-latency-topology.pdf),
  added on 18 September: 36 checkpoints (single-site naive L1 excluded), grouped as baseline, 1-site,
  4-sites and 7-sites, with [caption](observations/003-final-latency-topology.md).
- [Complete 40-checkpoint table](TABLE.md) and [full-precision data](data/results.json).
- [70M final port latency by topology](figures/.archive/04-70m-k050-port-sparsity-latency-topology.pdf):
  all 22 available checkpoints, four separate dashed pressure-family curves,
  and [caption](observations/004-70m-final-latency-topology.md).
- [Paired seven-minus-four-site distributions](figures/.archive/05-14m-paired-topology-effects.pdf):
  one row per pressure recipe, five matched kappa contrasts per row, and three
  panels for loss, logical sparsity and K050 latency. See the
  [observation and caption](observations/005-paired-topology-effects.md).
- [Paired OL1(all)-minus-OL1(h) distributions](figures/.archive/06-14m-paired-pressure-effects.pdf):
  four-site and seven-site rows, each with five matched kappa contrasts for loss,
  logical sparsity and K050 latency. See the
  [complete contrasts and caption](observations/006-paired-pressure-effects.md).
- [Pressure versus no pressure, 2-by-3 distributions](figures/.archive/07-14m-pressure-vs-none-effects.pdf):
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

The [combined 14M/70M latency figure](figures/.archive/08-14m-70m-final-sparsity-latency.pdf)
is reproduced with `08_plot_combined_latency.py`. It retains all 36/22 points
from Figures03/04, with separate dashed recipe curves, a common topology legend
and independently fitted linear axes in two panels. Source hashes, point and
connection identities, axis limits and the PDF hash are stored in
`data/combined-final-latency.json`. See the
[caption and comparison limits](observations/008-combined-final-latency.md).

The new [matched single-panel figure](figures/.archive/09-14m-70m-matched-sparsity-latency.pdf)
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

The separate [logarithmic latency version](figures/.archive/09-14m-70m-matched-sparsity-latency-log-y.pdf)
is reproduced with `09_plot_matched_combined_latency.py --log-y`. It retains
the same 44 points, eight curves and marker styling, using a logarithmic y axis
with tick labels in milliseconds. Its provenance is saved separately in
`data/matched-combined-latency-log-y.json`; the script checks exact point and
curve equality with the unchanged linear version. Observation009 documents
both versions and the ratio interpretation of distances on the log axis.

The [A0-normalized speedup figure](figures/.archive/10-14m-70m-matched-sparsity-a0-speedup.pdf)
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

The [quality-sparsity companion panel](figures/.archive/11-14m-70m-matched-quality-sparsity.pdf)
is reproduced with `11_plot_matched_quality_sparsity.py`. It restores both
1-site controls to Figure10's recipe cohort, yielding 44 points, and plots
ordinary final-checkpoint validation loss against model-wide sparsity. Colors,
open/filled markers, dashed recipe curves and model-size labels follow Figure10;
both axes are linear. The builder checks all losses against original metrics,
reconciles the uniform loss conventions in Analyses023/025, and verifies pooled
counts and checkpoint identities. Exact values and source hashes are in
`data/matched-quality-sparsity.json`; see the
[caption and provenance](observations/011-matched-quality-sparsity.md).

The separate [quality-sparsity figure with clipping](figures/.archive/12-14m-70m-quality-sparsity-clipping.pdf)
is reproduced with `12_plot_quality_sparsity_clipping.py`. It adds four dotted
post-hoc frontiers to Figure11's unchanged trained cohort: baseline and 1-site
at both 14M and 70M, with ten targets each. The full measured loss range is
shown. Each control's clipping acts at a,m,h,z, with its own retained p=0
measurement; no offset is applied to align different loss evaluation passes.
All 84 plotted records, source hashes and checkpoint checks are retained in
`data/quality-sparsity-clipping.json`; see the
[caption and measurement definitions](observations/012-quality-sparsity-clipping.md).
