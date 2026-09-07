# Auto-research progress on matched full-model speedup

## Question and requested presentation

How did measured full-model speedup progress as the auto-research process
tested successive sparse-kernel candidates? The user requested a single
metric against integer kernel iteration (K050 = 50), with multiple model
measurements shown as individual points at the same x coordinate. Figure06
has one axes, no grid, boxplots, bars, quantiles, variant descriptions, or
legend. This is a local replot of completed Run028 evidence, not a new kernel
experiment, training run, or cloud launch.

## Sources, selection, and coverage

Script `../125_reduce_search_progress.py` audits Run028's completed primary
development results, manifests, raw timing pairs, and numerical-quality
records under `../artifacts/k*-c*/`. It verifies candidate/checkpoint IDs,
runtime, full-logit shape, paired timing coverage, reported versus recomputed
speedup, and numerical-check coverage and status. All 123 result-bearing
attempts are inventoried: 116 complete primary development measurements,
six separate controls/alternate settings/precision diagnostics, and one
incomplete full-model attempt. Missing timings are never imputed as zero.

The frozen K036 and K050 final studies come from `../results/summary.json`
and `../results/summary-002.json`. Their source hashes are checked against
the existing figure-provenance records; frozen policy hashes, complete
35-model cohorts, and three-process geometric means are also verified.
These studies' detailed audits remain in observations01-04 and scripts65
and121. One result is selected per candidate/execution/checkpoint: final
cohort results supersede development results; otherwise greater validation
coverage, then timing coverage, then retry number take precedence. Neither
speedup nor qualification is used to choose among retries. The complete
inventory retains 234 selected points and 22 superseded points, with source
paths, byte counts, hashes, statuses, and selection metadata.

Figure06 uses **only native-graph / candidate-graph full-model speedup**.
The comparable metric is available for every candidate from K031 to K050:
136 checkpoint-level points, comprising 135 passing numerical checks and
one failure (K031/c25, retained as a cross). The 98 eager-mode points remain
in the machine-readable inventory but are not plotted or mixed into the
graph metric. Earlier kernel IDs are outside this figure's observed metric
coverage; iteration 31 is not relabeled as iteration 1.

All plotted workloads use the existing Pythia-14M checkpoints (training
seed1234, step712), one RTX5090, BF16, batch1, sequence length2048, and the
full50304-logit vocabulary output. There are 66 development points, each
timed on32 development input identities with7 paired passes in one process,
and70 final points (35 each at K036/K050), each timed on64 validation input
identities with7 paired passes in each of3 fresh processes. Each plotted
point has numerical checks over all500 MiniPile validation documents /
338 complete blocks:692224 input tokens,691886 prediction tokens, and the
reported excluded1444-token tail. Qualification includes reference/target
checks and the eager-stock numerical anchor where that older harness
requires it. The final c33/c34 small accepted errors are detailed in
observations03/04; passing is not a claim of bitwise identity.

Development speedup is recomputed as exp(mean(log(native latency /
candidate latency))) over the complete paired input/pass grid. Final points
are the equally weighted geometric means of three process-level paired
geometric means. Timing samples and process replicates do not become
additional model points. The plot uses exact integer x coordinates without
jitter; coincident points may overlap.

## Progress line and observed result

The orange step line is the cumulative best **fully numerically qualified
c30 checkpoint** result, holding checkpoint and graph execution fixed.
It is not the maximum over a changing model cohort, an average across models,
or an interpolated measurement. Lower-performing candidates remain visible
as scatter points and do not lower the incumbent line. Its monotonicity is
by construction, not evidence that every successive candidate improves.

The line starts at1.347607x for K031 and advances at K036 (1.379882x),
K038 (1.431262x), K039 (1.434493x), K045 (1.465584x), K049 (1.510580x),
and K050 (1.738990x). All other measured iterations hold the incumbent.
The scatter retains regressions below1x, including every K034 measurement.
K050's35-model graph speedups range from0.978736x to1.738990x; it is not a
universal acceleration across the whole cohort.

## Figure caption and encoding

**Figure06. Auto-research progress in specialized sparse-kernel development.**
The x-axis is kernel candidate iteration (K050 = 50); the y-axis is the
paired geometric-mean full-model native-graph / candidate-graph latency
ratio. Blue circles show individual model/checkpoint measurements at each
iteration; one orange cross retains a numerical-check failure. The orange
step line tracks the best fully qualified result so far on a single fixed
checkpoint, c30, and ends at1.739x. The plot contains136 points across
K031-K050, with35 models each at the frozen K036 and K050 endpoints and
two or four models at other iterations. All available values of this metric
are retained under the declared coverage-based retry policy. No per-model
legend or distribution bars are used. The y-axis is an explicitly labeled
expanded linear view (0.70-1.86x); values below1 indicate slowdowns.
Hardware, precision, workload, ratio baseline, and failed-point encoding
are stated on the figure.

## Limits and provenance

This is an adaptive engineering history, not a controlled comparison of
search algorithms or evidence that autonomous search outperforms human
tuning. The candidate number is an ordinal identifier, not a count of
independent trials, elapsed compute, or a cost-normalized budget. Coverage
starts at K031 because earlier primary results use a different execution
metric. Cohort sizes and model identities vary across iterations. Holding
c30 fixed in the line removes changing cohort composition from that line,
but development versus final timing inputs and process replication still
differ. No timing-uncertainty intervals are shown in this requested view.

The total speedup combines sparse execution with other implementation
improvements; this progress plot does not attribute all gains to sparsity
or demonstrate a universal R_model-to-speedup law. Use the matched controls
in observations03/04 for sparse-path and fusion attribution. The earlier
K044 implementation caveat remains in the audited record, without turning
the plot into a catalog of candidate mechanisms.

Reducer: `../125_reduce_search_progress.py`.
Plot builder: `../126_plot_search_progress.py`.
Output: `../figures/06-autoresearch-kernel-progress.pdf`.
Evidence inventory: `../results/search-progress-001.json`.
Figure/source/script hashes and presentation metadata:
`../results/search-progress-figure-001.json`.
Tests: `../test_search_progress.py` (raw pairing, non-performance retry
selection, fixed-checkpoint incumbent, single axes with all136 graph points
and no bars/grid/legend, and complete final cohorts).

The PDF skill's render-and-inspect workflow verified the final single-panel
figure at150dpi: embedded TrueType fonts, legible labels, no overlap or
clipping. Existing figures01-05 and frozen scientific sources are unchanged.
No manuscript text or finding registry is updated.
