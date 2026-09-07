# Auto-research progress across the full kernel-iteration axis

## Question and authorization

On 2026-09-07 the user requested a second single progress figure beginning
at kernel iteration 1, using earlier runs as needed. The existing 14M scope
is retained; the need to distinguish hardware and execution protocols was
explicitly surfaced during preparation. This is reprocessing of completed
evidence, not a new experiment, kernel change, training run, or cloud launch.
No manuscript text or consolidated finding is changed.

## Sources and point selection

Run025's primary sealed RTX PRO 4500 search history is under
`runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/retrieved/rtxpro4500-002/artifacts/`.
This analysis uses confirmed K001 endpoints, the K006 portfolio endpoints,
every K011 14M mask setting with an available timing, and the complete final
K012 and K013 cohorts. K001's earlier lower-resolution attempts are
superseded by confirmation, not selected by their speed. Within K011,
selection uses numerical coverage, timing sample coverage, then explicit
retry precedence, never speedup or numerical success. Different masks are
retained as different measured settings. The early selection inventory has
103 attempt entries: 90 with timings, 13 without; 34 timing retries are
superseded, leaving 56 plotted points. Timing failures are not assigned zero.
This is a declared primary-history view, not every diagnostic, site control,
hardware retest, primitive, or post-hoc replication in Run025.

Run026 contributes primary eager K017 and K018 from
`autoresearch/artifacts/003-k017-eager/` and `006-k018-eager/`.
Its exploratory K019 variants are superseded here by the complete Run027
35-checkpoint K019+K018 characterization. That characterization is read from
`runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/results/summary.json`,
checked against its original figure-provenance hash. All 35 rows have three
processes and 1344 paired samples; all 29 numerical failures remain visible.
The original Run027 reducer owns the raw-data/frozen-source audit. These
are checkpoint-level paired geometric means, not ratios of pooled medians.

Run028 contributes its already audited primary-history points from
`runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/results/search-progress-001.json`.
That source is hash-verified against the original Figure06 provenance, and
each selected raw source record is reverified. Eager comparisons are used
through K030; graph comparisons from K031 onward. K036/K050 use all 35
frozen final-cohort points under Run028's non-performance selection policy.
All 136 graph-phase coordinates/statuses are identical to Run028 Figures06/07.
Available eager measurements at later IDs remain in the original evidence,
not mixed into this phase's graph ratio.

The result is one point per selected model/setting, not one point per
timing sample or process replicate. Repeated x coordinates are exact integer
candidate IDs, without jitter; some points overlap. The ordinal kernel ID
is not an independent-trial count, elapsed time, or compute budget.

## Metric, numerical qualification, and coverage

Speedup is native full-model latency / candidate full-model latency,
paired on input/repeat and with matching execution mode in each phase.
Development ratios are recomputed from raw samples as the geometric mean
of native/candidate ratios. Duplicate/missing cells, nonpositive/nonfinite
times, wrong output shapes, and disagreement with stored ratios are
rejected. Final cohort points retain the audited equally weighted
geometric-mean process reduction.

All workloads are Pythia-14M, BF16, batch1, sequence2048, complete50304-logit
outputs, uncached causal inference. Existing randomly pretrained checkpoints
use seed1234 and step712. Weights/gates/thresholds are not changed. The c30
anchor is the earlier `14m/a7-0p5` checkpoint; its four source file records
are exactly matched across early manifests and Run027 inputs and verified
locally, including the model hash
`f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.

The three phases contain 56, 61, and 136 points, respectively. There are
193 passing recorded checks and 60 failures: 11, 48, and 1 failures by
phase. Of the total253 points, 230 use all338 validation blocks /500
documents,692224 input tokens,691886 prediction tokens, and the excluded
1444-token tail. The other23 use8- or16-block development checks. Blue
markers mean passing the check actually recorded, not universal complete
validation. The incumbent only admits passing338-block results.

Early RTX PRO 4500 timings use16 training-cache input identities x3,5,or10
passes in one process. Run026 primary timings use16 inputs x5 passes and
full validation. Run027 has64 validation inputs x7 passes x3 processes.
Run028's eager history uses8 inputs x3 passes or32 x7; its graph development
uses32 x7 and final K036/K050 use64 x7 x3 processes. Training/development
and final validation timing inputs differ. Numerical gate statuses are
retained under the fixed original tolerance contract; passing does not
mean exact bit-pattern identity. Canonical R_model is not used to compute
any speedup in this figure.

## Missing IDs and incumbent definition

The full axis runs from K001 to K050. Plotted IDs are1,6,11,12,13 and17-50.
IDs2,3,4,5,7,8,9,10,14,15,16 lack a14M full-model timing in the declared
history: earlier search work includes other model sizes, isolated primitives,
and failed diagnostics. No other-size ratio is substituted, and missing
values are never plotted as zero or interpolated. The source record retains
the missing-ID list explicitly.

An orange step trace tracks the best fully qualified c30 result within
each hardware/execution phase. A changing model cohort cannot update this
fixed-checkpoint trace. Each trace is separately initialized; no line joins
the RTX PRO 4500 and RTX5090 phases or eager and graph phases.
The early trace starts at K011, not K001, because the confirmed K001 source
has only16-block development qualification. It reaches1.095521x at K012
and remains there through K013. K012 passed this anchor but failed other
checkpoints, so its incumbent status is not a universal policy qualification.

The RTX5090 eager trace starts at1.271296x (K017), reaches1.807615x (K019),
2.132028x (K025), and2.274083x (K027), retained through K030. Failed higher
timings do not raise it. The separate RTX5090 graph trace starts at1.347607x
(K031) and ends at1.738990x (K050), exactly as in Run028 Figure06.
Step-line monotonicity is by construction, not universal improvement of
successive candidates. Ratios from different phases must not be ranked as
if they shared one fixed baseline.

## Figure caption

**Figure01. Pythia-14M auto-research history across K001-K050.** Blue points
show selected model/setting full-model speedups that pass their recorded
numerical check; orange crosses retain failures. The y-axis is native /
candidate paired geometric-mean latency, with execution matched within
each phase. Light grid lines aid coordinate reading. Dashed vertical lines
separate RTX PRO 4500 eager, RTX5090 eager, and RTX5090 graph phases;
hardware/execution labels are placed above the same single axes. Orange
step traces show the best338-block-qualified speedup on the same fixed
checkpoint within each phase and restart at protocol boundaries. The graph
phase ends at1.739x. All253 selected points are retained without bars,
quantiles, or a variant legend. Blank iterations have no14M full-model
timing; no zeros are imputed. The linear y-axis spans0.1-2.52x, including
all recorded slowdowns. Cohort and validation/timing coverage vary as
specified above; no timing confidence intervals are shown.

## Limits, reproducibility, and visual verification

This is an adaptive engineering history with explicit protocol changes,
not a controlled50-step performance curve. Hardware, stock-reference
internals, workspace/backend decisions, timing coverage, checkpoint cohorts,
and candidate compositions change. The two plotted major boundaries do not
claim to exhaust every within-phase implementation difference. Faster dense
controls that failed numerical qualification in Run026 are not substituted
for the native baseline. There is no universal causal attribution of these
gains to sparse products or to the agent's search process.

A genuinely homogeneous K001-K050 curve requires a newly approved matched
benchmark, including handling candidates that did not target14M or never
produced a qualified full-model implementation. The present figure cannot
reconstruct absent measurements. Run028's graph-only figure remains the
cleaner within-protocol progress view.

Source/reduction: `../01_reduce.py`; plotting: `../02_plot.py`.
Output: `../figures/01-kernel-progress-all-iterations.pdf`.
Point inventory, coverage, missing IDs, and source hashes: `../results/progress.json`.
Figure/script/data hashes: `../results/figure-provenance.json`.
Focused tests: `../test_progress.py`.

The PDF skill required rendering and visual inspection. The final160-dpi
complete render has clear title/subtitle separation, readable phase labels,
embedded vector fonts, and no clipping. No previous run/analysis figure is
overwritten, and no new GPU spending is required.
