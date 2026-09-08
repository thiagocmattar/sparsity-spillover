# Individual matched candidate outcomes

## Question, method and coverage

What is hidden by the cumulative-best progress line? Retain the complete fixed
retrospective cohort: c01(A0),c11(A4/kappa0),c25(A7/kappa0.5),c30(A7+OL1/kappa0.5).
There are54 configurations including P0,4 checkpoints and3 fresh processes:
648 scheduled evaluations and216 checkpoint-level comparisons. The baseline,
timing pairs, full338-block qualification and numeric bounds are identical to
[Observation01](01-matched-autoresearch-progress.md).

Of216 comparisons,140 qualify,74 fail numerical qualification, and2 are
unsupported. The latter are K017/K018 on ungated c01: these historical policies
require pre-existing a/m/h/z gates. All three replicates agree on each outcome.
The plot retains214 actual timed points, including all74 failed comparisons;
no timing is supplied for the two unsupported combinations. Slowdowns remain
on the native linear scale rather than being removed or renormalized.

Across the entire retrospective plus final matrix, outcomes are855 qualified,
312 numerical failures and6 unsupported fresh processes. There are no execution
or compilation failures. Every timed process completed all338 qualification
blocks; the common native graph passed throughout.

## Figure and caption

Current paper asset: [02-matched-individual-candidates-r02.pdf](../figures/02-matched-individual-candidates-r02.pdf).

**Caption:** Individual full-model speedups from the matched retrospective
cohort. Gray circles pass all numerical gates; red crosses fail them and must
not be interpreted as valid accelerations. Each point pools three fresh
processes. Multiple checkpoints and K011 configurations may share one proposal
ordinal; points are not jittered or converted to bars. P0 points are shown at
the separate left reference position. The blue line repeats the best qualified
c30 progress score, initialized by native1x; it is not the maximum over the
changing collection of plotted checkpoints.

## Result and limits

The early and intermediate history contains genuine slowdowns and numerical
failures. The qualified incumbent rises at proposals9,10,11 and42, ending at
1.783029x. Full-validation checks matter: several candidates and adapter/model
combinations that passed short smoke coverage fail on the complete stream.
The short smoke and calibration timings are not substituted into this figure.

The points use different trained checkpoints with different losses; this is
not a quality-matched model comparison. Numerical qualification preserves each
checkpoint's own native output. Repeated masks are configurations of one
historical proposal, not twelve independent research trials. Gate failures do
not imply crashes or large perplexity changes; raw logit and pooled-loss checks
are preserved to distinguish the criteria.

Source scripts: `03_matrix.py`, `10_reduce.py`, `12_figures.py --revision 2`,
`16_report.py`. Data: `results/matched-retrospective-001.json` and
`results/report-001.json`, including every unqualified group and its diagnostics.
Figure hashes: `results/figures-002.json`. The superseded revision1 PDF is
retained; revision2 displays quarter-step y ticks exactly (for example0.25,
not rounded0.2). No measurements or selections changed.
