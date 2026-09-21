# Results argument: broad 14M recipes to targeted execution and 70M

The author requested surgical edits following the reorganized introduction:
14M results, site/layer spillover, conditional execution savings, and promotion
to 70M. All pre-existing manuscript changes were committed as `05bbe79a`
before editing. The results retain the author's subsection order and the
existing figures and table. The convergent sites are **h,z**, as specified in
the introduction, T2 definition and measurements (the request's sketch said
k,z).

## Editorial changes

- Add evidence to the previously empty 14M subsection: the aggressive T7/Pall
  endpoint, eight matched pressure comparisons, and the sparsity/latency
  counterexample moved out of the 70M discussion.
- Preserve the moderate-threshold spillover ranges. Quantify pressure-cap
  saturation without attributing it causally to q,k,v, and explain the
  high-threshold heatmap where h,z are already nearly saturated without pressure.
- Lead the execution subsection with Table 2's conditional h/z effects, then
  explain why avoided weight reads matter and why attention bypass is not
  sufficient. Introduce T2/Ph from the convergence of these diagnostics.
- Lead the 70M subsection with T2's quality ordering. Retain the failed direct
  kernel transfer and additional optimization, including the remaining latency
  cost of the lower-loss T2 settings.
- Move the post-hoc clipping latency paragraph to the kernel appendix. Fix the
  duplicated figure label and allow the heatmap and 70M figures to float,
  removing the large gaps caused by fixed placement. Correct `accross` in
  `experimental-study.tex`.

No new measurement, figure, table or scientific code was generated. Subsequent
author edits to `introduction.tex` were read and preserved, but are outside this
edit's final commit.

## Evidence and claim checks

The endpoint values, ceilings and pressure comparisons use
[Analysis030](../../../../analyses/030-2026-09-20-70m-t2-optimized-endpoint/observations/001-later-70m-endpoint.md),
`01_integrate.py`, and its `data/full-trained-results.json` export:

- At 14M, T7/Pall at kappa=0.5 has 27.48% sparsity (91.8% of its ceiling),
  loss 5.829 versus Base's 5.209, and latency 0.473 ms. T4/Pall has 12.71%
  sparsity and 0.459 ms latency.
- In all eight 14M T4/T7 pressure pairs at kappa <= 0.1, Ph has lower loss
  (0.192102--0.319637) and lower measured latency. The largest absolute sparsity
  difference is 2.050796 percentage points. These are endpoint comparisons,
  not replicated-seed estimates.
- At 14M, T2/Ph at kappa=0.1 has 5.174573% sparsity against a 5.347147%
  ceiling, loss 5.151098 and latency 0.573427 ms. The retained native Base
  reference, 0.655442 ms, comes from Analysis024 `data/kernel-appendix.json`.
- T2/Ph has the lowest 70M loss at each of the five matched thresholds among
  T2/T4/T7, not among all controls. Its lower-threshold loss, sparsity and
  latency ranges, the aggressive T7/Pall endpoint, and the later T2 endpoint
  are checked against the same export. Timing-session references remain
  separate; no scale-dependent speedup trend is asserted.

Spillover uses
[Analysis021 O016](../../../../analyses/021-2026-09-10-training-results-figures/pressure-scope/observations/O016-pressure-placement.md),
`evidence.py`, `plot_layer_sparsity.py`, and `pressure-scope/data/evidence.json`.
The quoted ranges are minima/maxima of per-layer integer zero/total ratios,
not averages of percentages. At kappa=0.5, unpressured T4/T7 h,z exceed 99.5%
in every layer; T7 q,k,v span 13.19--35.98%. The 96.99%/0.25% cap frequencies
cover the full T7 threshold grid. They do not isolate q,k,v's causal effect.

Execution evidence comes from
[Run037](../../../../runs/037-2026-09-19-pythia14m-operation-latency/observations/001-operation-latency.md),
Analysis024 `25_build_operation_latency_table.py` and
`data/operation-latency.json`, and
[Run039](../../../../runs/039-2026-09-19-pythia14m-am-port-fixture/observations/001-am-port-verdict.md).
The h/z savings are 150.069792/28.959677 microseconds; PV adds 6.2 microseconds.
These are conditional, nonadditive effects on one 14M checkpoint. The a/m
precheck trials are distinct implementation changes. The retained 70M
optimization also changes dense components, so its improvement is not
attributed entirely to activation skipping.

## Verification

The isolated `latexmk -pdf -interaction=nonstopmode -halt-on-error` build in
`tmp/results-arc-20260921/` succeeds with 22 pages, resolved references and
citations, no overfull boxes, and two underfull horizontal-box warnings.
All pages were reviewed in contact sheets; the changed results and appendix
pages were inspected at higher resolution. Numerical checks and source hashes
are recorded in [verification.json](verification.json).

`manuscript/draft/main.pdf` was not replaced. The temporary review build
includes the author's introduction edits present at build time; its source
hash is recorded separately from this edit's files.

## Follow-up: 70M execution control

The author approved promoting the same-checkpoint h,z replacement control to
the main 70M results and conclusion. Analysis028 `04_controls.py` and
[`data/retained-controls.json`](../../../../analyses/028-2026-09-20-70m-optimized-grid/data/retained-controls.json)
give 1.127765964 ms for opt073 and 1.365126858 ms for selected-native-hz on
Run042 c21 (T7/Ph, kappa=0.5). The conditional full-model reduction is
`100 * (1 - 1.127765964 / 1.365126858) = 17.38746%`.
It includes fusion, layout and inspection effects alongside sparsity
exploitation. The session is separate from the 1.088 ms matched-grid result.

The 1.481x result is explicitly an end-to-end comparison against native Base,
including a different checkpoint and both sparse and dense implementation
changes. Figures, Table 2 and the appendix controls are unchanged. Existing
author edits to the introduction and spillover paragraphs are preserved and
excluded from this follow-up's source commit.

The requested rebuild installs the checked 22-page PDF with resolved references,
no overfull boxes and the same two underfull hbox warnings. All pages were
inspected in contact sheets; the changed results and conclusion were inspected
at higher resolution. Installed SHA256:
`ac15cb26f2fe3ef1f9c46aaa6ecd680412a8e00851992bf54e184abd02807c86`.
This rebuild supersedes the earlier PDF verification above.

## Approved abstract structure

The author approved replacing the findings list with the introduction's
diagnostic progression: placement question, broad 14M recipes, spillover and
conditional execution evidence converging on h,z, targeted T2, and the 70M
scale check. The abstract defines P/T and h/z once, retains the 150/29-us
conditional effects and the 14M T2 quality/latency example, and scopes the
17.4% path benefit to the tested 70M T7/Ph checkpoint. Its fusion/layout/
inspection qualification is preserved. Evidence is the endpoint and control
data already audited above; no new scientific claim or measurement is added.

Only the abstract prose changes. The rebuilt 22-page PDF has resolved
references, no overfull boxes and the same two underfull hbox warnings.
The opening page was inspected at high resolution and all pages in contact
sheets. Windows blocked replacement of the open `main.pdf`; the verified
build is saved as `main-updated.pdf`. The author's ongoing introduction and
spillover edits remain outside this revision's source commit.
