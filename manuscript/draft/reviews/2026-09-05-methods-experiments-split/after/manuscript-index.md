# Living Manuscript Draft

This directory is the paper-facing scientific layer shipped inside the
bootstrap at its final `manuscript/` path. Copying the bootstrap into a new
repository therefore carries the manuscript and its workflow integration in
the same operation.

## Role

The draft sharpens the research question, terminology, formal definitions,
metric contracts, and intended contributions. It is allowed to lead the
research: a manuscript idea may motivate a new run, analysis, metric, or method.

It is not:

- a frozen experiment plan;
- proof that a contribution claim has been established;
- a substitute for an executed config, code, tests, or artifacts;
- required to describe every implementation detail before exploratory work can
  proceed.

The workflow does not try to keep the draft and implementation textually
identical. The currently documented manuscript/operational differences are an
accepted baseline, and operational contracts execute by default. A new
substantive discrepancy is surfaced during experiment design only when it would
materially change the experiment or the user requests a manuscript-specific
variant.

## V2 workspace

The user requested a clean start for V2 on 2026-09-05 and then authorized an
argument-first introduction and related work, followed by a compact
methodology and a separate experimental-study opening. `draft/` now contains
these four sections, general-method and experimental-detail appendices,
a checked bibliography, and a lightweight reading wrapper. The
framing concerns how pressure, gate nonlinearity, and site placement interact
in the quality--sparsity trade-off. The text develops a candidate explanation
and asks how it transfers. The experimental opening defines setup and
comparison logic; empirical findings remain deferred. The user's `draft/supplementary.md` remains unchanged.

[Draft positioning notes](draft/positioning.md) record the literature checks,
length budget, and evidence boundaries, including the combined gate/pressure
change in A4-OL1 versus A7-OL1, OL1's lack of a loss-convergence guarantee, and
the separation between logical sparsity and measured execution. The earlier
spillover claim is not reused as the paper's central argument. The existing
architecture/ladder artifact is included as a setup reference. Draft
sources and the reading copy retain the local-only ignore policy.

The subsequent user-requested adversarial review used three independent
sub-agents with explicit five-criterion scoring rubrics for writing, scientific
rigor, and literature/systems positioning. The revised introduction centers
separately controllable interventions whose effects need not be separable,
and states the reader's design decisions before introducing measurement.
Related work now contains three paragraphs on sparsification interventions
and two on inference speedup, with workload-specific execution reasoning and
no dedicated gradient-surgery strand. Detailed reports, initial/final source
snapshots, scores, and the response to criticisms are indexed in
[the review record](draft/reviews/2026-09-05-adversarial/README.md).
The review strengthens positioning; the empirical insight and its transfer
remain obligations for the later results stage.

The methodology defines independent gate and pressure choices, exact-zero
model-wide sparsity, and the all-zero selected-site reach ceiling, with most
math inline. The paper now uses calligraphic S while preserving operational
`R_*` fields and fractional units. Exact gates, OL1 mechanics, integer counters,
architecture derivations, and validation coverage are in the appendix;
[provenance notes](draft/methodology-notes.md) identify the implementation
contracts and immutable architecture configurations. The setup artifact uses
the same notation and larger lettering without changing its analytic values.

The [methodology review record](draft/reviews/2026-09-05-methodology/README.md)
retains three independent source audits, initial and revised scored reports,
source/PDF snapshots, a final visual follow-up, and exact ceiling verification.
Final editorial scores are 20/25 for writing, 23/25 for scientific rigor, and
22/25 for literature/presentation; these assess the methods draft, not paper
acceptance or empirical evidence. That reading copy built cleanly, its eight
pages were inspected, and all twelve analytic architecture/topology count
pairs matched the existing implementation. No scientific code or result was
changed and no experiment was launched.

The subsequent user-approved section split keeps Methodology focused on
gate/pressure definitions, sparsity, and structural reach (about 380 words).
The separate [Experimental Study](draft/experimental-study.tex) opening
contains Pythia/MiniPile context, A* recipes, matched-comparison logic, selected
larger-model scope, and the existing setup figure. Pythia-specific counts,
configuration links, and validation coverage moved into an experimental
appendix. The four numbered equations and the figure artwork are unchanged.

The [split review](draft/reviews/2026-09-05-methods-experiments-split/README.md)
includes an independent writing proposal, source audits, and two scored review
rounds. Final scores are 20/25 for writing and 23/25 each for scientific rigor
and literature/presentation; all substantive review issues were resolved.
The nine-page reading copy builds cleanly, with no unresolved references or
box warnings. No result or scientific code was changed and no run was launched.

The 4 September matched-intervention paper is preserved in
[v0_2026-09-05](draft/archive/v0_2026-09-05/README.md), including its PDF,
sources, review notes, and complete former working directory. That paper covers
detailed 14M contrasts, selected 70M recipes, operation-level explanations,
fixed-budget 410M appendix results, and negative 70M execution evidence.

[Analysis 013](../analyses/013-2026-09-04-matched-intervention-manuscript/README.md)
owns its five figures, generated tables, verified reduction, and observations.
The draft remains under the existing local-only ignore rule. The older TeX
and report files below preserve earlier proposals and evidence cutoffs; their
410M-unobserved descriptions predate the completed cohorts.

## Draft archive

The local [archive index](draft/archive/README.md) lists snapshots named
`vN_YYYY-MM-DD`, starting at v0 and incrementing for each archive. The date is
the archive date; `draft/` is the clean workspace for the next manuscript.

| Version | Main argument / framing |
| --- | --- |
| [v0_2026-09-05](draft/archive/v0_2026-09-05/README.md) | Matched gate-placement and activation-pressure interventions improve parts of the quality-sparsity frontier; broader pressure can increase the quality cost. |

Each snapshot preserves the PDF, manuscript sources, bundled figures/tables,
style files, review notes, and a SHA-256 inventory. The archive retains the
draft's existing local-only Git policy. This tracked index records its location
and framing. v0 precedes the proposed pressure-allocation-by-nonlinearity study.
Its `working-folder/` preserves all 102 files moved out of `draft/`, including
earlier revisions, QA renders, and build outputs, with verified hashes recorded
in `working-folder-manifest.json`.

## Historical sources and artifacts

- `introduction.tex` states the paper motivation, sparsity-spillover question,
  proposed contribution structure, and architecture-wide research direction.
- `methodology.tex` gives the formal Pythia graph, sites, activation operators,
  L1/OL1 definitions, exact product counters, `R_block`, `R_model`, and the
  draft architecture ceiling `R_model_max`.
- `experiment-control/README.md` records the paper-intent, execution, analysis,
  and finding status of the intervention ladder. Its A4-OL1 Pythia-14M cell is
  completed by corrective Run 015 because Run 012 actually applied OL1 only at
  `h`; Finding F001 remains discarded and Run 015 is not yet finding-backed.
  Its A7 cell is backed by Run 013,
  Analysis 008, and Finding F002; its completed A7-OL1
  cell is backed by Run 014 and reported descriptively through Analysis 008.
  Its selected Pythia-70M A0, A1-H, A4-OL1, and A7-OL1 cells are complete in
  Run 018 and compared with 14M in Analysis 010; 410M remains unobserved.
- `reports/01-2026-08-30-status-update/` contains the user-requested TeX source,
  PDF, and report-local provenance notes for Status Report Number 1. This
  numbered report is an explicit tracked exception to the directory-wide
  ignore rule; LaTeX auxiliary files remain untracked.
- `reports/02-2026-09-01-status-update/` contains Status Report Number 2's
  generated TeX, compiled PDF, focused verifier, current Pod-ID-reconciled
  billing provenance, and report-local notes. It carries the verified selected
  70M promotion through Analysis 010 without promoting a finding.
- `artifacts/pythia-architecture-map.tex` and its compiled PDF provide the
  intervention-neutral, paper-ready map of one shared Pythia block; the
  adjacent Markdown file records its notation, scope, caption, and caveats.
- `artifacts/sparsification-ladder.tex` and its compiled PDF define the
  paper-facing eight-step intervention ladder, visually separating gate sites,
  gate families, pressure methods, and topology-conditioned architecture
  ceilings; the adjacent Markdown file records its terminology and assumptions.
- `artifacts/pythia-architecture-sparsification-ladder.tex` and its compiled PDF
  combine the architecture map above the intervention ladder as a single
  publication-ready vector figure; the adjacent Markdown file records its
  layout, caption, sources, and scope.

The methodology records upstream provenance in its header. The lean handoff was
separately distilled from `orthogonal-sparsity-pressure` commit
`a5171e2c13d279ec97bdf7b7ef77139fa776e149`. Preserve both identities when a
definition is reconciled.

## Earlier research direction

1. Establish whether pressure at targeted FFN/branch sites produces a
   systematic opposing response at untargeted attention sites: *sparsity
   spillover*.
2. Measure the full executed graph with exact site counters and actual
   zero-operand product counters rather than treating local sparsity as model
   compute.
3. Validate architecture- and sequence-dependent logical reach ceilings for
   candidate intervention topologies.
4. Test whether architecture-wide gates plus conflict-aware pressure improve a
   quality--logical-compute frontier relative to local pressure alone.
5. Keep logical opportunity, sparse-kernel realization, and measured speedup as
   separate claims.

These are directions and hypotheses. Their evidence status lives in run and
analysis observations and in user-approved findings.

## Authority and evidence

Use this order when answering different questions:

1. An immutable run config, code identity, and artifacts say what executed.
2. Tested shared code and compact `research/` contracts say how the current
   workflow operationalizes a method or metric.
3. The manuscript states the current scientific framing and formal proposal.
4. Observations record measured patterns; user-approved findings determine
   which empirical statements may be promoted into the paper.

None of these layers silently rewrites another. Record a mismatch and decide it
explicitly.

The compact list of currently known manuscript/operational differences is in
[`research/MANUSCRIPT.md`](../research/MANUSCRIPT.md).

## Adding future TeX

Add a `.tex` section or generated table only when it serves a current paper
need. Every result-bearing fragment must identify its source run/analysis,
observation or finding, generating script when applicable, coverage, and whether
the file is generated or hand-edited.

Figures and numerical reduction scripts remain owned by their numbered run or
analysis. The manuscript may reference those PDFs and may contain a generated
table fragment, but it does not become a second results store.

Do not build a generalized manuscript pipeline yet. Add a parent document,
bibliography, build command, or generated-fragment directory only when the paper
actually needs it.

## Result evidence crosswalk

- On 4 September 2026, the user requested the full manuscript rewrite using
  current evidence and adversarial-review-v3. Analysis 013 reconstructs the
  full 35-condition 14M study, 12 selected conditions each at 70M and 410M,
  all 60 clipping points, Run 021's learning-rate screen, and six completed
  Run 023 execution sentinels. The draft reports dose-dependent paired
  effects, common-loss operating points, architectural denominator effects,
  and the runtime limitation. The pressured A4/A7 comparison remains a
  combined gate-and-pressure intervention. The 410M results remain visible
  in the appendix with a main-text disclosure. This is authorized descriptive
  integration; the finding registry and immutable runs remain unchanged.

- On 1 September 2026, the user requested Status Report Number 2 after Run 018
  and Analysis 010 completed. The report retains Report 1's architecture
  diagram and 14M result structure, updates the execution scope and RunPod
  billing, adds matched 14M/70M A4-OL1 and A7-OL1 tables with count-pooled
  exact-zero masses, reports all ten 70M A0/A1-H post-hoc TEAL targets, and
  embeds Analysis 010's single validation-loss--`R_model` frontier. The result
  is descriptive one-seed evidence across two sizes; no scale-independent
  claim, runtime-speedup claim, or finding is promoted, and 410M is unobserved.

- On 31 August 2026, the user approved tentative Finding F002 from the five
  matched Pythia-14M A4/A7 full-pass pairs. Adding symmetric post-RoPE Q/K/V
  gates is a near-null topology expansion at `kappa=0`, improves both
  validation loss and measured `R_model` at `kappa=0.01`, and yields increasing
  logical opportunity with increasing validation-loss cost at larger
  thresholds. The finding is limited to one seed, one 14M scale, one MiniPile
  pass, and logical product opportunity rather than measured speedup. Analysis
  008 owns the expanded frontier, count-pooled eight-site table, and
  machine-readable reduction; Runs 011 and 013 own the verified endpoints.
- On 31 August 2026, Finding F001 was discarded after implementation audit.
  Run 012 declared four-site pressure but reused a training capture fixed to
  `h`; its endpoints are A4-Z + OL1@h, not A4-OL1. Analysis 007 remains a
  historical reduction of those endpoints. Corrective Run 015 subsequently
  completed all five matched four-site conditions with a verified 24-tensor
  pressure identity. Its run-local observation finds paired loss/`R_model`
  improvement at `kappa=0` and `0.01`, followed by increasing logical
  opportunity at increasing loss for larger thresholds. Existing F001-backed
  result prose in `introduction.tex` remains known stale: Run 015 does not
  restore F001 or authorize TeX changes without a new analysis and explicit
  user approval.
- On 30 August 2026, the user approved reporting Analysis 005's uniform
  TEAL-style post-hoc GeLU/ReLU controls as a strong descriptive result in
  `reports/01-2026-08-30-status-update/status-update.tex`. The report references
  Analysis 005 Observations O001 and O002, includes the analysis-owned combined
  trained/post-hoc PDF, and retains the one-seed, intervention-semantics, and
  logical-opportunity caveats. No centralized finding was promoted. Analysis
  005's tracked observations and figures remain the numerical evidence record.
- On 30 August 2026, the user approved updating Status Report Number 1 with
  what was then interpreted as Analysis 007's paper-scale A4-OL1 result.
  The report replaces the prior combined trained/post-hoc plot with Analysis
  007's augmented frontier PDF, replaces the incomplete pilot-only A4-OL1
  subsection with all five Run 012 endpoints, and links Observation O001. It
  retains the one-seed, joint-intervention, logical-opportunity, and
  no-runtime-speedup caveats. The report update preceded the later approval of
  tentative Finding F001. The 31 August implementation audit supersedes that
  interpretation: the report's Run 012 labels are stale, F001 is discarded,
  and Analysis 007 retains numerical provenance only for A4-Z + OL1@h.
- On 31 August 2026, the user approved updating Status Report Number 1 with
  Run 014's completed A7-OL1 cohort. The report now contains the ten-row matched
  A7/A7-OL1 count-pooled table and embeds Analysis 008's corrected single-panel
  frontier with plain OL1 labels. A7-OL1 is marked complete at the report
  cutoff, and Run 014's settled cost is included. The A7/A7-OL1 comparison
  remains descriptive and is not added to Finding F002. The report's README,
  TeX source, and compiled PDF are explicitly tracked for this requested
  update even though `manuscript/reports/` stays ignored by default.
