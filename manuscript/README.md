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

## Current files

- `introduction.tex` states the paper motivation, sparsity-spillover question,
  proposed contribution structure, and architecture-wide research direction.
- `methodology.tex` gives the formal Pythia graph, sites, activation operators,
  L1/OL1 definitions, exact product counters, `R_block`, `R_model`, and the
  draft architecture ceiling `R_model_max`.
- `experiment-control/README.md` records the paper-intent, execution, analysis,
  and finding status of the intervention ladder. Its A4 and A4-OL1 Pythia-14M
  cells are backed by Runs 011/012, Analysis 007, and Finding F001; its A7 cell
  is backed by Run 013, Analysis 008, and Finding F002.
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

## Current research direction

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

- On 31 August 2026, the user approved tentative Finding F002 from the five
  matched Pythia-14M A4/A7 full-pass pairs. Adding symmetric post-RoPE Q/K/V
  gates is a near-null topology expansion at `kappa=0`, improves both
  validation loss and measured `R_model` at `kappa=0.01`, and yields increasing
  logical opportunity with increasing validation-loss cost at larger
  thresholds. The finding is limited to one seed, one 14M scale, one MiniPile
  pass, and logical product opportunity rather than measured speedup. Analysis
  008 owns the expanded frontier, count-pooled eight-site table, and
  machine-readable reduction; Runs 011 and 013 own the verified endpoints.
- On 30 August 2026, the user approved tentative Finding F001 from the five
  matched Pythia-14M A4/A4-OL1 full-pass pairs. Four-site OL1 lowers validation
  loss and increases measured `R_model` at `kappa <= 0.1`; at `kappa=0.5`, the
  loss advantage reverses and the opportunity increment is negligible. The
  finding is limited to one seed, one 14M scale, one MiniPile pass, and logical
  product opportunity rather than measured speedup. Analysis 007 owns the
  figure, count-pooled site table, and machine-readable reduction; Runs 011 and
  012 own the verified endpoints.
- On 30 August 2026, the user approved reporting Analysis 005's uniform
  TEAL-style post-hoc GeLU/ReLU controls as a strong descriptive result in
  `reports/01-2026-08-30-status-update/status-update.tex`. The report references
  Analysis 005 Observations O001 and O002, includes the analysis-owned combined
  trained/post-hoc PDF, and retains the one-seed, intervention-semantics, and
  logical-opportunity caveats. No centralized finding was promoted. The
  `manuscript/reports/` directory remains intentionally Git-ignored; the tracked
  analysis observations and figures are the evidence record.
- On 30 August 2026, the user approved updating Status Report Number 1 with
  Analysis 007's paper-scale A4-OL1 result as a very strong descriptive result.
  The report replaces the prior combined trained/post-hoc plot with Analysis
  007's augmented frontier PDF, replaces the incomplete pilot-only A4-OL1
  subsection with all five Run 012 endpoints, and links Observation O001. It
  retains the one-seed, joint-intervention, logical-opportunity, and
  no-runtime-speedup caveats. The report update preceded the later approval of
  tentative Finding F001; the finding and this tracked crosswalk now carry the
  consolidated status. The report directory remains intentionally Git-ignored,
  while Analysis 007 retains the numerical provenance.
