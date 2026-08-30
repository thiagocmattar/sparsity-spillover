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
- `artifacts/pythia-architecture-map.tex` and its compiled PDF provide the
  intervention-neutral, paper-ready map of one shared Pythia block; the
  adjacent Markdown file records its notation, scope, caption, and caveats.
- `artifacts/sparsification-ladder.tex` and its compiled PDF define the
  paper-facing eight-step intervention ladder, visually separating gate sites,
  gate families, pressure methods, and topology-conditioned architecture
  ceilings; the adjacent Markdown file records its terminology and assumptions.

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
