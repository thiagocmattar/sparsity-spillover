# Manuscript Integration

The living manuscript under `manuscript/` is part of the research workflow. It
provides the current paper question, terminology, formal proposals, and intended
contribution structure. Read the relevant section before designing work that
could change a manuscript definition or claim.

## Relationship to operational research

- The manuscript may motivate experiments and analyses.
- `research/DEFINITIONS.md`, `METHODS.md`, and `METRICS.md` are compact
  operational contracts distilled from the draft and tested implementation.
- Executed configs, code identities, and artifacts remain the authority for what
  happened in a run.
- Observations and user-approved findings determine whether an empirical claim
  is supported.

The goal is precision, not forced textual identity. The user has accepted the
known starting differences below, and the operational contracts execute by
default without repeating that crosswalk during every design confirmation.
Reopen a difference only when the user requests the manuscript variant or when
a new difference would materially change the proposed experiment. Never repair
such a new discrepancy silently.

## Current manuscript-led goals

- Test for sparsity spillover from targeted pressure into untargeted attention
  activation distributions.
- Report exact sitewise zeros/near-zero mass and model-wide actual zero-product
  opportunity as different objects.
- Validate how topology, architecture scale, and sequence length change the
  analytic `R_model_max` ceiling.
- Test architecture-wide gate placement and conflict-aware pressure against
  simpler local-pressure baselines on a quality--logical-compute frontier.
- Avoid translating logical opportunity into a runtime claim without sparse
  kernels and direct measurements.

Contribution statements in the introduction are hypotheses or candidate claims
until their evidence is linked to approved findings.

## Acknowledged starting differences

These are accepted crosswalk notes, not blockers or a demand to rewrite the
draft:

- the operational baseline globally clips the accumulated task gradient to norm
  1.0 before AdamW; a run using OL1 must state that the same clipped task
  gradient drives AdamW and its adaptive task direction;
- the bootstrap's `orthogonal_l1` requires a positive trust budget, while the
  draft describes a budget as optional;
- the bootstrap retains a uniform-gate interface and now also supports an
  explicit per-site mapping; approved Run 008 uses one-sided `a,m,h,z` gates
  with symmetric post-RoPE Q/K/V gates, while the draft's proposed
  branch-ReLU/attention-threshold combination remains a different unexecuted
  operator assignment;
- the operational topology registry includes A2 = `m,h`, which is not listed in
  the current manuscript table;
- the architecture-map artifact includes the pre-`W_o` context site `z`, while
  the current methodology prose and topology table omit it; approved Run 006
  operationalizes `z` and topology A4-Z without silently editing manuscript TeX;
- artifacts store `R` fractions in `[0,1]`, while manuscript prose and tables
  display percentages;
- the default evaluation protocol uses all 500 MiniPile validation documents,
  a run-level data contract intentionally kept out of the formal graph section.

Surface the relevant item during design confirmation. The user may revise the
draft, retain the operational definition, or approve a new variant.

## During design confirmation

For work connected to the paper, identify:

1. the manuscript question, term, equation, table, or claim affected;
2. whether the work tests an empirical claim, validates an analytic definition,
   or proposes a new method;
3. the operational definition that will execute;
4. any new or design-material difference from the manuscript wording;
5. the observation, figure, table, or TeX fragment that could result.

This information belongs in the numbered run/analysis README, not in a central
executable plan.

## From results back to TeX

After the user approves a finding or requests manuscript work:

1. preserve the source observation and exact evidence links;
2. update or add the smallest relevant `.tex` section;
3. identify generated table fragments and their generating analysis script;
4. retain figures in the owning run/analysis folder as PDF;
5. record the manuscript change in `manuscript/README.md` or its future evidence
   crosswalk.

Do not automatically promote a successful run into a paper conclusion.
