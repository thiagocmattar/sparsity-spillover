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
until their evidence is linked to approved findings. Finding F001 is discarded:
Run 012 applied OL1 only at `h` despite declaring four pressure sites. Finding
F002 still supports the scoped, tentative Pythia-14M A4/A7 statement. Neither
supports scale-independent or runtime-speedup language. The F001-backed
introduction text is now a known stale claim pending explicit user-approved TeX
revision. Analysis 009 reconciles corrected Run 015 against Run 011 and the
historical Run 012 realization; the result remains descriptive and has not been
approved as a finding. Run 018 and Analysis 010 add descriptive Pythia-70M
evidence for the selected A0/A1-H/A4-OL1/A7-OL1 subset. They support reporting
whether the observed loss--`R_model` shape persists across 14M and 70M, but do
not support a scaling law, a 410M statement, or a runtime-speedup claim.

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

## Approved evidence integration

- Run 018 completed the selected one-seed Pythia-70M promotion of A0, A1-H,
  A4-OL1, and A7-OL1 with complete validation, sitewise diagnostics, logical
  counters, retained checkpoints, and complete A0/A1-H post-hoc TEAL sweeps.
  Analysis 010 owns the matched 14M/70M reduction and publication PDF. Status
  Report Number 2 reports these results descriptively, including all ten 70M
  TEAL targets; no finding is promoted and Pythia-410M remains unobserved.

- Finding F001 is discarded because Run 012's declared four-site pressure was
  actually realized only at `h`. Analysis 007 remains a historical reduction
  of Run 011 A4 versus Run 012 A4-Z + OL1@h and does not support A4-OL1.
- Run 015 completed the corrected five-condition four-site objective with a
  verified 24-tensor capture identity. Analysis 009 owns the matched corrected
  reduction. Status Report Number 1 uses Run 015 only for the main A4-OL1
  result and retains Run 012 as historical A4-OL1[`h`] evidence in an appendix.
  No four-site finding has been promoted.
- Finding F002 tentatively establishes that, for the same one-seed and
  one-pass scope, adding symmetric post-RoPE Q/K/V gates to A4 is near-null at
  `kappa=0`, improves both matched axes at `kappa=0.01`, and exchanges
  increasing validation-loss cost for increasing `R_model` at larger doses.
- Analysis 008 owns the count-reconciled interleaved A7/A7-OL1 zero-mass table,
  corrected single-panel frontier, observation, and source hashes. Its main
  A4-OL1 series comes from Run 015 through Analysis 009; Run 012 is excluded.
  Runs 011, 013, and 014 own the immutable A4, A7, and A7-OL1 evidence. Status
  Report Number 1 includes the Analysis 008 table and corrected figure; the
  A7/A7-OL1 comparison remains descriptive and is not part of Finding F002.
- `manuscript/introduction.tex` states these limited empirical scopes;
  `manuscript/methodology.tex` names the executed `z`, post-RoPE Q/K/V, A4-Z,
  and A7-Z-POST definitions. Logical opportunity remains distinct from runtime
  speedup.
