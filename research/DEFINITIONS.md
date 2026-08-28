# Definitions

These terms are the interface between the user's intent and the implementation.
Add or refine a term before using it ambiguously in a run.

## Workflow terms

- **Run** — one approved training experiment or diagnostic in
  `runs/NNN-YYYY-MM-DD-slug/`.
- **Analysis** — a numbered folder that compares or reprocesses multiple prior
  runs without changing them.
- **Observation** — one run- or analysis-local Markdown record of a measured
  pattern, including its figure caption and limitations.
- **Finding** — a user-approved consolidated conclusion. Status is `confirmed`,
  `tentative`, or `discarded`.
- **Manuscript-led goal** — a research question, proposed definition, or intended
  contribution from the living draft. It can motivate work but is not evidence
  or launch authorization.
- **Design approval** — confirmation that the agent understood the experiment.
  It authorizes implementation, not execution or cloud spend.
- **Launch approval** — confirmation of the implemented run, tests, ETC,
  execution location, cost envelope, transfer plan, and monitoring plan.

## Model and intervention terms

- **Random pretraining** — construct the model from a pinned architecture config
  using `from_config`; do not load released weights.
- **Sparsification site / site** — an exact activation tensor port feeding a
  counted downstream matrix multiplication and eligible for an intervention or
  diagnostic. Canonical aliases are `a`, `m`, `h`, `q_pre`, `k_pre`, `q_post`,
  `k_post`, and `v`.
- **Topology** — the set of ports where an architecture gate is active. It does
  not choose the gate, pressure method, optimizer, or pressure target.
- **Gate** — an architecture intervention replacing values at active topology
  sites: ReLU, one-sided threshold, or symmetric threshold.
- **Pressure target** — site activations used to construct the auxiliary pressure
  objective. It is independent of active gate sites.
- **`l1_naive` / L1** — optimize task loss plus a weighted mean absolute
  activation objective.
- **`orthogonal_l1` / OL1** — take a task-only AdamW step, precondition pressure
  with AdamW's task second moment, remove only a conflicting component, limit
  the correction with a trust budget, then apply it after AdamW.
- **Sparsity spillover** — a nonlocal response at an untargeted site when
  sparsity pressure is applied elsewhere. The draft's focal pattern is targeted
  FFN/branch mass moving toward zero while untargeted attention distributions
  lose near-zero mass or broaden. It is descriptive unless the comparison
  isolates the pressure intervention; activation marginals alone do not prove a
  causal route.
- **Post-hoc clipping** — at evaluation only, set selected activations to exact
  zero under a stated rule. It is not a trained gate or pressure method.

## Measurement terms

- **Exact-zero fraction** — `count(x == 0) / count(x)`.
- **Near-zero mass at epsilon** — `count(abs(x) <= epsilon) / count(x)`. Epsilon
  is part of the metric name.
- **Activation RMS** — `sqrt(sum(x^2) / N)` over finite elements. This is the
  preferred scale-comparable activation norm; raw L2 is `RMS * sqrt(N)`.
- **Weight norm** — L2 norm of an explicitly named parameter tensor or pooled
  parameter set. Record parameter names and element count.
- **Gradient conflict** — the global task-pressure gradient dot product is
  negative at the same optimizer boundary.
- **`R_block`** — zero-operand logical products in the six measured transformer
  block operation families divided by all logical products in those families.
- **`R_model`** — the same block zero-product numerator divided by the block
  denominator plus the dense LM-head product count. It counts all actual
  zero-operand products, whether or not a selected gate created the zero.
- **`R_model_max(topology, architecture, T)` / `R_max`** — the analytic fraction
  of the same model denominator reachable when every site selected by the
  topology is identically zero. Reach is `a→QKV`, `m→W1`, `h→W2`, either selected
  Q or K operand `→QK`, and `v→PV` plus the deterministic `V=0→C=0→W_o` closure.
  It is an architecture/workload ceiling, not an observed sparsity rate.
- **Architecture utilization (`U_arch`)** — manuscript normalization
  `R_model / R_model_max` for nonzero ceilings. Because measured `R_model`
  includes natural zeros outside selected-site reach, this ratio is not assumed
  to lie in `[0,1]`; report its exact numerator semantics or omit it.
- **Quality--logical-compute frontier** — nondominated evaluated conditions over
  task quality and a declared logical-compute metric such as `R_model`. It is not
  a hardware speed frontier without direct runtime evidence.
- **Logical opportunity** — a scalar product has at least one exact-zero operand.
  Dense kernels may still execute it.
- **Measured speedup** — observed wall time, throughput, energy, or kernel work
  saved by an implementation that exploits zeros. It is not inferred from `R`.
- **Count-first pooling** — sum integer hits and denominators across the requested
  batches/layers, then divide once.
- **Complete validation** — all complete 2,048-token blocks from all 500 MiniPile
  validation documents in deterministic order; 338 sequences and 692,224 input
  tokens, with a 1,444-token tail reported but not evaluated.
- **Matched data order** — identical realized ordered training block starts, not
  merely the same nominal seed.
- **ETC** — estimated time to completion, including setup, training, validation,
  diagnostics, checkpointing, and transfer where applicable.
