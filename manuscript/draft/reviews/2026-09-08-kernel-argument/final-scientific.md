# Final scientific review of the revised argument

Reviewed 8 September 2026 against the current kernel-evidence.md audit,
Analysis 018's 30-checkpoint runtime reduction, Run 029's historical protocol,
and the frozen attention implementation. Read kernel-autoresearch.tex,
the Kernel qualification and ablations subsection in results-appendix.tex,
and methodology.tex. No TeX edits, new experiments or commits.

## Material correction: resolved

**results-appendix.tex:237-242: distinguish historical search coverage from
final evaluation coverage.** The sentence "The retrospective replays frozen
proposals on the same 30-checkpoint 14M cohort used throughout the manuscript"
is false as written. Historical proposals were replayed on four checkpoints
(c01, c11, c25, c30), with the incumbent selected on c30. The final implementation
and ablations were evaluated on 35 checkpoints; Analysis 018 retains c01-c30.
This is a protocol fact, not an optional caveat.

Resolution verified in current `results-appendix.tex:237-244`: historical
proposals now use four representative checkpoints and fixed A7-OL1 kappa=0.5
for the incumbent; final summaries explicitly retain 30 from the historical
35-checkpoint sweep, with the excluded five explained. Separate progress and
final maxima remain correctly distinguished. This resolves the protocol issue.
Source: Run 029 README, Observation 01, and Analysis 018 O007.

## Two concise wording improvements: implemented

- Verified `kernel-autoresearch.tex:38`: "Ablations distinguish fusion from
  the contribution of sparse paths." This accurately leaves room for the
  hybrid SIMT changes in the implementation-path toggle.
- Verified `results-appendix.tex:283-287`: the attention counters are now
  explicitly "matrix multiply-accumulate (MMA) instructions," with the
  scalar-product versus MMA distinction retained.

## Assessment of the remaining revision

The reasoning reads naturally: compatibility problem, practical human-guided
agent method, matched search progress, frozen-kernel transfer across trained
conditions, and a mechanism-specific limitation. It strengthens the scientific
case without framing authors' expertise as a weakness.

- The unchanged TwELL control is kept separate from adapted P0. P0's 4/30
  qualification and 0.8910x subset mean are accurate and clearly scoped.
- Agent attribution describes a human-guided Codex trajectory and does not
  imply agent superiority, fully autonomous discovery or verified per-request
  model identity. Upstream components are identified in the appendix.
- Iteration 11 and 42 values are correct. "Substantial gains appear early,
  followed by a later improvement" is supported; early convergence is not claimed.
- The 30-checkpoint 1.2340x mean, R-squared 0.8167, 14/30 sparse-path benefits
  and 1.2506x attention-dense mean agree with independently recomputed evidence.
  "Strong, approximately linear association" and the caption's causal limit
  accurately characterize the regression.
- Attention skipping is described at its implemented fragment granularity.
  Either complete fragment may be zero; matching Q/K zero patterns are not
  required. The text correctly avoids claiming a measured profitable joint-zero
  pattern or break-even threshold. The measured workload limits the conclusion.
- The threshold-adaptation motivation is clearly posed as a hypothesis, with
  fixed cutoffs, variable retained counts, threshold semantics and pressure
  kept distinct. It does not promise a target sparsity or claim that threshold
  overhead was directly benchmarked against top-k. External Q-Sparse/Spark and
  coding-agent citation accuracy is outside this local-kernel re-review and
  belongs to the parallel literature audit.

All identified points are resolved in the current sources. No material
scientific objection remains within the reviewed scope. Figures and numerical
results need no changes for this argument revision.

Resolution check also independently verified the current `main.pdf` SHA-256:
`dbbd594907b46a79e2f4bf668b925a20969181f02d66a8cedc8f20e3f2eb2e1a`.
This identifies the root agent's final build; this re-review checked source
meaning and file identity, not a new page-by-page rendering audit.
