# O015 - Fixed thresholds and specialized-kernel argument

## Question

How can the manuscript motivate fixed thresholding from prior top-k methods
and connect the resulting sparsity to specialized full-model execution?

## Method and coverage

Revised only the methodology, kernel subsection and kernel-results appendix,
using the existing bibliography and measurements. Independent agents checked
the primary Q-Sparse/Spark papers, audited retained kernel evidence and reviewed
the revised prose and rendered pages. The first two introduction paragraphs,
all other scientific sections, figures, numerical tables and bibliography
are unchanged. No experiment or figure regeneration was performed.

## Argument and result

- Fixed thresholds avoid ranking and input-dependent moment estimates. They
  extend elementwise nonlinearities to Q/K/V coordinates, allowing training
  to change the retained activation fraction. Distributional adaptation is
  tested by the existing results rather than assumed as a guarantee.
- The unchanged Sakana control accelerated its supported SparseLM/H100
  workload by 1.3001x, but the Pythia-14M primitive failed numerical agreement
  before full-model timing. The adapted P0 qualifies on four of 30 checkpoints
  at geometric mean 0.8910x on that subset. These are distinct comparisons.
- The human-guided search's matched retrospective reaches 1.6024x at iteration
  11 and 1.7830x at iteration 42. K050 qualifies on all 30 checkpoints at
  geometric mean 1.2340x; its approximate linear association with model-wide
  sparsity has OLS R-squared 0.8167.
- Attention skipping is slower on all 30, despite substantial MMA instruction
  skipping at the search checkpoint. A skip requires either complete operand
  fragment to be zero, not matching zeros in both operands. The measured
  limitation concerns profitable hardware granularity and recurring overhead.
- Corrected an existing appendix coverage statement: the historical replay
  uses four checkpoints; the final sweep uses 35, with 30 retained for this paper.

## Caption and figure goal

Figure 07's artwork is unchanged. The manuscript caption now explicitly calls
panel (a) a matched retrospective. Its goal remains connecting logical sparsity
to measured full-model acceleration while distinguishing fusion and sparse-path
effects. No other figure or caption changed.

## Caveats

No direct speed comparison against Q-Sparse or Spark was run. Top-k does not
universally require a full sort; Spark uses a distribution-based approximation,
with different operators and attention-selection sites. The kernel curve shows
early gains and later refinement, not an established optimum. The regression
is descriptive across different checkpoints, not a causal or universal scaling
law. The agent-assisted trajectory is not a comparison of agent capabilities.
No joint-zero profitability threshold has been demonstrated.

## Verification and provenance

- Clean 27-page build; all references resolve, no LaTeX/box warnings, only
  embedded non-Type-3 fonts. Rendered changes were inspected, and unchanged
  page bodies compared against the previously inspected copy.
- All eleven figure copies, seven numerical table files, fourteen measurement
  copies and thirty protocol-source hashes remain intact.
- Final PDF SHA-256:
  `75b35b3dfcc4dd49b0c826ad6e218bd0d265ed2e04a0df847818e6d74ff5de05`.
- [Source/PDF snapshot](../provenance/manuscript-20260908-threshold-kernel/README.md),
  with argument map, independent reviews, file hashes and build verification.
- Source observations: O007 and O014; Run 022 README; Run 025 SOURCE_AUDIT.md;
  Run 029 observations 01, 03, 04 and 06. Current runtime data and progress:
  `figure_data.json:runtime`, reduced by `evidence.py:runtime_subset`.
- Literature: [Q-Sparse](https://arxiv.org/html/2407.10969v3) and
  [Spark Transformer](https://arxiv.org/html/2506.06644v2), checked against
  primary sources. The manuscript retains existing citation keys.
