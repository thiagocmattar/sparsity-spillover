# Separate methodology from experimental setup

User-authorized structural revision, 5 September 2026. The root agent edited
the manuscript using an independent writing proposal and scientific/presentation
source audits. Three sub-agents then reviewed the actual draft twice. They did
not edit active manuscript files. No empirical results were drafted.

## Review rubric

Five equally weighted criteria per perspective, each scored 1-5:
**1** blocking; **2** major revision; **3** competent; **4** strong;
**5** exceptional. Reviewers received explicit instructions to justify scores
with precise text, rank necessary changes, and distinguish prose-fixable issues
from evidence limitations. Second-pass scores were independent assessments,
with no target score or assumption of improvement.

| Perspective | Criteria | Initial | Revised |
| --- | --- | ---: | ---: |
| Writing | Clarity/precision; section separation; question-driven progression; economy; reader payoff | 20/25 | 20/25 |
| Science | Operational fidelity; metric/reach precision; contrast identifiability; setup/coverage honesty; claim rigor | 21/25 | 23/25 |
| Literature/presentation | Citation/provenance; methods/experiment boundary; notation/figure consistency; workload/coverage interpretation; utility/economy | 22/25 | 23/25 |

Scores assess the present writing stage, not acceptance probability or the
strength of eventual findings. No pooled score is reported.

## Independent work and reports

- [Writing proposal](writing-proposal.md): an unscored, independent draft of
  the experimental opening that informed the active text.
- Source audits: [science](science-split-audit.md) and
  [presentation](presentation-split-audit.md).
- Initial reviews: [writing](writing-initial.md), [science](science-initial.md),
  [literature](literature-initial.md).
- Revised reviews: [writing](writing-revised.md), [science](science-revised.md),
  [literature](literature-revised.md).
- Source and PDF snapshots: [before/](before/), [initial/](initial/), and
  [after/](after/), with [SHA-256 inventory](source-hashes.json).

## Changes and response to criticism

1. **Separate definitions from instantiation.** Methodology now contains two
   subsections: interventions; sparsity and architectural reach. Pythia/MiniPile,
   the named ports, all A* recipes, and matching/coverage move into Experimental
   Study. The figure follows its opening paragraph. The study is described as
   separately pretrained conditions, not a sequential curriculum.
2. **Carry the split through the appendix.** Gate derivatives, pressure
   mechanics, exact counters, and general normalization remain methodological.
   The Pythia site table, six-operation inventory, equal-width multihead
   denominator, A* reach counts, architecture pins, and validation coverage are
   in an experimental appendix. Existing numbered equations retain their labels
   and content; no graph-specific denominator is presented as universal.
3. **Preserve independent controls.** Pressure uses the executed site tensor,
   after a gate when present. This avoids implying that pressure targets must
   be gated. The final wording also permits a different gate family at each
   selected site, as required by A7.
4. **Make reach structural.** The first generalization could be read as counting
   every zero in a hypothetical all-zero-site forward pass. It now counts only
   zeros guaranteed by the selected-site condition and declared propagation,
   independently of checkpoint values or natural zeros elsewhere.
5. **Clarify the experimental question and scope.** The opening explicitly
   concerns quality and model-wide sparsity. Within-size matching, selected
   larger-model coverage, the common training-token budget, and one-seed scope
   are stated. Threshold settings are not treated as statistical replicates.
   A4-OL1 versus A7-OL1 retains its combined gate/pressure-objective limitation.
6. **Avoid overinterpreting aggregate measures.** Cross-size interpretation now
   examines each operation's zero-product fraction and workload share alongside
   aggregate sparsity and reach. Aggregate sparsity plus its ceiling alone is
   not claimed to identify a learned mechanism.

The final optional wording pass expanded "fixed-token" into the explicit
training-token-budget match and clarified per-operation fractions/workload
shares. No new experiment, result, or implementation choice was introduced.

## Verification and remaining scope

The reading copy compiles without undefined citations/references or overfull/
underfull boxes. All 15 citation keys resolve, and labels are unique and complete.
The four original numbered equations match the pre-split source after whitespace
normalization. Main Methodology contains no A* stages, Pythia, or MiniPile.
The experimental source owns the figure; its artwork and ceiling values are
unchanged. Introduction, related work, and bibliography are unchanged.

Main Methodology is about 380 words, down from about 560. Experimental Study is
about 320 words excluding the existing caption. The nine-page reading copy puts
methods on page 3, the experimental opening/figure on page 4, comparison logic
on page 5, references on page 6, and appendices on pages 7-9. All pages were
visually checked. Page 5 is deliberately a short continuation point for the
later results; the figure was not shrunk to force an artificial final layout.

Empirical findings, result ordering, and condition-level reproducibility
tables remain for the next writing stage. No scientific code or result was
changed and no experiment was launched. Draft sources and review records remain
local under the existing ignore policy; the tracked manuscript index and figure-provenance notes record
this coherent change.
