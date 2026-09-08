# Experimental-section scope and provenance

5 September 2026. The user approved separating methodological definitions
from the concrete study. `experimental-study.tex` introduces the setup,
ladder recipes, and the comparisons they support. It contains no empirical
finding, result figure, or new experiment design.

## Sources

- [Analysis 013](../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md)
  supplies the matched-comparison contract and its identifiability limits.
- [Analysis 011](../../analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/README.md)
  supplies the selected larger-model scope. The reported ladder families are
  A0, A1-H, A4-OL1, and A7-OL1 at 70M/410M, not all eight rows at every size.
- [Operational methods](../../research/METHODS.md) identify the ports and
  gate semantics; [data](../../research/DATA.md) identifies validation packing.
- [Methodology notes](methodology-notes.md) retain the exact architecture
  configuration pins, scalar-multiplication units, and operational R-to-paper-S
  crosswalk. The linked configuration identities remain unchanged.
- The figure is the existing
  [architecture and ladder](../artifacts/pythia-architecture-sparsification-ladder.md).
  Only its location in the reading copy changes in this revision.

## Interpretive boundaries

Ladder rows are separate pretraining conditions. They do not specify a
sequential curriculum, monotone improvement, or complete factorial experiment.
The detailed 14M study and selected larger-model extensions are distinguished.
There is one initialization seed per size; threshold levels are not replicates.
Within-size matched comparisons preserve initialization, realized data order,
and training budget. Cross-size runs have fixed token exposure but different
learning rates, so the prose does not claim one identical protocol across sizes.

A4-OL1 versus A7-OL1 changes the pressure target set and equal-tensor
normalization as well as the gate set. At zero threshold, added symmetric
gates are identity maps; the pressure objectives still differ. The exact
boundary behavior is preserved in the appendix. Comparisons of these recipes
do not isolate gate placement at a fixed pressure objective.

Training precision, optimizer values, and condition-level schedules are
deferred to the results cohort's reproducibility details. The source audit
notes that executed cohorts used FP16 loss scaling and explicit parameter
grouping; generic operational BF16/default descriptions must not be copied
as executed settings. This revision asserts neither default.

## Section responsibilities

| Location | Content |
| --- | --- |
| `methodology.tex` | Gate/pressure definitions, exact-zero measurement, and selected-site reach |
| `experimental-study.tex` | Model/data context, A* recipes, figure, comparison logic and coverage |
| `methodology-appendix.tex` | Gate derivatives, pressure mechanics, exact counters and general normalization |
| `experimental-appendix.tex` | Pythia ports, architecture-specific counts/ceilings, immutable config links and validation coverage |

The original equations and labels are retained on relocated material.
Pythia's equal-width multihead denominator is explicitly an experimental
architecture specialization; it is not presented as universal to transformers.

The [review record](reviews/2026-09-05-methods-experiments-split/README.md)
contains the independent writing proposal, source audits, two scored passes,
and the response to criticisms. Results remain a later drafting stage.

## 8 September 2026: kernel realization case study

The user requested an implementation/claim audit followed, if supported,
by a brief systems subsection. `kernel-autoresearch.tex`, included at the
end of the experimental opening, reports the qualified Run029 result with
the run-owned two-panel PDF. Its operational `R_model` is the manuscript's
`S_model`, retained in FP16 rather than redefined by BF16 timing counters.
`kernel-implementation.md` provides the source excerpt, exact runtime and
numerical contract, ablations, known defects and provenance limitations.
The source observation is Run029 `observations/06-implementation-and-claim-audit.md`;
independent reconstruction is `19_audit_evidence.py` and
`results/implementation-audit-001.json`. No kernel, checkpoint or figure data
was changed, no GPU was relaunched, and no general scaling law is asserted.
