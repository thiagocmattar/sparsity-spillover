# Revision decisions and completion audit

This revision follows the attached manuscript-review request and subsequent
terminology and framing instructions. The paper was edited in successive
subsection passes, not rewritten as a single replacement. The original reading
copy and sources were backed up before editing.

## Passes and decisions

1. **Introduction.** Preserved the user's first two paragraphs exactly, including
   their citation entries. Tightened the remaining paragraphs around the question,
   model-wide accounting, matched study and contribution. Clarified that this is
   a component study, not a reproduction of three complete prior methods.
2. **Related work and methods.** Verified all 15 references against primary
   records. Corrected Sparsing Law's scope, credited the projection antecedents
   at OL1's first mention and updated five same-year conference records. Kept
   the protected paragraphs' preprint citations unchanged. Equations and
   operational estimands were checked against source implementations.
3. **Setup.** Added a compact five-row recipe key and moved shared validation
   coverage into one main-text protocol statement. Preserved the matched
   initialization, order, budget and exact comparison boundaries.
4. **Results, one subsection at a time.** Each opens with an insight: different
   quality costs near similar sparsity; conditional pressure effects; more FFN
   sparsity need not produce more model-wide sparsity; high-threshold recipe
   ordering across sizes; conditional sparse-path benefit on hardware.
   Shortened each caption while keeping its display and denominator conventions.
5. **Kernel interpretation.** Defined SDPA, qualification, K049 and P0; restricted
   “best” to searched proposals. Reported the sparse/fusion speed ratio explicitly
   and retained the faster attention-dense ablation. No hardware extrapolation.
6. **Appendix and data.** Preserved every endpoint, paired comparison, histogram
   group, operation contribution and clipping trajectory. Added exact preparation
   and software provenance in a separately sourced protocol JSON. The existing
   14 measurement copies remain byte-identical. Kept the scoped off-cohort kernel
   defect in the protocol's known limits.
7. **Abstract and conclusion.** Added the missing opening summary and synthesis.
   Following the user's steering, put the contribution first and explain the
   support from agreement across matched contrasts, parameter sweeps,
   distributions, product counts, model sizes and kernel ablations. Seed and
   validation-reuse details appear in the appendix protocol, not the abstract or
   repeated captions. Treatment variability is not described as seed replication.
8. **Terminology.** Use activation sparsity, model-wide sparsity, sparsity ceiling,
   thresholding and nonlinearities. Preserve the ceiling's guaranteed-product
   definition at 100% activation sparsity of selected sites. Use common-A7
   `U_arch` consistently. Operational data keys and historical records retain
   their original names. No figure label contained the rejected words, so no
   artwork needed modification.
9. **Layout and checks.** Added a float barrier before the kernel subsection and
   embedded vector monospaced fonts for appendix filenames. Clean LaTeX build;
   all ten figures and ten tables are present. All 26 final pages were inspected:
   technical-reader agent pages 1–13, root pages 14–26. No clipping, overlap,
   unreadable symbol, missing reference or box warning remains. Some float and
   final-page whitespace is retained; template fitting was explicitly excluded.

## Review resolutions

The initial technical-reader, scientific-reviewer and literature-audit reports
are retained beside this log. Follow-up reports assess the revised argument,
terminology and final rendering. All substantive requested revisions were
resolved. The new recipe key was accepted on rereview; the original overview
remains selected because the paired figure and full tables already contain all
variants. The all-variant v2 remains available as an alternative.

No scientific numerical error was found. The evidence audit independently
reconciled 54 source endpoints, 29 matched contrasts, all 540 clipping
evaluations, 14 histogram groups, common-A7 normalization and runtime ablations.
The formatter additionally checks all 44 paired comparisons, including the
15 scale comparisons. All numerical table rows are unchanged from the preceding
version. The only formatter changes concern terminology in two table headers
and the endpoint caption.

## Requirement-by-requirement closeout

| Requirement | Current evidence |
| --- | --- |
| Read and follow the attached task | Revision plan, independent reviews and successive passes above |
| Preserve first two introduction paragraphs | Exact comparison plus saved paragraph hashes in `protected-introduction.json`; cited entries also unchanged |
| Preserve the argument while improving clarity | Same five-part result sequence; reader rereview; added opening/closing synthesis |
| Review every figure/table/result | Exhaustive exhibit checklist, scientific claim audit, all ten figures/ten tables verified |
| Simulate technical readers and reviewers with sub-agents | Technical reader, scientific reviewer and literature auditor reports plus follow-ups |
| Distill results into concise insights | Each results subsection and its caption leads with a distinct supported finding |
| Standardize terminology | Source and rendered-text searches have no “gate(s)” or “all-zero”; definitions and table notation checked |
| Strengthen contribution without overemphasizing seed scope | Revised abstract/introduction/conclusion; seed disclosure confined to appendix protocol |
| Verify publication content and rendering | `verification.json`, source hashes, resolved citations/labels, final main and appendix visual reviews |
| Preserve evidence and reproducibility | Unchanged artwork/measurement hashes; new protocol provenance; new analysis-owned source/PDF snapshot |

The content is ready for submission preparation within the stated study scope.
This is not a prediction of peer-review acceptance. Conference-template fit,
author metadata and actual submission were outside the requested task.
