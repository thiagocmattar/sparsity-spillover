# Scientific review — revised methodology

5 September 2026. Second independent scientific pass on the actual revised
`methodology.tex`, `methodology-appendix.tex`, `main.tex`, and
`methodology-notes.md`. The first version is preserved in the review folder's
`initial/` snapshots. Scientific code and manuscript files were not modified
by this reviewer.

## Scores

Unchanged anchors: **1 = blocking**, **2 = major revision**, **3 = competent
but uncompetitive**, **4 = strong**, **5 = exceptional**. Equal weights,
maximum 25. These scores assess the requested methodology stage, not the
completeness of deferred experimental-setup/results sections or paper
acceptance probability.

| Criterion | Revised | Initial | Assessment |
| --- | ---: | ---: | --- |
| Operational fidelity | 4/5 | 4/5 | Gate semantics, post-gate pressure, equal tensor weighting, clipping, AdamW state, stabilized conflict projection, required budget, and correction order remain faithful to the current implementation. Within-size matching is now explicit. This remains a strong compact operational account. |
| Metric and ceiling precision | 5/5 | 4/5 | The contradictory opening reach sentence is removed. Attention counter scope and global aggregation are explicit. The full declared multiplication denominator, dense-head treatment, causal validity, finite-precision zero inclusion, selected-site union/closure, and observed-versus-reach distinction are unusually precise. |
| Intervention identifiability | 5/5 | 5/5 | The independent gate/pressure sets, changing pressure normalization, recipe comparison boundary, and nonfactorial ladder interpretation remain explicit. No result or identification claim was strengthened without evidence. |
| Main/appendix reproducibility | 4/5 | 4/5 | The brief main preserves the necessary conceptual choices; the appendix carries their operational definitions and validation coverage. Immutable architecture-config links improve traceability. A small local index reuse is harmless but could be cleaned up. Condition-specific protocol values remain deliberately deferred. |
| Claim rigor for results | 5/5 | 5/5 | No speedup inference, quality-preservation guarantee, observed ceiling bound, or unsupported cross-size initialization match remains. Candidate conditions and analytic values are clearly distinguished from results. |
| **Total** | **23/25** | **22/25** | **Scientifically ready for this manuscript stage; no substantive unresolved issue or mandatory correction identified.** |

## Verification of requested fixes

- “To characterize the computational reach of a gate-site set, we define its
  all-zero reach ceiling” replaces the earlier statement that only selected
  sites determine which operations can receive zeros. It now agrees with the
  observed numerator's inclusion of natural zeros outside selected reach.
- “Within each model size, matched contrasts share initialization, data order,
  and training budget” correctly scopes the match.
- “Within one block and evaluation batch” now scopes the QK/PV formulas, and
  the following sentence states that the model numerator sums their counts
  over blocks and batches.
- The OL1 main paragraph names the adaptive task direction as the reference
  for the correction norm. The appendix retains the exact stabilized ratio
  and update definitions, so the concise main statement does not silently
  change the implementation.
- The local sparsity formula explicitly defines `|A|` as the tensor's element
  count. It is not a norm denominator.
- The three pinned config-link architecture names and immutable revision IDs
  match the local Run 013, Run 018, and Run 019 configs. Their accompanying
  sentence correctly distinguishes architecture configuration from released
  pretrained weights. This check establishes local provenance consistency;
  it is not a new remote-link availability test.
- The wrapper still defines stylized-S macros consistently and describes the
  ladder as candidate conditions, with no claim of exhaustive execution.

## Remaining optional notation cleanup

The local-sparsity paragraph uses b for an evaluation-batch index, while b in
the subsequent attention tensors indexes examples within one batch. Context
and the new explicit scope make both formulas interpretable, so this is not a
scientific error. If another editorial pass touches these equations, use e
for evaluation batches and retain b for examples. No additional formulas or
methodological discussion is needed.

## Boundaries for later manuscript work

The following are obligations of the later experimental-setup/results stage,
not defects to repair by adding invented specifics now:

- attach actual condition-specific seeds, schedules, pressure weights, trust
  budgets, precision, initialization recipes, and checkpoint-selection rules;
- keep A4/A7 pressure comparisons conditional on their distinct target sets;
- pair reported loss and logical sparsity from the same declared forward pass;
- distinguish any runtime workload from the full-sequence counting workload;
- establish empirical regularities and transfer with evidence rather than
  treating these definitions as a mechanism result.

The main section is brief without hiding the scientific distinctions needed to
read such results. The formal appendix is sufficient to understand and
reimplement the stated intervention and counting rules. No additional main-text
forward-graph derivation is necessary at this stage.

Only this review file was modified by this reviewer.
