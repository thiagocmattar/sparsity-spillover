# First methodology review: provenance, notation, and presentation

Independent adversarial source review, 2026-09-05. Read the new main methodology, appendix, wrapper, introduction, provenance notes, bibliography, and changed ladder sources/companion notes. Compared definitions with operational methods/metrics/data and inspected the pressure/evaluation implementation where relevant. No manuscript edits or experiments. This pass evaluates source content; final PDF layout inspection was still pending.

## Rubric and score

Equal weights. Anchors: 1 = blocking, 2 = major revision, 3 = competent but uncompetitive, 4 = strong, 5 = exceptional. Scores assess this methods-writing stage, not acceptance prospects for a paper without its result section.

| Criterion | Score | Assessment |
|---|---:|---|
| Citation/definition provenance | 4/5 | Operational definitions are faithfully separated from contextual citations; the three “pinned” architecture tuples need a more explicit source in the paper. |
| Main/appendix progression | 4/5 | Main text gives the scientific distinctions, appendix gives executable semantics. Forced placement detaches the figure from its explanation. |
| Notation/figure consistency | 4/5 | Calligraphic S is consistent and units are explicit. Small terminology/caption issues remain. |
| Workload/systems interpretation | 5/5 | Exact products, excluded work, full-sequence scope, dense head, natural zeros, and reach limits are unusually clear without claiming runtime benefits. |
| Reader utility/economy | 4/5 | The section is compact and keeps protocol detail subordinate to interpretation; two figure edits would improve the reading path. |
| **Total** | **21/25** | **Strong foundation; focused presentation/provenance corrections remain.** |

## Substantive checks that pass

The introduction's individually controllable choices are implemented in the main text as separate gate and pressure sets. A4-OL1 versus A7-OL1 is explicitly described as a recipe comparison that changes both sets. The text does not imply a fully crossed design, execution of every candidate row at every size, or a new result.

The new S notation preserves the existing measurement rather than changing the estimand. Observed S pools exact-zero multiplication counts; the LM head adds denominator only. Smax is selected-site all-zero reach and explicitly does not bound every observed S. A0 is used to make the distinction concrete. Fraction/percentage conversions are consistent across equations, wrapper caption, and figure values. These match [DEFINITIONS](../../../../research/DEFINITIONS.md) and [METRICS](../../../../research/METRICS.md).

The exact port descriptions agree with [METHODS](../../../../research/METHODS.md): a/m follow separate LayerNorms, h replaces the original FFN activation, q/k follow partial RoPE, v follows the QKV split, and z precedes the attention output projection. The appendix usefully distinguishes the derivative at zero of ReLU and the fixed one-sided threshold gate. It does not describe these masks as STE or top-k.

The pressure average agrees with [`pressure.py`](../../../../src/sparsity_research/pressure.py): float32 absolute means, equal weight per captured tensor, task-only moments for OL1, global conflict test, stabilized projection, and a required positive budget. The references to PCGrad and Bloop provide intellectual context without transferring guarantees. AdamW's new entry matches its [primary record](https://arxiv.org/abs/1711.05101).

The attention OR masks prevent double credit, causal exclusion avoids scoring future masked positions as sparsity, and PV accounting retains legitimate underflow zeros. The value-to-context closure is restricted to the all-zero limit. The evaluation text separates 2047 loss targets from 2048 forward input positions and correctly gives complete MiniPile coverage. No decoding-latency claim is smuggled into full-sequence accounting. See [`evaluation.py`](../../../../src/sparsity_research/evaluation.py) and [DATA](../../../../research/DATA.md).

## Ranked concrete corrections

### 1. Give the architecture tuples explicit source provenance

The appendix calls `(6,128)`, `(6,512)`, and `(24,1024)` “pinned” without identifying the configurations or a source table. The [Pythia paper](https://arxiv.org/abs/2304.01373) describes the original released suite beginning at 70M; it should not alone bear the provenance burden for the 14M tuple. The manuscript's prose correctly says randomly initialized architectures, so this is a reproducibility/source issue, not a mistaken continuation-training claim.

Add a compact appendix footnote or source table identifying the three architecture configuration names and immutable revisions, or explicitly defer those identities to a linked experimental-configuration appendix. Reuse already verified local identities; do not discover new architecture variants or add a cross-family analysis. The existing [ladder notes](../../../artifacts/sparsification-ladder.md) already give the dimension tuples and are the immediate analytic source.

### 2. Place the ladder with its methods explanation where practical

`main.tex` currently inserts a forced page break after the entire methodology, then the figure. The intervention subsection refers the reader to that map for all seven sites, so a detached page makes a compact section harder to follow. Prefer declaring the figure near the intervention subsection and allowing an appropriate top/bottom placement, then inspect the actual PDF. This is a layout recommendation, not a request to enlarge the figure or violate readability to save a page. If the composite requires a dedicated page at readable type size, retain that page and ensure the methods/appendix transition makes its role clear.

### 3. Make the figure's terminology self-contained

The main text says “Naive L1” and the diagram says “L1N.” Define `L1N` in the legend or caption once. Remove “corrected” from “In the corrected A4-OL1 and A7-OL1 recipes”: that word refers to repository history and has no scientific meaning for a reader encountering these methods for the first time. The rest of the caption correctly states the pressure sets and mixed comparison.

### 4. Use a conventional table caption position

The appendix site table places its caption below the table. Move its caption and label above the tabular body if following the common conference-paper convention of table captions above and figure captions below. This is low priority and does not affect mathematical validity.

## Optional polish, not blockers

- “The selected sites determine which operations can receive zero operands” can be read more broadly than the immediately following reach definition. “The selected sites determine which operations the gates can reach” avoids suggesting natural zeros cannot occur elsewhere.
- The ladder still uses color alone to distinguish dot categories. A redundant symbol would improve grayscale accessibility if it can be added economically; assess the final rendering before making a larger visual change.
- The exact source configuration and future result-specific optimizer/seed table are not the same obligation. It is appropriate to defer schedules and selected cohorts while writing methods gradually; do not invent values merely to make the appendix look complete.

## Figure decision

No new ceiling chart is needed. The ladder already provides the analytical values for all three declared architectures, and this draft adds the previously missing denominator/reach explanation. A repeated ceiling plot would spend space without answering a new question. No cross-family experiment or result should be inferred from an optional suggestion to visualize architecture dependence.

## Second-pass acceptance check

Verify that the final PDF preserves readable mathematical subscripts and table widths, that displayed ceiling notation is S throughout the actual embedded PDF, and that figure placement supports the intervention text. Confirm architecture provenance and L1N decoding are resolved or explicitly deferred to a named source. No additional literature or systems mechanism is required to make this section scientifically interpretable.
