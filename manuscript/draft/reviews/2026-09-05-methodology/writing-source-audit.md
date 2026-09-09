# Methodology source audit: writing and structure

Preparatory review of `manuscript/methodology.tex`, the revised introduction, and the operational definitions/methods/metrics. No scores assigned before the new methodology is available. No manuscript source edits made.

## Smallest useful main-section structure

Aim for approximately 500–650 words across three short subsections. The exact total is secondary to letting a reader interpret the intervention labels and the axes of the results without consulting formal appendices.

1. **Sites and controlled choices.** Introduce the parallel FFN and attention branches and refer to the intervention ladder. Name the seven site classes economically: branch inputs `a,m`, FFN hidden activation `h`, attention context `z`, and attention operands `Q,K,V`. Retain only placement distinctions essential to interpreting the study: the `h` gate replaces GeLU, `z` is before the output projection, and the focal Q/K gates act after RoPE. Distinguish the gate-site set from the pressure-site set. No full forward graph is needed in the main text.
2. **Gates and activation pressure.** Give inline scalar formulas for one-sided and symmetric thresholding, with nonnegative threshold and equality retained. State that their zero-threshold limits are ReLU and identity respectively; this difference is scientifically useful, not implementation trivia. Define the pressure objective as a mean of per-tensor mean absolute activations so that the weighting is clear. Briefly distinguish naive L1 from OL1: task-only AdamW moments, a preconditioned pressure direction with its conflicting adaptive component removed, then a bounded correction. Avoid promising unchanged convergence, fully orthogonal pressure, or removal of all pressure hyperparameters.
3. **Model-wide sparsity and architecture reach.** Define exact zeros, pooled integer counting, and the inline ratio for the stylized sparsity metric. State the counted operations, valid causal-pair restriction, full-sequence workload, and retained dense LM head in one compact passage. Define the all-zero selected-site reach ceiling using the same denominator. Make clear that observed zeros include natural zeros outside selected-site reach, so the ceiling does not universally bound the observed metric. A final sentence should link these quantities to the result interpretation: distinguish learned zero formation from the amount of computation exposed by placement; runtime is measured separately.

A one-sentence definition of quality as held-out next-token cross-entropy is useful if it is not introduced immediately in the experimental setup. Model sizes, training-token budgets, seeds, optimizer constants, and detailed validation coverage belong in that setup rather than being repeated here.

## Main-text essentials versus appendix material

| Main text: prerequisite for interpreting the result | Appendix: verification or derivation |
| --- | --- |
| Named sites and how to read ladder labels | Full Pythia forward graph with tensor dimensions |
| ReLU/one-sided versus symmetric thresholding | Gate backward semantics and boundary convention beyond a short main-text statement |
| Independently specified gates and pressure sites | Complete topology registry and per-site gate maps |
| Tensor-mean L1 and concise OL1 idea | Full OL1 moments, projection, cap, parameter-group handling, clipping and procedural definition |
| Exact-zero count-based model-wide sparsity | Indexed product-counter equations, overlap handling, causal multiplicities |
| Meaning and scope of selected-site reach ceiling | Per-token denominator algebra, topology ceiling numerators, deterministic closure proofs |
| Observed numerator versus topology-defined reach | Worked architecture-size tables and PRE/POST RoPE derivation |

## Notation usability

Use `\mathcal S_{\mathrm{model}}` and `\mathcal S_{\mathrm{model}}^{\max}` consistently if the stylized S is calligraphic. The old manuscript uses `\mathcal S` for the site set; avoid reusing it. `\mathcal G` and `\mathcal P` are economical for gate and pressure sets. Do not introduce block-level sparsity, a sitewise vector, or an architecture-utilization ratio unless the results actually need them.

Inline math can carry the short gate and sparsity definitions. Full indexed sums, cases, and multi-line derivations should move intact to an appendix rather than being squeezed into unreadable inline expressions.

## Risks in compressing the old methodology

- Its “compact main text” comment does not match its actual length: it contains most of the graph, counter algebra, topology table, and OL1 derivation before the appendix. Relocation is necessary, not merely shorter sentences.
- The old text includes an optional trust budget; the operational OL1 uses a positive trust budget. Use the executed rule.
- The phrase “measured removable compute” sounds more operational than a logical product count. Prefer “model-wide sparsity” while stating exactly which products are counted.
- A zero-coordinate count is neither a zero-product count nor a runtime measure. Compression must preserve these differences.
- The main text should explain why the definitions affect interpretation. Leave equation-by-equation proof and implementation provenance to the appendix.

## Scored-review rubric reserved for the new draft

Five criteria, each scored from 1 to 5: clarity and precision; progression to results; economy and main/appendix balance; notation usability; explanatory payoff. Anchors: 1 = blocking weakness; 2 = major revision; 3 = competent but uncompetitive; 4 = strong conference prose; 5 = exceptional. Total /25. A concise draft does not receive a high score if omitted definitions prevent interpretation of the results.
