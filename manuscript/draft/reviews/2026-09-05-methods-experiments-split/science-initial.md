# Scientific review — initial methods/experiments split

5 September 2026. Reviewed the actual split manuscript: `methodology.tex`,
`experimental-study.tex`, both appendices, `main.tex`, and
`experimental-notes.md`. Compared against the source audit, operational
contracts, and retained setup evidence. No active manuscript was edited.

## Rubric and scores

Equal 1–5 criteria: **1 blocking**, **2 major revision**, **3 competent**,
**4 strong**, **5 exceptional**. Maximum 25. This review assesses the structural
manuscript stage and does not demand unrequested results or select new runs.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Operational fidelity | 4/5 | Relocation preserves post-gate tensor-weighted pressure, gate equality/derivatives, OL1 clipping/state/projection/budget, and exact site semantics. Actual experimental precision and optimizer settings are not replaced by generic defaults. |
| Metric/reach precision | 4/5 | The counters, dense-head denominator, causal validity, Pythia denominator, and reach table remain correct. The newly generalized reach sentence needs to distinguish structurally guaranteed zeros from coincidental zeros in a hypothetical clamped forward pass. |
| Contrast identifiability | 5/5 | Separate pretraining conditions, matched same-size contrasts, conditional pressure effects, A4/A7 objective changes, and the zero-threshold identity nuance are preserved. The structural split improves access to these comparison limits. |
| Setup/coverage honesty | 4/5 | The concrete recipe family, selected larger-size coverage, one-seed scope, and candidate-versus-executed distinction are clear. Fixed-token exposure is a key transfer qualification present only in editorial notes; it should appear briefly in the manuscript. |
| Claim rigor | 4/5 | No result or speedup is invented. One sentence slightly overstates what inspecting aggregate sparsity alongside reach can distinguish about learned behavior. The definitions can inform that analysis but do not alone identify its causes. |
| **Total** | **21/25** | **Strong split; one formal wording correction and two compact scope/interpretation improvements remain.** |

## Ranked issues and concrete fixes

### 1. Define reach by guaranteed consequences of selected-site zeros

**Text:** “take the union of products that have a zero activation operand
when the selected sites are all zero, including exact propagation through the
declared graph.” [`methodology-appendix.tex`](../../methodology-appendix.tex)

This is broader than the implementation if interpreted as counting every zero
in a particular hypothetical clamped forward pass. Such a pass can include
natural zeros not caused or guaranteed by the selected sites. With no selected
sites, the condition is vacuously satisfied, although A0's selected-site reach
must be zero. The later natural-zero caveat states the desired distinction but
does not completely remove the ambiguity in the definition itself.

The implementation in [`ceilings.py`](../../../../src/sparsity_research/ceilings.py)
uses a declared structural reach union, independent of checkpoint values. For
example, all-zero V guarantees a zero context, whereas merely observing zero
QKV biases in some checkpoint does not enlarge the topology's reach.

**Fix:** “Take the union of products guaranteed to have a zero activation
operand as a consequence of setting the selected sites to zero, using the
declared graph's structural propagation rules rather than checkpoint values.”
Keep the existing once-only counting and explicit Pythia specialization. This
is the only must-fix formal precision issue found.

### 2. State fixed-token scope in the experimental manuscript itself

**Text:** “The 70M and 410M extensions cover A0, A1-H, A4-OL1, and A7-OL1.”
[`experimental-study.tex`](../../experimental-study.tex)

The sentence states family coverage correctly. However, the reader sees
transfer across sizes as a study objective without learning that these are
fixed-token extensions. This limits the meaning of transfer independently of
the numerical optimizer details. The editorial notes correctly record this
qualification, but they are not part of the rendered scientific argument.
[Analysis 011](../../../../analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/README.md)
states one MiniPile pass per size and equal processed token counts, alongside
different learning rates at 410M.

**Fix:** “The fixed-token 70M and 410M extensions cover ...” or “The 70M and
410M extensions use the same token exposure and cover ...”. Do not say all
training settings are matched across sizes. This requires only a few words,
not a schedule table or new experimental-design decision.

### 3. Do not imply that ceiling comparison alone separates learned response

**Text:** “Across sizes, observed sparsity is interpreted alongside
architectural reach to distinguish changes in learned behavior from changes
in the fraction of computation reached by the same sites.”
[`experimental-study.tex`](../../experimental-study.tex)

The intended analysis is scientifically useful. But observed aggregate
sparsity plus a selected-site ceiling does not uniquely decompose learned
behavior from architecture effects, especially when observed counts include
natural zeros outside reach. Operation-level contributions and activation
changes provide the more direct evidence. The current wording reads as an
analytic objective rather than a completed result, so this is modest
overstatement rather than a fabricated finding.

**Fix:** “Across sizes, we interpret operation-level sparsity alongside
architectural reach, so changes in the fraction of reachable computation are
not mistaken for stronger learned sparsity.” Alternatively replace “to
distinguish” with “to help interpret”, without claiming identification from
the aggregate pair alone. No new result is required.

## Optional scope cleanup

The Methodology section now functions as general definitions, but its N0
inventory still lists fused QKV and exactly two FFN projections, which is the
Pythia graph's declared inventory. This is scientifically valid as the paper's
chosen metric scope; it should not be presented as universal to all Transformer
architectures. The notes already recognize this limitation. If further
shortening, define N0 over “the declared block matrix products” in Methodology
and leave the exact six-family inventory in Experimental Study/Appendix.
Otherwise a simple “For the graph studied here” makes the scope explicit.

## Semantic checks that passed

- The site table is now appropriately attached to the actual Pythia setup;
  generic pressure and gate formulas no longer depend on A* recipe names.
- Counter formulas refer to the specified multihead graph, not arbitrary
  grouped-query, gated-FFN, or other architectures.
- Pythia's all-zero V closure is retained without inferring partial context
  masks or ignoring affine biases.
- Fractional units and the S notation remain unchanged; operational artifact
  names are preserved in the editorial crosswalk.
- The attention formulas remain scoped per block/batch and pooled globally.
- The main and appendices say that the dense head contributes only to the
  denominator; normalization, softmax, RoPE, additions, and memory traffic
  remain outside the counted workload.
- The selected larger-model recipe list is accurate. The figure caption
  explicitly avoids turning analytic columns into evidence of complete
  experimental coverage.
- The zero-threshold A4/A7 forward-map equivalence is stated in the concrete
  appendix while the pressure-objective distinction is preserved.
- Shared initialization is scoped within size, and dose levels are not
  called replicates.
- No stock pretrained weights, full factorial, exact loss-preserving OL1,
  complete kernel transfer, or achieved scientific result is newly claimed.

The main definitions remain understandable before the recipe map. The new
Experimental Study gives enough setup and comparison logic for the present
argument-development stage. Condition-specific settings and complete cohort
coverage can accompany the later result sections as already stated.

Only this review file was modified by this reviewer.
