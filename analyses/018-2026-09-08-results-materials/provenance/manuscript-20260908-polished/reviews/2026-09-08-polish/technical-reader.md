# Technical-reader review

Reviewed the current 24-page reading copy and all included TeX, ten figures,
nine tables, and the operational definitions/data/manuscript context on
8 September 2026. PDF SHA-256 at review:
`1d2cd69782f8adbe0e6c267412d3ad5555ba2e9b38c6cd8cddf1aed075cdba8e`.
All pages were rendered and inspected; paired effects, densities, scaling,
and the architecture diagram received larger renders. This is a reader and
exhibit review, not an independent numerical reconstruction or literature audit.
No manuscript or artwork was changed. The first two uncommented introduction
paragraphs must remain byte-for-byte unchanged.

## Scores and decision

Scores use 1 = blocks understanding, 2 = substantial revision, 3 = usable
with noticeable friction, 4 = clear with focused improvements, 5 = immediately
clear and self-contained at the intended reading scale.

| Dimension | Score | Main reason |
| --- | ---: | --- |
| Sentence clarity | 4/5 | Definitions are careful and results concrete; abstract nouns and repeated scope statements still lengthen the route to each conclusion. |
| Narrative sequence | 4/5 | Overview, paired effects, distributions, size transfer, and kernels form a coherent argument. An abstract and final synthesis are missing. |
| Figure integration | 3/5 | The right five main figures are selected, but long captions and float placement interrupt the result sequence; the intervention key lives far away. |
| Table utility | 4/5 | The compact scale table resolves the central comparison; full tables support auditing. Kernel identifiers and grouping need small clarification. |
| Self-containedness | 3/5 | The scientific scope is present, but SDPA/MMA/P0/K049 and accompanying-data access need clearer identification. |

The paper has a strong, usable structure. It needs editorial completion and
tighter result interpretation, not a replacement structure or figure redesign.
The five main figures each answer a distinct question; retain all five.

## Recommended editing sequence

1. Preserve the first two introduction paragraphs exactly. Edit the remaining
   introduction one paragraph at a time, then read the complete introduction
   before proceeding. The tone to match is concrete subject, active verb,
   explicit question, and a stated consequence.
2. Tighten related work without changing its two-subsection structure. Then
   resolve methodology and setup definitions before touching result prose.
3. Edit each result subsection as one small unit: opening insight, evidence,
   interpretation, relevant boundary, caption. Verify the cited numbers and
   render before moving to the next unit.
4. Add a short abstract and a short closing synthesis/limitations section after
   the results. This completes the paper without reorganizing its core.
5. Finish appendix definitions, exhibit labels, and data availability; render
   the whole document to check reading order and the final ending.

## Paragraph and subsection recommendations

### Introduction after the protected paragraphs

- Paragraph 3 should say directly what is compared under a common setup.
  Readers could otherwise infer that the study reruns the full TEAL,
  ProSparse, and Q-Sparse methods named in the protected paragraph. Define
  pressure, thresholding, and placement; make clear these are the controlled
  design choices. The italic sentence about separately controllable but
  inseparable effects restates the following question and can be removed.
- Paragraph 4 has a useful distinction between small values, exact zeros,
  and affected work. Retain that example, but compress its final three
  sentences, which restate the same diagnostic purpose.
- Paragraph 5 should define model-wide sparsity and the selected-site reach
  reference once. The sentences beginning “This accounting” and “It helps”
  repeat paragraph 4. End with the reason direct timing is still necessary.
- Paragraph 6 repeats the ladder's role twice. State the matched pretraining
  design and the three sizes. Refer to the architecture map as an appendix
  aid, rather than making the distant figure sound essential to following
  the opening argument.
- Paragraph 7 is the right place for the result preview. Keep one sentence
  each for conditional pressure effects, cross-size recipe ordering, and
  scoped hardware realization. Avoid making the aggregate kernel gain sound
  attributable only to activation sparsity; the ablation is part of this result.

### Related work

The existing pressure/placement/execution sequence is helpful. Preserve the
concrete differences among post-hoc clipping, pretraining/continuation, and
projection versus internal-attention sparsity. The last inference-speedup
paragraph repeats “complete-model timings” and overhead accounting already
stated earlier. End once with why this paper measures the complete workload.
“Offers an emerging tool” and similar generic framing can be shortened.
Ensure every comparison remains about the cited method's actual workload;
the reader should not infer the paper's uniform clipping control reproduces
TEAL's optimized allocation.

### Methodology and experimental setup

- The gate definitions and pooled-count metric are necessary and readable.
  Keep the difference between ReLU and zero-threshold derivative semantics
  in the appendix.
- “Positive budget” and later “trust budget” should use one term and one
  short definition: a bound on pressure-update norm relative to the task
  direction. Appendix A.2 supplies the equation.
- A “ceiling” that is not a bound on every observation is unintuitive. Keep
  “selected-site reach” immediately attached to the term and give the existing
  A0 example once in the appendix. Do not repeat the full exception in three
  places.
- The setup already defines A0/A1-H/A4/A7; retain that compact key close to
  Figure 1. Clarify that `-L1`/`-OL1` names the pressure update. A new large
  main table or relocation of the approved diagram is unnecessary.
- The matching paragraph does too much: matching, causal interpretation,
  coverage, optimizer budget, and scale accounting. Split at the transition
  to larger models. State one seed once here, then refer to it economically.
- Put the full 500-document/338-block/1,444-tail contract in one visible setup
  statement and the evaluation appendix. Repeating it in every figure caption
  adds little reader value. Retain the precise numbers in the paper.

### Results

- **4.1:** Open with the qualitative separation between local pressure's
  low-loss regime and broader gates' higher-sparsity regime. The close-sparsity
  A1-H-OL1 versus A0-clipping example is useful. Three decimals for percentages
  and four for loss are more than needed in running prose; retain precision
  in tables. Explain that different site reach prevents treating the complete
  recipe comparison as an isolated pressure result.
- **4.2:** The two main insights are that OL1's benefit is not uniform, and
  that adding pressure depends on threshold and sites. Keep representative
  contrast pairs, but avoid a string of six numbers without an intervening
  interpretation. Distinguish changing gates alone from changing the pressure
  targets and their normalization once, at the conclusion of this subsection.
- **4.3:** This is the clearest mechanistic bridge. Keep the contrast that
  A4-OL1 has more pooled FFN zeros while A7-OL1 has greater model-wide sparsity.
  Define spillover as a change at directly untargeted attention sites, and
  retain the qualification that the A0 comparison does not isolate pressure.
  The 16.77 versus 0.06 pp attention contribution makes the distribution result
  consequential. Do not remove the omitted-zero explanation when shortening
  the caption; it prevents a major misreading of the blue central peak.
- **4.4:** The main claim is specific and supported by the displayed table:
  the high-threshold complete-recipe ordering persists, whereas low-threshold
  ordering changes. Introduce common A7 normalization in plain language before
  the formula. State that it removes the dense-head share, not all architectural
  differences. The 410M baseline has worse loss than 70M at this budget; avoid
  leaving readers to interpret the plot as quality scaling with parameter count.
- **4.5:** Expand scaled dot-product attention (SDPA) on first use. Define
  “qualified” briefly before quoting qualified speedup. Put the selected K050
  result next to the sparse-path ablation, so fusion is not mistaken for a
  sparsity effect. The attention-dense variant being faster on every checkpoint
  is an essential finding. “Best kernel” in the approved artwork can remain,
  but the caption must identify panel (b) as K050 and point to the better
  attention-dense ablation. Do not suggest universal superiority of K050.

### Missing closing elements

Add a short abstract that names the controlled study, model sizes and dataset,
the conditional pressure finding, the high-threshold recipe comparison, and
the workload-specific kernel result. Add a final synthesis that answers the
opening practical question: assess quality cost, the operations zeros affect,
and implementation overhead together. Consolidate the important limitations
there: one seed per size; fixed token budget and different 410M learning rate;
recipe rather than isolated-pressure transfer; one measured systems workload.
No new experiment is required to state those boundaries honestly.

## Exhaustive exhibit checklist

Numbers refer to the reviewed reading copy and may move after reflow.

| Exhibit | Role and verdict | Caption / readability action |
| --- | --- | --- |
| Figure 1, 14M overview, p.5 | Keep in main. Introduces distinct quality-sparsity regimes and the A0 clipping control. | Preserve approved artwork and original 16-point selection. Caption should identify settings and A0-only clipping, then give the insight. Move repeated full-validation and matching details to setup. Preserve the distinction between connecting settings and fitted Pareto envelopes. The all-variant v2 is a retained alternative, not currently embedded; do not silently substitute it. |
| Figure 2, paired effects, p.6 | Keep in main. Supplies the controlled evidence behind the overview. | Small labels are readable at enlarged view but dense at paper width. The existing two label rows do the necessary work; no redesign needed. Caption should state treatment-minus-reference, sign convention, lambda/kappa meaning, and the main conditional response; remove duplicated gate/protocol explanations. |
| Figure 3, signed densities, p.8 | Keep in main. Shows explicit reshaping and the untargeted-attention response. | Strong layout. Retain early bold notice that exact zeros are omitted and point to Table 7. Keep pooled groups, symlog axis, threshold meanings and no renormalization. Move detailed count/coverage accounting to D.1. Caption needs one interpretive sentence, especially that A4 changes untargeted attention while A7 directly gates it. |
| Figure 4, size comparison, p.9 | Keep in main. Tests persistence and separates head-share effects. | Large clipping losses compress trained curves; Table 1 makes the key comparison explicit, so preserve artwork. Caption should define both rows, common A7 denominator, and high-threshold ordering. Six decimal-like reference values are unnecessary in caption when exact values are retained elsewhere. |
| Figure 5, kernels, p.10 | Keep in main. Separates logical opportunity from measured execution. | Balanced panels and compact legend work. Explain K050, qualified incumbent, measured workload and descriptive fit; refer to ablations. “Best” must not be read as better than the attention-dense ablation. The title's auto-research framing should remain scoped to one human-guided trajectory. |
| Figure 6, architecture/ladder, p.14 | Keep in appendix as notation and topology reference. | Clear when enlarged; smallest ladder labels are compact but legible. Caption already defines L1N. Explain that “Step” is an ordering of separately trained recipes, not a curriculum, as the setup does. Main text must not rely on first seeing this figure. |
| Figure 7, operation decomposition, p.20 | Keep in appendix. Supports the distribution-to-work bridge. | Current stacked bars show contributions, not local zero rates. Caption should state the specific result: attention contribution explains A7's larger total despite reduced projection contributions. Table 8 provides precision; the figure provides composition. |
| Figure 8, all 14M clipping, p.21 | Keep in appendix as complete coverage. | Paths are intentionally faint and highly overlapping; useful for coverage rather than ranking individual checkpoints. Caption should keep 30 endpoints/300 evaluations, fixed-checkpoint paths, full loss range, and reference to machine-readable values. |
| Figure 9, all 70M clipping, p.22 | Keep in appendix. | Same conventions as Figure 8 are appropriate. Whitespace on this float page is cosmetic; avoid artwork changes just to fill a page. State 12 endpoints/120 evaluations. |
| Figure 10, all 410M clipping, p.23 | Keep in appendix. | Same role and conventions. Retain the high-loss points even though they compress the useful operating region. State 12 endpoints/120 evaluations and the complete-data location. |
| Table 1, boundary recipe contrasts, p.7 | Keep in main. Adds real value beyond Figure 4. | Caption's sign convention and matched-threshold definition work. Light spacing between model-size pairs would improve scanability without changing content. The selected thresholds must remain explicitly boundary examples; all five are in Table 6. |
| Table 2, activation sites, p.14 | Keep in appendix. Necessary lookup for site symbols and tensor placement. | Readable, informative caption. Expand QKV and FFN terms in nearby prose if not already defined. Keep actual post-RoPE operand semantics. |
| Table 3, training settings, p.16 | Keep in appendix. Supports fair-comparison and transfer limits. | Readable and concise. Define microbatch versus effective global batch in prose as currently done. Tokens-per-parameter row usefully makes the fixed-budget limitation explicit. |
| Table 4, all trained endpoints, p.17 | Keep in appendix. Necessary full result record. | Dense but legible on a full page. Add subtle spacing or midrules between size groups if reflow permits. Keep exact values. Harmonize `U_A7` with `U_arch` used in main text, or state equivalence in caption. “All 54” should remain distinct from Figure 1's 16-point selection. |
| Table 5, all paired effects, p.18 | Keep in appendix. Numerical counterpart to Figure 2. | Add small separation between contrast families if it fits. Caption should give sign convention and refer to one-seed protocol. No need to repeat main-text examples. |
| Table 6, all scale contrasts, p.18 | Keep in appendix. Prevents boundary-threshold selection from hiding reversals. | Readable. Add model-size grouping if possible. Caption is already concise and defines which signs favor A7-OL1. |
| Table 7, zero and tail mass, p.19 | Keep in appendix and cite conspicuously from Figure 3. Essential to interpret omitted zero spikes. | Caption should make clear “view tail” and “grid tail” are nested tail summaries, not disjoint terms to add together. Zero mass is separate. Preserve the all-elements denominator and seven-checkpoint/14-group distinction. |
| Table 8, operation contributions, p.19 | Keep in appendix. Exact numerical support for Figure 7 and the 16.77 pp insight. | Readable and appropriately grouped. Preserve shared model denominator and rounding notice. This table and Figure 7 have complementary roles; no need to remove one merely because they use the same data. |
| Table 9, kernel ablations, p.24 | Keep in appendix. Essential attribution of runtime gains. | Define K049 and P0 in prose or labels; currently a new reader cannot interpret them. Explain whether K049 is a predecessor and identify the provenance of P0 using verified evidence. Keep P0's four-checkpoint qualification boundary. Define GM and range as already done. |

## Final readability checks

- The result prose precedes Figures 3 and 4 by one or two pages. A carefully
  placed float barrier or shorter captions may restore local reading order;
  do not force every float with `[H]`.
- Avoid shortening the methods by making captions carry their details again.
  Shared protocol belongs in setup; a caption needs its particular estimand,
  key symbols, comparison and takeaway.
- Spell out SDPA and tensor-core matrix multiply-accumulate (MMA) before their
  first use. Explain K049/P0, and use one notation for common-A7 utilization.
- The data section says “accompanying data directory” without an external
  reader-facing location. Provide a concrete supplementary archive/directory
  name and included manifest; an anonymous public URL can be added when one
  is actually available. Do not invent a public release.
- Do not add uncertainty bars from thresholds or evaluation blocks: neither
  supplies independent training-seed replication. Keep the one-seed limitation
  visible but avoid repeating a disclaimer after every result.
- No visual collision, clipping, missing figure, or unreadable glyph was found
  in the reviewed PDF. Dense caption text and float distance are the main
  layout issues. The approved figures themselves can remain unchanged.
