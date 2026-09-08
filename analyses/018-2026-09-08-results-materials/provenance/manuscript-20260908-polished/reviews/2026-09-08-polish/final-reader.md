# Final reader and design review

## Frozen state reviewed

I copied the current PDF before reviewing it. The frozen 25-page copy is
`tmp/pdfs/technical-reader-final/main.pdf`, SHA-256
`ef257635b528e69041e4443ca2d53c51a050fd513e1eaa401f88be5d3afe424b`.
All 25 pages were rendered and visually inspected. The setup/recipe table,
scale-to-kernel transition, kernel figure, and final appendix page also
received larger renders. I read the revised main TeX and captions against
that copy. This review does not independently reproduce the experiments.

The first two uncommented introduction paragraphs compare **exactly** with
`tmp/manuscript-polish-before-20260908/protected-introduction.json`.
Both checks are true; details are in the review scratch's
`protected-check.json`. No manuscript source, figure, or PDF was edited.

The root subsequently reported fixes to the abstract's scalar-multiplication
wording and appendix definitions/data pointers. Those changes are outside
this frozen copy. The new user terminology request also arrived after the
freeze; the cautions below apply to that forthcoming edit.

## Assessment

| Dimension | Earlier | Revised | Reason |
| --- | ---: | ---: | --- |
| Sentence clarity | 4/5 | 4/5 | Much more direct; results lead with insights. Some numerical sequences remain dense but accurate. |
| Narrative sequence | 4/5 | 4/5 | Abstract and conclusion now complete the argument. The result order remains coherent. |
| Figure integration | 3/5 | 4/5 | Captions now explain the findings, and the float barrier keeps scale evidence ahead of the kernel subsection. |
| Table utility | 4/5 | 5/5 | The five-row recipe table resolves the initial label burden with little space; the boundary comparison table remains useful. |
| Self-containedness | 3/5 | 4/5 | Setup, SDPA, qualification, and the selected-K050 distinction are clear. Pending appendix identifiers/data access complete the remaining small gaps. |

This is a substantial editorial improvement. The revision preserves the
original question and sequence rather than substituting a different paper.
It is ready for a final terminology and factual/compilation check; it does
not require another structural rewrite or new figure design. Submission
acceptance cannot be inferred from editorial readiness: the one-seed,
fixed-budget and single-workload limits remain real and are now disclosed.

## Critical prose findings

1. **Abstract metric wording:** the frozen version says “fraction of counted
   matrix products,” which can mean whole matrix multiplications. Use scalar
   multiplications within the counted matrix products. The root reports this
   fix has landed; verify it in the final rendered copy.
2. **The introduction now makes the study's identity clear.** Comparing
   components under matched pretraining, rather than claiming a complete
   reproduction of each prior method, resolves the main expectation mismatch.
   The protected prose flows naturally into the new third paragraph.
3. **The abstract and conclusion use the runtime ablation well.** Reporting
   the full-model gain next to the much smaller sparse-path increment keeps
   the systems result from becoming an unsupported “zeros caused 1.234x” claim.
   The attention-skipping negative result belongs in both the abstract and
   discussion because it changes the interpretation of useful sparsity.
4. **“The location of zeros matters as much as their frequency” is slightly
   stronger than necessary.** It reads like a quantitative equality. Optional
   direct replacement: “The location of zeros determines which computations
   they affect.” This is a small style improvement, not a blocker.
5. **Paired-effects prose still has the densest numerical paragraph.** Its
   A4/A7 representative increments are useful and the figure supports them.
   There is no need to add another example or expand it. A shorter final
   sentence would be preferable to introducing more numeric comparisons.
6. **The distribution subsection is the strongest bridge.** It distinguishes
   exact-zero probability, nonzero density and affected work; it then explains
   how greater FFN sparsity can coexist with lower model-wide sparsity.
   Preserve this three-step explanation during terminology edits.
7. **The scale subsection now handles the unusual 410M quality result.**
   The sentence about higher A0 loss under this budget prevents readers from
   interpreting the panel as a standard parameter-scaling result.
8. **The conclusion gives a usable answer to the introduction.** Its final
   paragraph consolidates limitations rather than scattering them after
   every observation. Do not add speculative future-work promises.

## Exhibit and page inspection

All ten figures and ten tables are present. No overlap, cropped text,
unresolved reference, broken glyph, or unreadable table cell was found in the
reviewed rendering. Bibliography entries visually render normally.

| Exhibit(s) | Revised-copy assessment |
| --- | --- |
| Table 1, recipes, p.4 | Five rows cover the eight named families clearly. The slash notation is understandable from the aligned pressure column. Its position before the first result removes reliance on a distant appendix diagram. Keep this table. |
| Figure 1, overview, p.5 | Shorter caption identifies the selected checkpoints and A0-only clipping and states the trade-off. Original approved artwork and selection remain appropriate. |
| Figure 2, paired effects, p.6 | Caption now leads with the conditional pressure result and retains the sign/parameter conventions. Small plot labels remain dense but usable; no redesign is justified. |
| Figure 3, densities, p.8 | Early bold omitted-zero notice is conspicuous. The explicit warning that filled areas on the symlog axis are not probability mass is useful. Pooling, thresholds and site interpretation remain intact. |
| Figure 4, sizes, p.9 | Caption defines both rows and the common A7 denominator. It precedes subsection 4.5, so the added barrier fixes the previous sequencing concern. |
| Table 2, boundary contrasts, p.7 | Gives the exact central comparison that is compressed in the scale panels. Keep it; full comparison table remains in appendix. |
| Figure 5, kernel, p.10 | The caption resolves “best” versus selected K050 and its faster attention-dense ablation. Runtime precision and logical-sparsity precision are explicitly distinguished. |
| Figure 6 and Table 3, architecture and sites, p.15 | Still useful as formal reference. Main Table 1 now handles the first-read recipe need. Preserve the detailed diagram and placement semantics. |
| Table 4, training settings, p.17 | Clear and compact. The tokens-per-parameter row makes fixed-budget interpretation concrete. |
| Table 5, all endpoints, p.18 | Dense but readable; all 54 records appear. Common-A7 utilization needs the same name/equivalence as the main text. No mandatory redesign. |
| Tables 6 and 7, full paired/scale comparisons, p.19 | Readable numerical audit and useful protection against selective boundary reporting. |
| Table 8, zero/tail masses, p.20 | Necessary support for Figure 3. Keep the all-elements denominator and exact-zero distinction. A one-line note that grid tails lie within the reported view tails would prevent summation mistakes, but this is optional if already explained in prose. |
| Table 9 and Figure 7, operation decomposition, pp.20-21 | Complementary precision and visual composition. The revised figure caption now states why A7's attention contribution matters. |
| Figures 8-10, complete clipping, pp.22-24 | Complete range and faint paths are appropriate for appendix coverage. Large whitespace is cosmetic; do not change approved artwork to fill float pages. |
| Table 10, kernel ablations, p.25 | Fully readable. P0/K049/MMA definitions were still absent from this frozen copy; root reports these are being completed. |

The new barrier correctly preserves the sequence
**distribution figure -> scale figure -> kernel subsection -> kernel figure
-> discussion**. Figure 3 still appears after the scale prose has started,
but both training figures precede the systems section; this is acceptable
float behavior and does not warrant hard-positioning every figure.

The conclusion begins on p.10 and ends with a short continuation on p.11,
leaving substantial whitespace before the reference page. This is a minor
reading-copy polish issue, not a scientific or template-readiness blocker.
Avoid shrinking approved artwork or type to recover that space. If the final
terminology reflow naturally resolves it, no additional action is needed.

## New terminology request: ambiguity checks

The user now requests no “gates,” no “all-zero,” and use of threshold/nonlinearity
and sparsity-ceiling terminology. A mechanical replacement would create some
incorrect sentences, so review these distinctions explicitly:

- **Threshold versus operator:** `kappa` is a scalar threshold;
  `G_{+,kappa}` and `G_{pm,kappa}` are thresholding nonlinearities. Use
  “thresholding nonlinearities” for the functions, “thresholds” for parameter
  values, and “thresholded sites” for placement. A threshold itself does not
  replace GELU; the selected nonlinearity does.
- **Recipe table:** a heading such as “Nonlinearity and sites” is more precise
  than simply replacing “Gates” with “Thresholds.” Its ReLU and stock-GELU
  entries are nonlinearities without a swept threshold.
- **Site counts:** “four-site intervention” and “seven-site intervention” avoid
  suggesting four or seven distinct scalar thresholds. The experiment applies
  a common kappa per condition with different nonlinearities at different sites.
- **Sparsity ceiling:** removing the old phrase must not remove the mathematical
  construction. Define the fraction of counted scalar multiplications reached
  when every activation at the selected sites is zero. Retain the distinction
  between this architectural reference and sparsity achievable at acceptable
  loss. Natural zeros outside the selected sites can still affect the observed
  metric; naming the reference a ceiling does not make that distinction vanish.
- **Untargeted attention:** A4's internal q/k/v are directly unthresholded and
  unpressured, but its attention input a is targeted. Keep naming q/k/v when
  discussing spillover rather than saying A4 never targets attention at all.
- **Common-A7 utilization:** the bottom-row denominator is the A7 sparsity
  ceiling for every recipe within a model size. Do not let the terminology
  edit restore recipe-specific denominators.
- **Figure labels and cross-references:** preserve operator symbols and exact
  site semantics while changing labels. Re-render the architecture diagram
  and paired-effects figure if any embedded word labels change.

After those targeted edits, inspect the final PDF again for stale terminology,
figure/caption consistency and reflow. The frozen copy supports the assessment
above; it does not verify modifications made after the freeze.
