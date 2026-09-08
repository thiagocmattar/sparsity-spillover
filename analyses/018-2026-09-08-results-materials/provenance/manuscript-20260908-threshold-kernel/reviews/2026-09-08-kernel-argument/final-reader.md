# Final reader and visual QA

Frozen PDF reviewed:
`tmp/pdfs/manuscript-kernel-argument/review.pdf`

Independently verified SHA-256:
`dbbd594907b46a79e2f4bf668b925a20969181f02d66a8cedc8f20e3f2eb2e1a`

Coverage: **pages 3-5, 10-13 and 26-27**, individually inspected using the
corresponding fresh 110-dpi `final-NN.png` renders. I also read current
`methodology.tex`, `kernel-autoresearch.tex`, and the relevant protocol,
qualification, ablation and data sections of `results-appendix.tex`.
This is a bounded reader/rendering review; it does not independently
reconstruct the experimental measurements. No TeX, figure or PDF was edited.

## Result

**Pass. No material layout or readability defect found in the reviewed pages.**

The new reasoning is connected and understandable:

1. Fixed elementwise thresholds avoid ranking or estimating a prescribed
   retained count during selection.
2. The hypothesis is that training can adapt the distribution to that fixed
   nonlinearity, with pressure providing an additional control.
3. Successful upstream kernels do not automatically support the target
   shapes, signs and numerical contract; the original primitive and adapted
   P0 are distinguished.
4. The human-guided coding-agent loop motivates architecture-specific
   specialization, and the matched retrospective shows its measured progress.
5. The checkpoint association connects sparsity to runtime without treating
   the fit as causal attribution.
6. Fusion and skip ablations identify the conditional sparse-path benefit;
   operand-fragment granularity explains why many scalar zeros need not make
   an instruction skippable, and why even skipped instructions can lose time.

This develops the user's requested argument without changing the approved
figure or overstating the P0 subset as a full-cohort upstream benchmark.
The fixed-threshold adaptation paragraph is explicitly a hypothesis, while
the later sections report measured outcomes.

## Page-level checks

| Pages | Check |
| --- | --- |
| 3 | The new top-k motivation paragraph flows into the formal intervention definition. Citations and both thresholding equations render correctly. The inline symmetric expression wraps across a line but remains clear and unclipped. The hypothesis paragraph and OL1 definition fit without an orphaned heading. |
| 4 | Sparsity and ceiling equations remain legible. The experimental setup and five-row recipe table fit cleanly. Table headings, operators, site lists and pressure assignments agree with the prose. |
| 5 | Quality-sparsity, paired-effect and distribution paragraphs remain readable after reflow. No sentence is obscured by a figure or table. Referenced evidence appears later in the document as expected. |
| 10 | The whole revised kernel subsection forms a coherent unit. The 4/30 P0 qualification boundary, 42-proposal retrospective, K050 30-checkpoint result and sparse-path ablations are visibly distinguished. Model/hardware/workload details are legible. No dense paragraph collides with a page boundary or footer. |
| 11 | Figure 5 and its caption remain together. The two panels, legend, axes and symbols are readable. The caption explains the selected K050 implementation, descriptive fit and faster attention-dense ablation. Discussion begins after the figure, with adequate spacing. |
| 12 | The discussion continuation is readable. The scope paragraph clearly limits transfer to the measured workload. The extra whitespace is cosmetic and requires no fitting or figure change. |
| 13 | References are legible and remain inside the page boundary. URL wrapping and italic titles are intact. |
| 26 | Upstream compatibility, adapted P0, cohort filtering, qualification tolerances, timing, K049/K050 differences and the MMA predicate are readable. Math signs and symbols render correctly. The fragment predicate uses either-zero semantics consistently with the main text. |
| 27 | Table 10 is complete, legible and consistent with the main K050/P0 claims. The qualified-subset warning remains explicit. Supplementary paths and filenames render correctly, and the data-release boundary is clear. |

No overlap, cropped glyph, unresolved citation/reference, missing exhibit,
or orphaned heading was found in this coverage. I do not request figure
changes or compression to remove the extra reading-copy page.

## One optional precision edit

Appendix D.4 currently says:

> The measurements establish no profitable joint-zero pattern or break-even threshold.

This could be read as proof that no such pattern exists. A clearer statement
of the evidence boundary is:

> The measurements do not identify a profitable joint-zero pattern or break-even threshold.

The surrounding paragraph already limits the conclusion to measured 14M
shapes and hardware, so this is a small wording clarification rather than
a material defect in the current argument. The root was informed; no source
change was made by this reviewer.

The reviewed PDF is suitable for the root's final provenance/build closeout,
subject to its separate inspection of the remaining pages. This report
verifies the frozen hash above, not any later rebuild.
