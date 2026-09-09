# Final visual QA: pages 1-13

Reviewed frozen PDF:
`tmp/pdfs/manuscript-polish/review-final.pdf`

SHA-256, independently checked:
`c6b07a23495ca90fe237a5332d6d20cee52be9b8b17b704c55b826455f3b0d5d`

Coverage: **every page from 1 through 13**, using the fresh 110-dpi
`final-01.png` through `final-13.png` renders and extracted text from that
same PDF. Pages 14-26 are outside this delegated review and are being
inspected by the root. No source, PDF or artwork was edited.

## Result

**Pass: no material visual or caption-consistency defect found in pages 1-13.**

- Pages 1-4: title, abstract, introduction, related work, methodology and setup
  are legible. The first two introduction paragraphs visibly retain the
  protected text; the earlier exact-source check is recorded in
  `final-reader.md`. The abstract now defines the metric using scalar
  multiplications. The recipe table uses “Nonlinearities,” fits its columns,
  and matches the site/pressure definitions in the text.
- Pages 5-6: overview and paired-effect figures retain readable labels and
  separate caption/text blocks. Captions match the selected 16 trained
  overview checkpoints, A0-only clipping, 29 paired comparisons and their
  treatment-minus-reference convention. Parameter and operator symbols render
  correctly. Dense paired-effect labels remain usable at the intended width.
- Pages 7-9: the six-row boundary table agrees with the stated cross-size
  ordering. The density caption conspicuously states that exact zeros are
  omitted, explains pooling and the symlog axis, and matches the visible
  threshold lines. The scale caption correctly identifies A7 as the common
  normalization reference. Both training figures precede the kernel subsection.
- Pages 9-10: kernel text, plot and caption agree on the selected K050 result,
  30-checkpoint coverage, descriptive fit, workload and faster attention-dense
  ablation. “Best” is explained in the caption. Neither figure panel is clipped,
  and the legend remains clear below the panels.
- Pages 10-11: the discussion's stronger argument connects the comparisons
  without asserting seed replication or a demonstrated causal spillover route.
  Paragraph continuation across the page boundary remains readable. Whitespace
  on page 11 is cosmetic and requires no layout change.
- Page 12: all references on the page are legible, including URL wrapping;
  no entry is clipped by the footer.
- Page 13: nonlinearity definitions, OL1 equations and the first accounting
  definitions are legible. Equation (1), its cases, and inline symbols render
  correctly. Section headings have body text beneath them.

Across the reviewed pages I found no overlap, cropped glyph, orphaned heading,
single-line paragraph fragment, missing figure/table, unresolved cross-reference,
or caption separated from its figure. Several sentences continue across page
boundaries or around top floats; these are ordinary, readable breaks and do
not justify forcing new float positions. The inspected text and figure labels
use the requested threshold/nonlinearity and sparsity-ceiling terminology.

No further edit is requested from this bounded visual review. Final delivery
still depends on the root's separate pages 14-26 inspection and final artifact
verification.
