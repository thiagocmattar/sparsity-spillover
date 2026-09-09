# Second methodology review: revised sources and rendered usability

Independent review, 2026-09-05. Re-read the revised main methodology, appendix, wrapper, and provenance notes. Inspected rendered pages 4, 6, and 8 at the supplied reading width: the main figure, appendix site table/OL1 definition, and architecture source footnote/evaluation scope. Read-only PDF inspection used the PDF skill. No manuscript edits or experiments.

## Same rubric and revised score

Five equal criteria. Anchors: 1 = blocking, 2 = major revision, 3 = competent but uncompetitive, 4 = strong, 5 = exceptional. Maximum 25. This evaluates the present methodology stage, not the future empirical claims.

| Criterion | First pass | Second pass | Assessment |
|---|---:|---:|---|
| Citation/definition provenance | 4 | 5 | Explicit immutable architecture links now match local run identities; operational definitions and contextual citations remain separated. |
| Main/appendix progression | 4 | 4 | The figure now follows the reach explanation on page 4 rather than occupying a detached exhibit page. |
| Notation/figure consistency | 4 | 3 | Symbols, labels, percentages, and caption meanings are correct. Actual rendering reveals very small figure lettering that source-only review could not assess adequately. |
| Workload/systems interpretation | 5 | 5 | Exact products, denominator scope, natural zeros, topology reach, and full-sequence evaluation remain precise. |
| Reader utility/economy | 4 | 4 | Compact methods and detailed appendix have a useful division; no redundant chart was added. |
| **Total** | **21/25** | **21/25** | **Provenance improved; rendered figure legibility is the remaining limitation.** |

The equal totals do not indicate unchanged quality: the first review judged figure sources while this review examined the actual scaled PDF. I have not raised scores merely because edits were made.

## Corrections resolved

1. **Architecture provenance:** the appendix footnote identifies immutable configuration revisions. I matched the 14M, 70M, and 410M revision hashes to [Run 004](../../../../runs/004-2026-08-29-pythia14m-full-pass-l1n/config.yaml), [Run 018](../../../../runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init/config.yaml), and [Run 019](../../../../runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/config.yaml), respectively. This verifies the local provenance of the pins; it is not a claim that I successfully fetched every remote URL. The footnote distinguishes architecture configurations from released weights.
2. **Reading path:** page 4 contains the reach explanation followed immediately by the graph/ladder and its caption. The figure now has a clear methods role.
3. **Caption terminology:** L1N is decoded as naive L1; historical “corrected” wording is removed. Gate and pressure differences remain explicit.
4. **Table caption and layout:** the site table caption is above the table. Page 6 shows clean column alignment, complete labels, and no clipped shapes or port descriptions.
5. **Scientific wording:** matching is explicitly within model size; the v-to-output example now gives its zero-context reason. Finite threshold domain, tensor-cardinality meaning, and attention-count scope are explicit.

## One remaining figure usability issue

The composite is geometrically clean, but its legend, site labels, and especially the ladder's ceiling values are small at the 5.5-inch text width. The source table is approximately 20 cm wide and uses footnote-sized lettering before reduction; its smallest text is roughly 5–6 pt in the embedded figure. The supplied page rendering confirms that users need zoom to inspect the details comfortably. The caption is substantially larger than the information it explains.

This is not clipping or a notation error. It is a normal-size reading/print issue, and should be corrected before treating the figure as submission-ready. Prefer enlarging the figure's internal type and simplifying low-value visual furniture, or allowing a wider figure area at the eventual template stage. Do not add another ceiling chart to compensate. Preserve the seven site columns, pressure identity, and three analytic ceiling columns; those are the scientifically useful information.

The dots also use color alone to encode gate family. This remains a secondary grayscale/accessibility limitation. If the graphic is revised for legibility, redundant short symbols can address it in the same edit; it does not require a new scientific figure.

No other material source or literature correction is needed in this pass. Pages 6 and 8 have readable equations/prose and no visible overlap. The configuration footnote is small but conventional and the linked names are legible.

## Final interpretation checks

- The actual embedded PDF uses calligraphic S, and its values include percent signs. The caption correctly presents displayed ceilings as `100 Smax %`.
- Observed S and selected-site Smax remain different objects. A0's possible natural zeros and the absence of a universal observed-S upper bound are stated in the appendix.
- Full-sequence uncached evaluation, dense-head counting, excluded operations, integer pooling, and paired loss/zero measurements remain consistent with [METRICS](../../../../research/METRICS.md) and [DATA](../../../../research/DATA.md).
- The main text does not claim OL1 preserves loss or isolates gate placement when pressure targets also change.
- The draft adds no new runtime finding, cross-family comparison, universal training schedule, or unapproved experimental coverage.
- The existing ladder plus the new reach derivation answer the architecture-dependence question adequately. A new chart remains unnecessary.

The methodology content is ready for the next drafting stage. Figure lettering still needs attention before publication layout is finalized; result-specific schedules and cohort coverage appropriately remain for the later results/configuration stage.
