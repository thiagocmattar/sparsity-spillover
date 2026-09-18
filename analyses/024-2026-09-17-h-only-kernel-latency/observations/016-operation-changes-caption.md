# Figure 4: Absolute operation contributions to model-wide sparsity

PDF: [04-operation-sparsity-changes.pdf](../figures/04-operation-sparsity-changes.pdf).
Proposed manuscript placement: [Section 4.4](../../../manuscript/draft/training-results.tex),
`sec:activation-reshaping`, alongside/replacing the operation portion of
`fig:activation-reshaping`.

## Question, method, and coverage

How much does each operation contribute to the model-wide sparsity of each
retained threshold/pressure recipe? Twelve Pythia-14M checkpoints cover κ=.05/.5,
T4/T7, and P0/Ph/Pall. The previous difference plot involved these same twelve
checkpoints; the user's revision displays their absolute levels as six bars
per panel. No absolute-value transformation of the previous signed differences
is used.

Every segment is `C_o = 100 * N_zero,o / N_model`, using exact pooled
zero-product counts and the common full-model denominator of
6,363,055,915,008 products. Full validation covers 338 complete 2,048-token
blocks from all 500 MiniPile validation documents, excluding the 1,444-token
tail. Counters cover all six layers, use the canonical FP16 logical pass,
exclude future causal QK/PV positions, and retain the dense LM head in the
denominator. The six operation numerators sum exactly in integers to the
checkpoint's block zero-product numerator; their contributions sum to the
displayed model-wide sparsity within floating-point precision. No segment is
inferred from aggregate sparsity, activation percentages, or latency.

The two panels retain κ=.05/.5 with a shared 0-30 pp scale. Each panel orders
T4/P0, T4/Ph, T4/Pall, T7/P0, T7/Ph, T7/Pall. The style follows Figure 3:
DejaVu Sans; title/panel/axis/tick/legend sizes 14/11.5/11/10/11 points;
thin axes and light horizontal grid. The 10.8 × 4.6-inch figure has one
shared operation legend, without numerical annotations or net-change markers.

Colours and stack order exactly follow the operation panel in the requested
[manuscript reference](../../../manuscript/draft/figures/pressure-scope/05-site-structure.pdf)
and its [source script](../../021-2026-09-10-training-results-figures/pressure-scope/figures.py):
QKV `#9eaec2`, QK `#df985c`, PV `#f0c68b`, attention output `#76969c`,
FFN up `#817ca9`, and FFN down `#b5acd0`.

## Publication caption

**Operation contributions to model-wide sparsity on Pythia-14M.**
Stacked bars show absolute contributions at (a) κ=0.05 and (b) κ=0.5,
on a shared scale. Each panel includes four- and seven-site thresholding with
no pressure (P0), h-only OL1 (Ph), and OL1 at all selected threshold sites
(Pall). Colour identifies the counted operation. Each segment is
`C_o = 100 N_zero,o / N_model` in percentage points, using the same full-model
product denominator, including the dense LM head. The stack total equals the
checkpoint's model-wide sparsity expressed as a percentage. Counts are pooled
over all six layers and 338 complete validation sequences; the excluded tail
contains 1,444 tokens. Each recipe has one seed and one final checkpoint.
These are logical zero-product opportunities, not measured runtime savings.

## Result and proposed manuscript writing

> At κ=0.5, T4/P0, T4/Ph and T4/Pall reach 10.216%, 10.227% and 12.713%
> model-wide sparsity, respectively. The corresponding T7 values are 15.387%,
> 16.664% and 27.483%. QK and PV together contribute 16.773 pp under T7/Pall,
> compared with 6.385 pp under T7/Ph and 0.057 pp under T4/Pall. Thus the large
> high-threshold seven-site total is concentrated in internal-attention
> products. At κ=0.05, QK/PV contribute 1.753 pp under T7/Pall versus 1.069 pp
> under T7/Ph, but the total sparsity is lower (9.863% versus 10.126%) because
> projection contributions decrease. The operation decomposition distinguishes
> the composition of sparsity from its aggregate amount.

## Caveats and provenance

These are descriptive checkpoint levels, not estimates across independent
training seeds or causal attribution to a propagation path. Pressure scope
changes both the target set and the equal-tensor pressure normalization.
Logical product counts do not establish whether zeros align with kernel
skipping predicates. No 70M operation figure or new measurement is inferred
from these 14M bars. Proposed manuscript writing is retained here only;
manuscript files are unchanged by this revision.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[source checkpoint counters](../data/paper-checkpoints.json), and
[twelve integer decompositions and totals](../data/paper-derived.json).

Regenerate only this figure with
`13_rebuild_paper_figures.py --only-operation-contributions`.
Seven focused evidence tests pass, including exact per-operation count
identity, the full-model denominator, all twelve recipe/checkpoint identities,
and reconciliation of every stack with model-wide sparsity. The one-page PDF
was rendered at 1,900 pixels and visually inspected; all three listed fonts
are embedded. Other figure PDFs and their derived records are unchanged.
