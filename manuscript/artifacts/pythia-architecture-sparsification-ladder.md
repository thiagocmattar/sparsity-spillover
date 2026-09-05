# Pythia architecture and sparsification ladder

## Question

How can the intervention-neutral Pythia block map and the proposed
sparsification ladder be presented as one publication-ready figure?

## Method and coverage

The composite places `pythia-architecture-map.pdf` above
`sparsification-ladder.pdf` on a single page. Both source PDFs are included as
vector graphics at the same displayed width. The composite adds a neutral vertical gap between the panels and preserves
each source panel's content. The architecture key sits beside its title; the
ladder key occupies a separate line beneath its title for readable lettering
at manuscript width.

The upper panel defines the shared Pythia block and the seven intervention
sites used by the ladder. The lower panel maps the eight proposed conditions
across those sites, their pressure methods, and the analytic
$\mathcal{S}_{\mathrm{model}}^{\max}$ columns for the 14M, 70M, and 410M architectures.

## Caption

**Figure caption.** Shared Pythia transformer block and sparsification ladder.
The upper panel locates the activation sites in the attention, residual, and
FFN paths. The lower panel progresses from the stock GeLU baseline to mixed
one-sided and symmetric thresholding with optional L1N or OL1 pressure, and
reports the topology-conditioned logical-product ceiling for three Pythia
architectures at `T=2,048`.

## Result and caveats

This artifact is a layout composite and introduces no new scientific result or
definition. Interpretations, operational crosswalks, analytic assumptions, and
caveats remain in the companion Markdown files for the two source panels.

## Source

- Editable source: `pythia-architecture-sparsification-ladder.tex`
- Generated artifact: `pythia-architecture-sparsification-ladder.pdf`
- Upper panel: `pythia-architecture-map.pdf`
- Lower panel: `sparsification-ladder.pdf`

## Methodology notation revision (5 September 2026)

The displayed ceiling now uses calligraphic S to match the new draft.
Its definition and numeric values are unchanged: the existing artifact field
`R_model_max_fraction` is the fraction, and the figure reports its percentage.
Observed sparsity can include natural zeros outside this selected-site reach.
General definitions are in `../draft/methodology-appendix.tex`; the
Pythia-specific derivation is now in `../draft/experimental-appendix.tex`.
The historical `../methodology.tex` remains a preserved earlier source.

The methodology review enlarged architecture labels and compacted the ladder
source so that both panels remain legible in the 5.5-inch reading layout.
Their source PDFs and this composite were rebuilt and visually checked; the
operation graph and all analytic values are unchanged.
