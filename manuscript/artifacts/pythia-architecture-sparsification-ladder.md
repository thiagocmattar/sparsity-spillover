# Pythia architecture and sparsification ladder

## Question

How can the intervention-neutral Pythia block map and the proposed
sparsification ladder be presented as one publication-ready figure?

## Method and coverage

The composite places `pythia-architecture-map.pdf` above
`sparsification-ladder.pdf` on a single page. Both source PDFs are included as
vector graphics at the same displayed width. Their internal notation, values,
legends, colors, borders, and typography are unchanged; the composite adds only
a neutral vertical gap between the panels.

The upper panel defines the shared Pythia block and the seven intervention
sites used by the ladder. The lower panel maps the eight proposed conditions
across those sites, their pressure methods, and the analytic
`R_model^max` columns for the 14M, 70M, and 410M architectures.

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
