# Shared Pythia block and intervention sites

## Question and method

Where do the sparsification sites occur in the parallel attention and FFN
branches? This structural schematic follows the canonical ports in
[research/METHODS.md](../../../research/METHODS.md) and the
[experimental appendix](../experimental-appendix.tex).

## Coverage and caption

One Pythia/GPT-NeoX block, including the identity residual path. Green nodes
are activations, purple nodes are learned weights, and gray nodes are operators.
The branches have separate LayerNorms. Query and key sites follow RoPE;
`z` denotes concatenated attention context before the output projection.
The probabilities `p` are shown for context and are not intervention sites.

## Result and caveats

The figure locates sites; it contains no measured results. Biases, reshaping,
dropout, embeddings, and the final language-model head are omitted. The `h`
node denotes the FFN nonlinearity output; a selected gate replaces GELU there.
Recipe assignments and analytic ceilings are in
[experimental-study.tex](../experimental-study.tex).

## Source

Copied unchanged from `manuscript/artifacts/pythia-architecture-map.pdf`.
The matching [TeX source](pythia-architecture-map.tex) is retained beside the
[PDF](pythia-architecture-map.pdf); compile it with `pdflatex`.
