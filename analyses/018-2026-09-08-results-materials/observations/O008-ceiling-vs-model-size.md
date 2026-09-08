# Analytic ceiling versus model size

## Question

How does the selected-topology all-zero reach ceiling change with architecture size?

## Method

Recompute integer product counts for A0/A1-H/A4/A7 at each retained architecture using architecture_ceiling. Hold T=2048, full uncached causal attention and the dense vocabulary head accounting fixed. Plot against actual parameter counts on a log x axis; do not fit a scaling law.

## Coverage

Twelve analytic topology/architecture combinations at 14,067,712; 70,426,624; and 405,334,016 parameters. Counts are logical scalar products per one complete sequence, not pooled validation counts. This figure contains analytic counts rather than validation measurements.

## Figure and caption

[Publication PDF](../figures/08-ceiling-vs-model-size.pdf)

Architecture reach ceilings at fixed workload. The analytic maximum model-wide sparsity is shown for A0, A1-H, A4 and A7 at the three retained Pythia architectures, with actual parameter counts on a logarithmic axis and nominal size labels. Each ceiling assumes all selected activation sites are zero, counts the operations they can reach and retains the dense LM head in the denominator. OL1 variants share the corresponding gate topology ceiling. Lines connect the three computed architectures; they are not fitted scaling laws or measured sparsity curves.

## Result

A7 rises from 29.9524% to 49.4239% and 87.2452%; A4 from 12.8332% to 37.0634% and 74.7764%. A0 is zero because it selects no intervention sites; naturally occurring zeros do not redefine this reach ceiling.

## Caveats

These are analytic reach ceilings, not attainable quality-constrained targets, observed zero rates or speedups. They depend on architecture, vocabulary, topology and sequence workload. They do not apply unchanged to cached decoding. No uncertainty bars are needed for the exact stated arithmetic.

## Source script and evidence

`plots.py:ceilings`, invoked by `01_build.py`. Supporting evidence: src/sparsity_research/ceilings.py; tables/ceiling-vs-model-size.md.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).
