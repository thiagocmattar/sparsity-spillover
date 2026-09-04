# O003 - Attention opportunity and architecture-dependent totals

## Question and method

Which products explain the difference between pressured A4 and A7 at
kappa=.5, and why does the model-wide total grow from 14M to 70M?
Use four existing endpoint files. Pool activation counts and multiplication
counts before division across all 338 validation blocks (tail 1,444 tokens).
Each bar segment is operation zero-product count / complete model count.
The head receives its full dense count and zero opportunity credit.

## Figure and caption

[Figure PDF](../figures/03-sites-and-operations.pdf).
Left: exact-zero activation percentage at a,m,h,q,k,v,z; q/k are post-RoPE.
Right: six operation contributions, with QKV and feed-forward/output
projections separated from QK and PV. Every label A4/A7 refers here to its
respective all-gated-site OL1 recipe at kappa=.5.

## Result

- At 14M, QK+PV contribute 16.773 pp of A7's 27.483% total, versus .0566 pp
  of A4's 12.713% total.
- At 70M, QK+PV contribute 10.424 pp of A7's 40.602% total, versus .0112 pp
  of A4's 35.596% total.
- A7's R_block falls from 91.75% to 82.15% from 14M to 70M. Its R_model
  nevertheless rises because the block share grows from 29.95% to 49.42%.
  R_model = block_share * R_block exactly under the declared denominator.
- All four high-threshold endpoints have h and z exact-zero fractions above
  99.8%. Very high opportunity therefore coexists with heavily suppressed
  branch representations; a statically reduced/capacity-matched comparator
  would help assess their value.

## Caveats and provenance

The data explain where endpoint zeros occur. They provide neither a training
mechanism nor cross-seed evidence. Architectural denominator changes make
an increasing R_model insufficient to establish stronger sparsity learning
with scale. Dense head cost depends on full-sequence workload and vocabulary.

Source: [evidence.py](../evidence.py), [plots.py](../plots.py),
[01_build.py](../01_build.py), plus the hashed raw counters recorded in
[figure_data.json](../figure_data.json). Table: architecture.tex.
