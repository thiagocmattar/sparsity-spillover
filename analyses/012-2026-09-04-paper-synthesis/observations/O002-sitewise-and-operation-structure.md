# O002 - Sitewise zeros and operation-level opportunity are not interchangeable

## Question

How is the measured model-wide opportunity distributed across activation sites
and transformer operations at the selected low- and high-dose endpoints?

## Method and coverage

For A4-OL1 and A7-OL1 at `kappa` 0 and 0.5, Analysis 012 reads the source
activation and logical-product diagnostics named and SHA-256 pinned by Analysis
011. Site values are exact-zero counts pooled over every layer and all 338
complete validation blocks. Each operation contribution is its measured
zero-product count divided by the complete model denominator, including the
dense LM head; the six stacked contributions reconcile exactly to `R_model`.

## Result

The same local zero percentage has different model-wide value depending on its
site and architecture. A4-OL1 at `kappa=0.5` drives `a`, `m`, `h`, and `z`
near saturation but leaves Q/K and V essentially dense. Its measured
`R_model` rises from 12.7134% at 14M to 35.5962% at 70M and 71.5914% at 410M,
primarily through QKV, W1, W2, and the attention output projection. The
architecture-dependent denominator, not a monotonic increase in every sitewise
zero rate, creates much of this cross-scale increase.

A7-OL1 at `kappa=0.5` opens the two attention products directly. QK plus PV
contribute 16.7733, 10.4238, and 11.0639 percentage points of `R_model` at 14M,
70M, and 410M. The corresponding pooled Q/K/V zero masses are
93.545/94.541/98.713%, 71.531/72.123/86.939%, and
77.806/80.611/93.960%: neither the activation fractions nor the attention
contributions vary monotonically with parameter count.

At 410M, high-dose A7-OL1 has lower branch-site sparsity than high-dose A4-OL1
at `a` and `m`, yet reaches higher `R_model` (80.6155% versus 71.5914%) because
it additionally creates zeros in QK and PV operands. This is the paper's
cleanest concrete example of why a local or averaged activation sparsity number
cannot stand in for model-wide opportunity.

## Figure captions

**Figure 3.** Count-pooled exact-zero activation mass by site for A4-OL1 and
A7-OL1 at `kappa` 0 and 0.5. Q and K are measured post-RoPE at the actual QK
operands. Values below 0.01% are displayed as `<.01`.

**Figure 4.** Additive contributions to measured `R_model` from the six counted
transformer operation families. Each colored segment is an actual zero-product
count divided by the full model denominator; stacks sum to the labeled
`R_model`. The figure describes logical opportunity, not sparse-kernel work or
wall-clock time.

## Caveats and next questions

These endpoint aggregates do not reveal tokenwise mask structure, batch-to-batch
variance, co-occurrence between sites, or whether zero patterns are favorable
to a particular sparse layout. Layerwise rows exist in the diagnostics and are
a natural next analysis, but the current artifacts do not retain per-token or
per-validation-batch masks. The cross-scale recipes remain one-seed outcomes,
and A4-OL1/A7-OL1 does not isolate attention gate placement from attention-site
OL1 pressure.

The non-monotonic Q/K/V scale pattern raises testable questions: does additional
training stabilize where attention sparsity concentrates; are the zero masks
token-, head-, or layer-specialized; and does model quality depend more on mask
location than on aggregate mass? Answering those questions requires a
predeclared diagnostic that retains token/head/layer structure rather than only
pooled marginals.

## Provenance

- Source script: `../01_build.py`
- Count-reconciled reduction: `../figure_data.json`
- Figures: `../figures/03-sitewise-zero-structure.pdf` and
  `../figures/04-operation-contributions.pdf`
