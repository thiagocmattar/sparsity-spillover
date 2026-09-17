# Analysis 022 — A7 with no, h-only, or seven-site OL1 pressure

## Question and sources

Does h-only OL1 recover the high-threshold logical-opportunity increase of
seven-site OL1 while retaining comparable validation loss? The user requested
this matched control and then authorized all five thresholds on five A100s.

Compare the verified final checkpoints from Run 013 (A7, no pressure), Run 032
(A7, h-only OL1), and Run 014 (A7, seven-site OL1), at kappa 0, 0.01, 0.05,
0.1, and 0.5. Initialization and data-order hashes, 712 updates, and
1,493,172,224 training input tokens match. Each final validation covers all
500 MiniPile validation documents through 338 complete 2,048-token blocks;
the excluded tail contains 1,444 tokens.

Run 032 has fully recovered and verified four conditions, including all their
checkpoints; those Pods are deleted. Kappa=0.5 remains unretrieved on its
stopped volume. `02_compare_recovered_four.py` and observation 001 report an
explicitly interim comparison of the four completely verified conditions.
The primary high-threshold question is still open. `01_compare.py` deliberately
requires the complete five-condition verified cohort and has not yet run.

## Reduction

`01_compare.py` reads the three retained verification records and checks the
matched identities and intervention assignments. It produces the 15-endpoint
table in CSV, JSON, and Markdown. Source-file hashes travel with the JSON.
The ordinary final-checkpoint validation loss is reported in nats. R_model
and R_block are logical scalar-product opportunities, not measured speedups.
Sitewise exact-zero fractions are the source verifiers' pooled integer-count
fractions, never averages of batch or layer percentages.

The positive all-site increment recovered by h-only pressure is
`(R_model_h - R_model_none) / (R_model_all7 - R_model_none)` when the
denominator is positive. It is undefined otherwise and is not clipped to
[0,1]. Report the validation-loss differences alongside this descriptive
fraction; sparsity alone does not establish a useful quality trade-off.

## Interpretation limits

This is one seed at one model scale and one token budget. H-only recovery
would support sufficiency in this setting without direct Q/K/V pressure.
Failure would not isolate Q/K/V necessity: the all-site arm also pressures
a, m, and z. The pressure scalar averages six h tensors in Run 032 and
42 tensors in Run 014, so changing the target set also changes its composition
and normalization. No uncertainty interval, runtime gain, causal pathway,
manuscript edit, or promoted finding follows automatically.
