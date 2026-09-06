# Fusion control and kernel-eligible sparsity

## Question

Does a model-wide logical opportunity summarize the products this kernel can
actually skip, and how much performance remains after QKV fusion alone?

## Sources, method and coverage

Same 35 checkpoints, full 338-block validation coverage, four paired modes,
three processes, 64 timing inputs, seven passes, and crossed bootstrap as
Observation 01. `../diagnostics.py` runs after timing and full validation in
the first process only, using the canonical native BF16 forward. Activation
hooks capture actual a/m/h outputs and a direct pre-hook captures the input
to Wo (z). The attention implementation is not replaced for diagnostics.
Record exact zeros, |x| <= .001 and .01 counts, RMS/L2 statistics, weight norms,
and the distribution of active features per token row at h and z, per layer.

Eligible zero products = 128 times the pooled exact-zero count of h and z.
Eligible products = 338 x 2048 x 6 x (512 + 128) x 128. Pool these integer
counts before dividing. This measures the sparse projections' native BF16
operands and differs from canonical model-wide R_model in both eligible
operations and numerical execution. It includes exact BF16 zeros without
imposing an additional clipping threshold.

The h/z projections account for 340241940480 of 6363055915008 model-wide
logical products over this validation pass (about5.35%). This structural
fraction is not a bound on runtime savings: the operation mix, implementation
overhead and throughput per operation are not uniform.

## Figure caption and legend

**Disentangling fusion gains and exploitable projection sparsity.**
(a) Latency with exact QKV/RoPE/gate/layout fusion and stock dense projections,
divided by the complete fused sparse implementation, versus canonical R_model.
This ratio includes both joint projection/residual fusion and zero skipping;
it is not a pure zero-sparsity effect. (b) The no-skip/sparse controlled latency
ratio versus count-pooled exact-zero products in the h/z projection operands
actually eligible for skipping in BF16. Markers, dose-connecting segments,
historical labels, numerical gates, and timing intervals follow Observation 01.
The no-skip kernel is deliberately untuned and is not an optimized dense rival.

## Result

Awaiting execution and reduction; no measured claim is made yet.

## Caveats

Eligible zero fraction is not a wall-clock saving fraction. Ballot/control
overhead, zero distribution among rows, dense upstream operations and the LM
head matter. No-skip and sparse kernels share arithmetic order and rounding;
their difference is execution of zero-valued FMA terms. Cross-variant slopes
remain descriptive, whereas within-checkpoint paired control directly tests
the speed contribution of the skip toggle in this implementation. Diagnostics
are not included in timed regions and cannot explain training gradient effects.

## Reproduction

Source: `../diagnostics.py`, `../10_reduce.py`, `../11_plot.py`.
Raw evidence: `../artifacts/c??-r1/diagnostics.json`; table: `../results/summary.csv`.
Output: `../figures/02-fusion-and-eligible-sparsity.pdf`.
