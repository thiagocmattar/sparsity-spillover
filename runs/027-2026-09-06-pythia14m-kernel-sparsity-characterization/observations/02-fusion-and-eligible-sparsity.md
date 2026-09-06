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
imposing an additional clipping threshold. These are native-path operand
counts, not instrumented counts of the CUDA kernel's executed skips. Candidate
hidden states may differ, especially for numerically unqualified checkpoints;
the skip-toggle timing comparison, rather than this proxy axis alone, measures
the skipping benefit directly.

The h/z projections account for 340,241,940,480 of 6,363,055,915,008 model-wide
logical products over this validation pass (about 5.35%). This structural
fraction is not a bound on runtime savings: the operation mix, implementation
overhead and throughput per operation are not uniform.

## Figure caption and legend

**Disentangling fusion gains and exploitable projection sparsity.**
(a) Latency with exact QKV/RoPE/gate/layout fusion and stock dense projections,
divided by the complete fused sparse implementation, versus canonical R_model.
This ratio includes both joint projection/residual fusion and zero skipping;
it is not a pure zero-sparsity effect. (b) The no-skip/sparse controlled latency
ratio versus count-pooled exact-zero products in native BF16 h/z projection
operands that are eligible for this kernel's skipping rule. Markers, dose-connecting segments,
historical labels, numerical gates, and timing intervals follow Observation 01.
The no-skip kernel is deliberately untuned and is not an optimized dense rival.

## Result

Among the six qualified checkpoints, sparse execution is 1.1746-1.2532x faster
than QKV fusion with stock dense projections, and 1.2177-1.5310x faster than
the identical joint-fusion implementation with skipping disabled. These are
different comparisons, not interchangeable attribution measures.

For A7+OL1@7 kappa 0.5, no-skip/sparse is 1.236452x, 95% CI
[1.222202, 1.260107]. The paired arithmetic mean latency difference is
0.486188 ms, versus 1.668512 ms for native minus sparse: a 29.14% share of
the net measured saving. The net non-skipping implementation difference is
1.182324 ms. This is a control-specific decomposition, not a hardware FLOP
decomposition or proof that the no-skip implementation is optimal.

A1-H+OL1 lambda 0.1 qualifies at only 3.338579% canonical R_model and 62.4257%
native BF16 h/z eligible zeros, yet its skip-toggle ratio is 1.531043x. The
five qualified high-kappa gated models have roughly 99.86-99.95% eligible
zeros and toggle ratios near 1.22-1.24x. Thus even eligible zero fraction alone
does not determine the end-to-end ratio across different topologies: product
location/distribution and the rest of the execution path also matter. The
controlled toggle supports a real skipping benefit, without a universal
monotone sparsity-to-speed law.

The arithmetic attribution share for A1-H+OL1 lambda 0.1 is above 100%
(108.27%) because its untuned no-skip fused implementation is slower than
stock eager. This is not an impossible saving: skipping offsets a negative
non-skipping implementation difference. Do not interpret these shares as
bounded physical work fractions.

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
