# Canonical logical opportunity versus measured acceleration: K036

## Question

Across all 35 existing Pythia-14M variants, does higher canonical R_model
correspond to higher full-model speedup, and how much acceleration is caused
by enabling the new implementation's zero-skipping paths?

## Sources, method, and coverage

Frozen policy: `../final-policy-001.json`, SHA256
`fa4415ad8fe8e6d0c852b60b7d095602a829b7d879c363219d7ff59860001441`.
This is K036 with prefix shortcuts disabled uniformly, not a per-checkpoint
selection. Source checkpoints, interventions, and canonical integer logical
counts are identified in Run027 `prelaunch/inputs.json`, copied into each
process manifest. These are existing random-initialization pretraining
checkpoints, seed1234, step712; no further training or pruning is performed.

Raw evidence: `../artifacts/final-matrix-001/` and all
`../artifacts/final-cNN-pR-001/` folders. All 105 processes are complete.
Evidence-030 archive SHA256
`ff4154c069274a04d027ba803db592aae932df4e795bce5b5d381ddbc0ad7d9f`;
all 7,911 exported files /305,125,906 bytes verified locally. This bundle also
retains the preceding terminal development and smoke evidence.

Hardware/runtime: one RTX5090 32GB, PyTorch2.11.0+cu128, CUDA12.8,
BF16, batch1, uncached causal T2048, all 50,304 vocabulary logits.
Resident input staging, compilation, and graph capture are outside the warm
timing window. Every comparison matches the checkpoint, inputs, execution
mode, and full-logit workload. Three fresh processes per checkpoint each
time the same 64 validation identities, selected with seed2504, over seven
paired passes. The controller uses seed2504 to randomize checkpoint order
within each replicate. There are 1,344 paired observations per variant per
comparison. Points are the geometric mean of paired ratios, equally weighted
across processes; bars show the minimum/maximum process geometric means.
They are not confidence intervals.

Every process validates all338 complete 2,048-token blocks from all500 MiniPile
validation documents:692,224 input tokens,691,886 prediction tokens,1,444-token
excluded tail. All35 new eager/graph implementations and both skip controls
have bitwise-identical logits to eager-native in all three processes. No
numerical failures, outliers, or unfavorable variants are removed. The fixed
quality gate remains atol0.25, rtol0.02, relativeL2<=0.02, and absolute pooled
loss change<=0.001nat/token, although the new modes achieve zero error.

Canonical R_model is computed from pooled source FP16 integer logical-product
counts, including the dense LM-head denominator. It is not the BF16 zero rate,
the fraction of hardware instructions removed, or a speed prediction.
`../65_reduce.py` audits source hashes, identities, coverage, numerical gates,
paired sample completeness, and diagnostic count conservation. Its output is
`../results/summary.json`; `../66_figures.py` creates the figure and the complete
`../results/per-variant.md` table. Figure/source hashes and plotting versions
are in `../results/figure-provenance.json`.

## Figure caption and legend

**Figure01. Logical opportunity, full-model acceleration, and isolated
zero-skipping contribution across all35 Pythia-14M variants.** (a) Native
graph latency divided by K036 sparse graph latency. (b) The same K036 graph
implementation with all six matmul families' skips disabled divided by its
latency with skips enabled. Panel(b) uses an explicitly expanded vertical
scale; panel(a) starts at zero. A ratio above one favors the sparse execution.
Colors and marker shapes distinguish activation/pressure families. Historical
A4+OL1@h applies pressure at h, not all four A4 sites. Each point aggregates
three fresh processes,64 validation inputs and seven paired passes per
process; bars span process geometric means, not confidence intervals. All35
new kernels qualify on all338 validation blocks in every process. Canonical
R_model uses source FP16 logical products; measured execution uses BF16.

Output: `../figures/01-rmodel-speedup-and-skip-contribution.pdf`.

## Results

| Variant | R_model (%) | Native/new graph | All skips disabled/enabled |
|---|---:|---:|---:|
| A0, c01 | approximately0 | 1.09235 | 0.96037 |
| A4, kappa0.5, c15 | 10.21554 | 1.20889 | 1.01424 |
| A4+OL1@4, kappa0.5, c20 | 12.71345 | 1.21188 | 1.01677 |
| A7, kappa0.5, c25 | 15.38681 | 1.37583 | 1.01536 |
| A7+OL1@7, kappa0.5, c30 | 27.48268 | 1.37988 | 1.01912 |

The broad association is positive, including positive net skip benefit at the
high-dose endpoints of the A4/A7 families. It is not monotonic for every
variant or proportional to R_model. Zero-skipping is a net overhead in most
variants: only c15/c20/c25/c30/c34/c35 have paired geometric mean ratios above
one. At c30 its process ratios range1.01832-1.01969; at c25,1.01449-1.01596.
At A0 the range is0.96010-0.96052.

At c30, mean paired native-to-new graph saving is0.22698ms, while enabling
skips saves0.01142ms relative to the same new implementation without skips.
Thus this ablation attributes about5.03% of the overall saved latency to the
net effect of skipping, including its overhead and interactions. It does not
attribute the entire1.37988x acceleration to sparsity. The remaining gain
includes fusion/layout/implementation effects; the abrupt A7 topology-related
step must not be interpreted as a continuous R_model effect.

## Caveats and nonclaims

This establishes a small, reproducible positive net skip contribution at high
sparsity for K036, not the full requested optimization goal. The previous
kernel remains faster in matched graph execution at the qualified high-R
endpoints; attention skipping alone still loses time. See observation02.
The kernel has paths for all six matmul families, but does not remove every
individual zero scalar product. The LM head remains dense. All-site dispatch
coverage is not full capture of canonical logical opportunity.

Different variants have different trained weights, gate overhead, sparsity
patterns, and predictive losses; this is not an equal-quality model frontier
or a causal regression on R_model. The within-checkpoint skip ablation is the
direct attribution test. One training seed, one GPU, one sequence length and
one batch size are covered. Three timing processes do not quantify training
seed variation. BF16 counters and canonical FP16 R_model are distinct
estimands. No claim or TeX is promoted to the manuscript by this observation.

The PDF is vector output with embedded fonts and has been rendered at150dpi
and visually checked; no overlap or clipping remains.

## Terminology clarification added during K050 closeout

The earlier phrase "bitwise-identical" above should be read as **zero observed
numerical discrepancy** in the recorded logit and loss checks. The harness
did not independently compare output bit patterns, including signed zeros.
This clarification changes no stored measurements or numerical qualification.
