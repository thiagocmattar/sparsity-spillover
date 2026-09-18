# 005 - Paired topology-effect distributions

**Question and request.** How do the matched seven-minus-four-site differences
vary over the five kappa settings within each pressure recipe? The user requested
a 1-row, 3-column horizontal distribution figure for validation loss, model-wide
sparsity and latency, saved in this analysis's figures folder.

**Method.** Pair A7 with A4 at each kappa in {0, 0.01, 0.05, 0.1, 0.5},
separately for no pressure, OL1(h), and OL1(all). This gives 15 contrasts from
30 checkpoints, each defined as seven sites minus four sites. Use the uniform
ordinary final-validation losses retained in Analysis023, rather than its
mixed-pass archival table. Sparsity is 100 times the pooled integer zero-product
count divided by the full model-product count. Latency is the geometric mean
of 1,344 raw K050 host timings per checkpoint, converted to microseconds before
subtraction. The pairing checks checkpoint source paths, equal full-model
denominators, exact sparsity-count agreement, timing input indices and numerical
qualification. Source reduction SHA-256 hashes are retained.

**Coverage.** All models use seed 1234, the matched initialization and training
order, 712 updates and 1,493,172,224 training input tokens. Quality and sparsity
cover all 500 validation documents and 338 complete 2,048-token blocks,
692,224 input tokens; the 1,444-token tail is excluded. K050 timing uses
64 common validation inputs, seven passes and three processes, BF16,
RTX 5090, batch 1, sequence length 2,048 and all 50,304 output logits.

**Caption and legend.** Distribution of paired seven-minus-four-site effects
across five matched kappa settings in Pythia-14M. Rows show no pressure (gray),
OL1(h) (green) and OL1(all) (purple). Columns show validation-loss difference
in nats/token, model-wide logical-sparsity difference in percentage points,
and full-model K050 latency difference in microseconds per forward. Boxes span
the 25th to 75th percentiles, vertical bars indicate medians, and whiskers extend
to the minimum and maximum. Quantiles use NumPy's linear method; for five values,
the box edges are the second and fourth order statistics. All five contrasts
are shown as dots with small fixed vertical offsets for visibility. Dashed
vertical lines mark zero. Negative loss/latency and positive sparsity changes
favor seven sites on the corresponding metric. These are distributions over
fixed kappa settings, not confidence intervals or distributions over seeds.

**Result.** Seven sites have lower loss in 7/15 pairs, higher logical sparsity
in 11/15 pairs and higher observed K050 latency in all 15 pairs. Some latency
differences are very small; their sign is not a significance claim. The
equal-weight recipe means are retained below for reference; the figure's bars
show medians, not these means.

| Pressure | Mean loss difference | Mean sparsity difference (pp) | Mean latency difference (microseconds) |
| --- | ---: | ---: | ---: |
| None | +0.009257 | +1.554779 | +2.435152 |
| OL1(h) | +0.003027 | +1.869406 | +12.548403 |
| OL1(all) | -0.062976 | +2.591835 | +30.805970 |

**Caveats.** The 15 conditions share a single training seed and are not 15
independent seed replicates. The boxes do not estimate sampling uncertainty.
OL1(all) expands both gating and pressure from four to seven sites and changes
the pressure objective's target composition/normalization. The h-only latency
pairs compare Run029 A4 timings with Run033 A7 timings on different physical
GPU/host sessions; their difference cannot be attributed solely to topology.
Logical sparsity is not measured acceleration. Kappa-dependent heterogeneity,
especially the high-threshold sparsity gains, is visible in the ranges but the
individual dots do not label kappa. No manuscript claim or finding is promoted.

**Source script and artifacts.** `../05_plot_paired_topology.py` reads
`../data/results.json` and
`../../023-2026-09-17-14m-pressure-targets-paper-table/results.json`.
It writes `../data/paired-topology-effects.json` with all 15 endpoint pairs,
the differences, exact counts, source hashes and nine sets of box statistics,
and `../figures/05-14m-paired-topology-effects.pdf`.

**Verification.** All 15 pairs and 45 plotted values are present, all selected
endpoints qualify, checkpoint source paths and exact counts agree across the
two reductions, and all values are inside the displayed axes. Independent
checks reconcile the saved quartiles and means with the five sorted contrasts
per recipe. The PDF was rendered and visually inspected, its fonts are embedded,
and regeneration reproduces the PDF and JSON byte-for-byte. Figure/source number
05 leaves 04 available to the existing 70M plotting work.
